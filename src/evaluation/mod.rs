//! Tournaments: the agent plays every named game against every named
//! opponent strategy for the run's declared `evaluation.episodes`, and the
//! results are scored (`metrics`). A game marked for self-play puts the
//! agent's own kind in the opponent's seat; one marked cross-model puts the
//! declared opponent model there.

pub mod metrics;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;

use crate::agent::brama::Brama;
use crate::agent::{ModelAgent, StrategyAgent};
use crate::env::{Agent, Environment, Reset, RoundResult};
use crate::error::{Error, Result};
use crate::game::{Game, Library, OpponentMode};
use crate::settings::Settings;

/// Who holds a seat in a tournament.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatSpec {
    /// A model behind Brama, by route.
    Model(String),
    /// A library strategy, by name.
    Strategy(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct EpisodeResult {
    pub player_score: f64,
    pub opponent_score: f64,
    pub rounds_played: usize,
    pub cooperation_rate: f64,
    pub history: Vec<RoundResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpponentResult {
    pub total_player_score: f64,
    pub total_opponent_score: f64,
    pub mean_cooperation_rate: f64,
    pub episodes: Vec<EpisodeResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GameResult {
    pub name: String,
    pub opponent_mode: OpponentMode,
    pub opponents: BTreeMap<String, OpponentResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Tournament {
    pub seed: u64,
    pub agent: SeatSpec,
    pub episodes_per_pairing: usize,
    pub total_episodes: usize,
    pub games: BTreeMap<String, GameResult>,
    pub metrics: metrics::Metrics,
}

/// Whether a played move is cooperative. In an amount game (every move
/// carries an amount) a move is cooperative when at least as many of the
/// game's moves lie below it as above it; otherwise it is cooperative when it
/// carries the base game's first, cooperative move.
pub fn cooperated(game: &Game, played: &str) -> bool {
    let amounts = game.base_actions.iter().all(|base| crate::game::amount(base).is_ok());
    if !amounts {
        return game.cooperated(played);
    }
    let Some(base) = game.base_move(played) else {
        return false;
    };
    let Some(index) = game.base_actions.iter().position(|listed| listed == base) else {
        return false;
    };
    let (below, rest) = game.base_actions.split_at(index);
    rest.split_first().is_some_and(|(_, above)| below.len() >= above.len())
}

pub struct Runner {
    library: Arc<Library>,
    settings: Arc<Settings>,
    agent: SeatSpec,
    opponent_model: Option<String>,
    brama: Option<Arc<Brama>>,
}

impl Runner {
    /// A runner for `agent`; `opponent_model` holds the opponent's seat in a
    /// cross-model game. Brama is reached only when a seat is a model.
    pub fn new(settings: Arc<Settings>, agent: SeatSpec, opponent_model: Option<String>) -> Result<Self> {
        let needs_brama = matches!(agent, SeatSpec::Model(_)) || opponent_model.is_some();
        let brama = match needs_brama {
            true => Some(Arc::new(Brama::from_env()?)),
            false => None,
        };
        Ok(Self {
            library: Arc::new(Library::standard()),
            settings,
            agent,
            opponent_model,
            brama,
        })
    }

    fn seat(&self, spec: &SeatSpec, game: &Game, seed: u64) -> Result<Box<dyn Agent>> {
        match spec {
            SeatSpec::Model(route) => {
                let brama = self.brama.clone().ok_or_else(|| Error::Config("a model seat needs Brama".to_owned()))?;
                Ok(Box::new(ModelAgent::new(brama, route, &self.settings)?))
            }
            SeatSpec::Strategy(name) => Ok(Box::new(StrategyAgent::new(name, game.clone(), &self.settings, seed)?)),
        }
    }

    pub fn run(&self, games: &[String], strategies: &[String]) -> Result<Tournament> {
        if games.is_empty() || strategies.is_empty() {
            return Err(Error::Usage("a tournament needs at least one game and one opponent strategy".to_owned()));
        }
        let episodes = self.settings.section("evaluation")?.count("episodes")?;
        let mut environment = Environment::new(self.library.clone(), self.settings.clone())?;
        let seed = environment.seed();
        let mut results = BTreeMap::new();
        for key in games {
            let game = self.library.build(key, &self.settings)?;
            let mut opponents = BTreeMap::new();
            for strategy in strategies {
                let mut played = Vec::new();
                for _ in std::iter::repeat(()).take(episodes) {
                    played.push(self.episode(&mut environment, &game, strategy, seed)?);
                }
                let rates: Vec<f64> = played.iter().map(|episode| episode.cooperation_rate).collect();
                let label = match game.opponent_mode {
                    OpponentMode::Strategy => strategy.clone(),
                    OpponentMode::SelfPlay => "self_play".to_owned(),
                    OpponentMode::CrossModel => "cross_model".to_owned(),
                };
                opponents.insert(
                    label,
                    OpponentResult {
                        total_player_score: played.iter().map(|episode| episode.player_score).sum(),
                        total_opponent_score: played.iter().map(|episode| episode.opponent_score).sum(),
                        mean_cooperation_rate: rates.iter().sum::<f64>() / rates.len() as f64,
                        episodes: played,
                    },
                );
            }
            results.insert(key.clone(), GameResult { name: game.name.clone(), opponent_mode: game.opponent_mode, opponents });
        }
        let total_episodes = results
            .values()
            .flat_map(|game: &GameResult| game.opponents.values())
            .map(|entry| entry.episodes.len())
            .sum();
        let metrics = metrics::compute(&results);
        Ok(Tournament {
            seed,
            agent: self.agent.clone(),
            episodes_per_pairing: episodes,
            total_episodes,
            games: results,
            metrics,
        })
    }

    fn episode(&self, environment: &mut Environment, game: &Game, strategy: &str, seed: u64) -> Result<EpisodeResult> {
        let opponent: Option<Box<dyn Agent>> = match game.opponent_mode {
            OpponentMode::Strategy => None,
            OpponentMode::SelfPlay => Some(self.seat(&self.agent, game, seed)?),
            OpponentMode::CrossModel => {
                let route = self.opponent_model.clone().ok_or_else(|| {
                    Error::Usage(format!("{} is played against another model: name it with --opponent-route", game.key))
                })?;
                Some(self.seat(&SeatSpec::Model(route), game, seed)?)
            }
        };
        let request = Reset {
            game: game.key.clone(),
            strategy: Some(strategy.to_owned()),
            rounds: None,
            episode_id: None,
        };
        let mut agent = self.seat(&self.agent, game, seed)?;
        let mut observation = environment.reset(&request, opponent)?;
        while !observation.done {
            let action = agent.act(&observation)?;
            observation = environment.step(&action)?;
        }
        let rounds = observation.history.len();
        let cooperative = observation.history.iter().filter(|round| cooperated(game, &round.player_action)).count();
        Ok(EpisodeResult {
            player_score: observation.player_score,
            opponent_score: observation.opponent_score,
            rounds_played: rounds,
            cooperation_rate: cooperative as f64 / rounds as f64,
            history: observation.history,
        })
    }
}
