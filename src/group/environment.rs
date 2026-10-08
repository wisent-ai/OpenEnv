//! The group environment: seat zero is the agent, stepped one move at a
//! time; every other seat is played by a group strategy or another agent in
//! the same step. In a free-chat game each seat's action may carry a
//! `message`, and every seat sees the others' messages of the last round.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::env::{round_after, GameAction};
use crate::error::{Error, Result};
use crate::settings::Settings;

use super::strategies::{self, GroupStrategy, GroupView};
use super::{GroupGame, GroupLibrary, AGENT_SEAT, AGENT_SEATS};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupRound {
    pub round_number: usize,
    pub actions: Vec<String>,
    pub payoffs: Vec<f64>,
    #[serde(default)]
    pub messages: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupObservation {
    pub done: bool,
    pub reward: f64,
    pub game_name: String,
    pub game_description: String,
    pub available_actions: Vec<String>,
    pub current_round: usize,
    pub total_rounds: usize,
    pub history: Vec<GroupRound>,
    pub scores: Vec<f64>,
    pub num_players: usize,
    pub player_index: usize,
    pub last_round: Option<GroupRound>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupState {
    pub episode_id: String,
    pub step_count: usize,
    pub game_name: String,
    pub current_round: usize,
    pub total_rounds: usize,
    pub num_players: usize,
    pub scores: Vec<f64>,
    pub history: Vec<GroupRound>,
    pub is_done: bool,
}

/// Another model holding one of the other seats.
pub trait GroupAgent: Send {
    fn act(&mut self, observation: &GroupObservation) -> Result<GameAction>;
}

/// Who holds one of the other seats.
pub enum Seat {
    Strategy(Box<dyn GroupStrategy>),
    Agent(Box<dyn GroupAgent>),
    /// The caller supplies this seat's move with every round through `play`
    /// (the coalition layer does, after negotiation).
    Given,
}

impl Seat {
    /// One strategy per other seat, or one name for all of them.
    pub fn strategies(names: &[String], others: usize) -> Result<Vec<Seat>> {
        let names: Vec<&String> = match names {
            [only] => std::iter::repeat(only).take(others).collect(),
            many if many.len() == others => many.iter().collect(),
            _ => {
                return Err(Error::Usage(format!(
                    "name one strategy for every other seat ({others}) or one for all of them, not {}",
                    names.len()
                )))
            }
        };
        names
            .into_iter()
            .map(|name| strategies::named(name).map(Seat::Strategy))
            .collect()
    }
}

pub struct GroupEnvironment {
    library: Arc<GroupLibrary>,
    settings: Arc<Settings>,
    seed: u64,
    rng: StdRng,
    game: Option<GroupGame>,
    seats: Vec<Seat>,
    state: GroupState,
}

impl GroupEnvironment {
    pub fn new(library: Arc<GroupLibrary>, settings: Arc<Settings>) -> Result<Self> {
        let seed = match settings.seed()? {
            Some(seed) => seed,
            None => rand::random(),
        };
        Ok(Self {
            library,
            settings,
            seed,
            rng: StdRng::seed_from_u64(seed),
            game: None,
            seats: Vec::new(),
            state: GroupState::default(),
        })
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn game(&self) -> Option<&GroupGame> {
        self.game.as_ref()
    }

    pub fn state(&self) -> &GroupState {
        &self.state
    }

    /// Start an episode of `key` with `seats` holding seats one onwards.
    pub fn reset(&mut self, key: &str, seats: Vec<Seat>, rounds: Option<usize>, episode_id: Option<String>) -> Result<GroupObservation> {
        let game = self.library.build(key, &self.settings)?;
        let others = game.players.saturating_sub(AGENT_SEATS);
        if seats.len() != others {
            return Err(Error::Usage(format!(
                "{key} seats {} players; give a strategy or agent for each of the {others} seats besides the agent's",
                game.players
            )));
        }
        self.state = GroupState {
            episode_id: match episode_id {
                Some(id) => id,
                None => uuid::Uuid::new_v4().to_string(),
            },
            game_name: key.to_owned(),
            total_rounds: match rounds {
                Some(rounds) => rounds,
                None => game.rounds,
            },
            num_players: game.players,
            scores: vec![Default::default(); game.players],
            ..GroupState::default()
        };
        self.seats = seats;
        self.game = Some(game);
        self.observation(AGENT_SEAT, Default::default(), None)
    }

    /// The agent's move; every other seat moves in the same step.
    pub fn step(&mut self, action: &GameAction) -> Result<GroupObservation> {
        let game = self.game.clone().ok_or(Error::NotStarted)?;
        let mut moves = vec![action.action.clone()];
        let mut messages = vec![action.message()];
        let history = self.state.history.clone();
        for (index, seat) in self.seats.iter_mut().enumerate() {
            let number = index + AGENT_SEATS;
            let (played, said) = match seat {
                Seat::Strategy(strategy) => {
                    let view = GroupView { game: &game, seat: number, history: &history };
                    (strategy.choose(&view, &mut self.rng)?, String::new())
                }
                Seat::Agent(agent) => {
                    let observation = Self::view(&game, &self.state, number, Default::default(), None);
                    let answer = agent.act(&observation)?;
                    let message = answer.message();
                    (answer.action, message)
                }
                Seat::Given => {
                    return Err(Error::Usage(format!(
                        "seat {number}'s move is supplied by the caller; record the round with play"
                    )))
                }
            };
            moves.push(played);
            messages.push(said);
        }
        self.play(moves, messages)
    }

    /// Record a round whose every move is given, seat zero first; the reward
    /// is seat zero's payoff.
    pub fn play(&mut self, moves: Vec<String>, messages: Vec<String>) -> Result<GroupObservation> {
        let game = self.game.clone().ok_or(Error::NotStarted)?;
        if self.state.is_done {
            return Err(Error::Finished {
                episode: self.state.episode_id.clone(),
                rounds: self.state.history.len(),
            });
        }
        for played in &moves {
            if !game.actions.contains(played) {
                return Err(Error::InvalidAction {
                    game: game.key.clone(),
                    action: played.clone(),
                    allowed: game.actions.join(", "),
                });
            }
        }
        let payoffs = game.pay(&moves)?;
        let round = GroupRound {
            round_number: round_after(self.state.history.len()),
            actions: moves,
            payoffs: payoffs.clone(),
            messages,
        };
        for (score, paid) in self.state.scores.iter_mut().zip(&payoffs) {
            *score += paid;
        }
        self.state.history.push(round.clone());
        self.state.current_round = self.state.history.len();
        self.state.step_count = self.state.history.len();
        self.state.is_done = self.state.current_round >= self.state.total_rounds;
        let reward = payoffs.get(AGENT_SEAT).copied().ok_or(Error::NotStarted)?;
        self.observation(AGENT_SEAT, reward, Some(round))
    }

    /// The episode as `seat` sees it.
    pub fn observation(&self, seat: usize, reward: f64, last: Option<GroupRound>) -> Result<GroupObservation> {
        let game = self.game.as_ref().ok_or(Error::NotStarted)?;
        Ok(Self::view(game, &self.state, seat, reward, last))
    }

    fn view(game: &GroupGame, state: &GroupState, seat: usize, reward: f64, last: Option<GroupRound>) -> GroupObservation {
        let mut metadata = Map::new();
        if game.has_variant("free_chat") {
            metadata.insert("free_chat".to_owned(), Value::Bool(true));
        }
        if let Some(round) = state.history.last() {
            let heard: Vec<Value> = round
                .messages
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != seat)
                .map(|(_, said)| Value::String(said.clone()))
                .collect();
            if heard.iter().any(|said| said.as_str().is_some_and(|text| !text.is_empty())) {
                metadata.insert("last_opp_messages".to_owned(), Value::Array(heard));
            }
            if let Some(own) = round.messages.get(seat).filter(|said| !said.is_empty()) {
                metadata.insert("last_player_message".to_owned(), Value::String(own.clone()));
            }
        }
        GroupObservation {
            done: state.is_done,
            reward,
            game_name: state.game_name.clone(),
            game_description: game.description.clone(),
            available_actions: game.actions.clone(),
            current_round: state.current_round,
            total_rounds: state.total_rounds,
            history: state.history.clone(),
            scores: state.scores.clone(),
            num_players: state.num_players,
            player_index: seat,
            last_round: last,
            metadata,
        }
    }
}
