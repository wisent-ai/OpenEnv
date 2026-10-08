//! Who plays a seat: a model behind Brama, or a library strategy. Both
//! answer an observation with an action, so either can hold the agent's seat
//! in a tournament or the opponent's seat in an episode.
//!
//! A model seat reads the run's `agent` section: `history_rounds` (how many
//! past rounds its prompt shows), and optionally `temperature`, `top_p` and
//! `max_tokens`, sent only when declared.

pub mod brama;
pub mod prompt;

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use serde_json::Value;

use crate::env::{Agent, GameAction, GameObservation};
use crate::error::{Error, Result};
use crate::game::Game;
use crate::settings::Settings;
use crate::strategy::{self, Strategy, Turn, View};

use brama::{Brama, Sampling};
use prompt::Phase;

/// One exchange with a model: what it was asked and what it answered.
#[derive(Clone, Debug, Serialize)]
pub struct Exchange {
    pub prompt: String,
    pub answer: String,
}

pub struct ModelAgent {
    brama: Arc<Brama>,
    route: String,
    sampling: Sampling,
    history_rounds: usize,
    transcript: Vec<Exchange>,
}

impl ModelAgent {
    pub fn new(brama: Arc<Brama>, route: &str, settings: &Settings) -> Result<Self> {
        let declared = settings.section("agent")?;
        let optional_number = |name: &str| -> Result<Option<f64>> {
            match declared.has(name) {
                true => declared.number(name).map(Some),
                false => Ok(None),
            }
        };
        let max_tokens = match declared.has("max_tokens") {
            true => Some(declared.whole("max_tokens")?),
            false => None,
        };
        Ok(Self {
            brama,
            route: route.to_owned(),
            sampling: Sampling {
                temperature: optional_number("temperature")?,
                top_p: optional_number("top_p")?,
                max_tokens,
            },
            history_rounds: declared.whole("history_rounds").map(|rounds| rounds as usize)?,
            transcript: Vec::new(),
        })
    }

    pub fn transcript(&self) -> &[Exchange] {
        &self.transcript
    }

    pub fn sampling(&self) -> &Sampling {
        &self.sampling
    }
}

impl Agent for ModelAgent {
    fn act(&mut self, observation: &GameObservation) -> Result<GameAction> {
        let asked = prompt::build(observation, self.history_rounds);
        let answer = self.brama.chat(&self.route, prompt::SYSTEM, &asked, &self.sampling)?;
        self.transcript.push(Exchange { prompt: asked, answer: answer.clone() });
        match prompt::phase(observation) {
            Phase::Message => {
                // The message step reads only the message; its action is the
                // first listed move, which the environment does not play.
                let first = observation.available_actions.first().ok_or_else(|| Error::NoMoves {
                    strategy: self.route.clone(),
                    game: observation.game_name.clone(),
                })?;
                let mut action = GameAction::new(first);
                action.metadata.insert("message".to_owned(), Value::String(answer.trim().to_owned()));
                Ok(action)
            }
            Phase::Action | Phase::Plain => Ok(GameAction::new(&prompt::parse(&answer, &observation.available_actions)?)),
        }
    }
}

/// A library strategy holding a seat, seeing the episode from that seat.
pub struct StrategyAgent {
    game: Game,
    strategy: Box<dyn Strategy>,
    rng: StdRng,
}

impl StrategyAgent {
    pub fn new(name: &str, game: Game, settings: &Settings, seed: u64) -> Result<Self> {
        Ok(Self {
            game,
            strategy: strategy::named(name, settings)?,
            rng: StdRng::seed_from_u64(seed),
        })
    }
}

impl Agent for StrategyAgent {
    fn act(&mut self, observation: &GameObservation) -> Result<GameAction> {
        let history: Vec<Turn> = observation
            .history
            .iter()
            .map(|round| Turn {
                agent: round.opponent_action.clone(),
                own: round.player_action.clone(),
            })
            .collect();
        let view = View {
            game: &self.game,
            moves: &observation.available_actions,
            history: &history,
            answering: None,
        };
        Ok(GameAction::new(&self.strategy.choose(&view, &mut self.rng)?))
    }
}
