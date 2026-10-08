//! One client's environment: reset routes the requested game to the two-seat
//! or the group environment, step plays the agent's move, state reports the
//! episode. Every answer is OpenEnv's step result: the observation, its
//! reward and whether the episode is done.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::env::{Environment, GameAction, Reset};
use crate::error::{Error, Result};
use crate::game::Library;
use crate::group::environment::{GroupEnvironment, Seat};
use crate::group::{GroupLibrary, AGENT_SEATS};
use crate::settings::Settings;

use super::space::{self, KantBenchAction, KantBenchObservation};

/// What a reset asks for: the game, the opponent strategy (for a group game,
/// the strategy every other seat plays), and optionally a variant applied
/// over the game and a round count other than the game's declared one.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
pub struct ResetRequest {
    pub game: String,
    pub strategy: String,
    #[serde(default)]
    pub num_rounds: Option<usize>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub episode_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct StepResult {
    pub observation: KantBenchObservation,
    pub reward: Option<f64>,
    pub done: bool,
}

enum Running {
    Idle,
    Pair(Environment),
    Group(GroupEnvironment),
}

pub struct Session {
    pairs: Arc<Library>,
    groups: Arc<GroupLibrary>,
    settings: Arc<Settings>,
    running: Running,
}

impl Session {
    pub fn new(pairs: Arc<Library>, groups: Arc<GroupLibrary>, settings: Arc<Settings>) -> Self {
        Self { pairs, groups, settings, running: Running::Idle }
    }

    pub fn reset(&mut self, request: &ResetRequest) -> Result<StepResult> {
        let key = match &request.variant {
            Some(variant) => format!("{variant}_{}", request.game),
            None => request.game.clone(),
        };
        if self.groups.entries().any(|entry| entry.key == request.game) {
            let players = self.groups.build(&key, &self.settings)?.players;
            let seats = Seat::strategies(&[request.strategy.clone()], players.saturating_sub(AGENT_SEATS))?;
            let mut environment = GroupEnvironment::new(self.groups.clone(), self.settings.clone())?;
            let observation = environment.reset(&key, seats, request.num_rounds, request.episode_id.clone())?;
            self.running = Running::Group(environment);
            return Ok(StepResult { reward: None, done: observation.done, observation: space::group(&observation)? });
        }
        let mut environment = Environment::new(self.pairs.clone(), self.settings.clone())?;
        let observation = environment.reset(
            &Reset {
                game: key,
                strategy: Some(request.strategy.clone()),
                rounds: request.num_rounds,
                episode_id: request.episode_id.clone(),
            },
            None,
        )?;
        self.running = Running::Pair(environment);
        Ok(StepResult { reward: None, done: observation.done, observation: space::pair(&observation) })
    }

    pub fn step(&mut self, action: &KantBenchAction) -> Result<StepResult> {
        let mut played = GameAction::new(&action.played);
        if let Some(message) = &action.message {
            played.metadata.insert("message".to_owned(), Value::String(message.clone()));
        }
        match &mut self.running {
            Running::Idle => Err(Error::NotStarted),
            Running::Pair(environment) => {
                let observation = environment.step(&played)?;
                Ok(StepResult { reward: Some(observation.reward), done: observation.done, observation: space::pair(&observation) })
            }
            Running::Group(environment) => {
                let observation = environment.step(&played)?;
                Ok(StepResult { reward: Some(observation.reward), done: observation.done, observation: space::group(&observation)? })
            }
        }
    }

    /// The episode's id and how many steps it has taken.
    pub fn state(&self) -> Result<Value> {
        match &self.running {
            Running::Idle => Err(Error::NotStarted),
            Running::Pair(environment) => {
                let state = environment.state().ok_or(Error::NotStarted)?;
                Ok(json!({ "episode_id": state.episode_id, "step_count": state.step_count }))
            }
            Running::Group(environment) => {
                let state = environment.state();
                Ok(json!({ "episode_id": state.episode_id, "step_count": state.step_count }))
            }
        }
    }
}
