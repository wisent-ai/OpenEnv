//! The KantBench wire documents the Space has always published: an action is
//! a `move`, an observation reports the round from the agent's side. A field
//! with nothing to report yet (no round has been played) is left out rather
//! than sent as a zero no round produced.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::env::GameObservation;
use crate::error::{Error, Result};
use crate::group::environment::GroupObservation;
use crate::group::AGENT_SEAT;

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct KantBenchAction {
    /// The agent's move, one of the observation's `available_moves`.
    #[serde(rename = "move")]
    pub played: String,
    /// A free-chat message sent with the move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
pub struct KantBenchObservation {
    pub game_name: String,
    pub game_description: String,
    pub available_moves: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub your_move: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_move: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub your_payoff: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_payoff: Option<f64>,
    pub cumulative_score: f64,
    pub round_number: usize,
    pub max_rounds: usize,
    pub opponent_strategy: String,
    pub history: Vec<Value>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_players: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all_scores: Option<Vec<f64>>,
    /// Messages the free-chat channel carried, when the game has one.
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
}

fn status(done: bool) -> String {
    if done {
        return "Game over: reset to start a new episode.".to_owned();
    }
    String::new()
}

pub fn pair(observation: &GameObservation) -> KantBenchObservation {
    let last = observation.last_round.as_ref();
    KantBenchObservation {
        game_name: observation.game_name.clone(),
        game_description: observation.game_description.clone(),
        available_moves: observation.available_actions.clone(),
        your_move: last.map(|round| round.player_action.clone()),
        opponent_move: last.map(|round| round.opponent_action.clone()),
        your_payoff: last.map(|round| round.player_payoff),
        opponent_payoff: last.map(|round| round.opponent_payoff),
        cumulative_score: observation.player_score,
        round_number: observation.current_round,
        max_rounds: observation.total_rounds,
        opponent_strategy: observation.opponent_strategy.clone(),
        history: observation
            .history
            .iter()
            .map(|round| {
                json!({
                    "round": round.round_number,
                    "your_move": round.player_action,
                    "opponent_move": round.opponent_action,
                    "your_payoff": round.player_payoff,
                    "opponent_payoff": round.opponent_payoff,
                })
            })
            .collect(),
        message: status(observation.done),
        metadata: observation.metadata.clone(),
        ..KantBenchObservation::default()
    }
}

pub fn group(observation: &GroupObservation) -> Result<KantBenchObservation> {
    let last = observation.last_round.as_ref();
    let score = observation.scores.get(AGENT_SEAT).copied().ok_or_else(|| Error::Usage(format!(
        "{} reports no score for the agent's seat",
        observation.game_name
    )))?;
    Ok(KantBenchObservation {
        game_name: observation.game_name.clone(),
        game_description: observation.game_description.clone(),
        available_moves: observation.available_actions.clone(),
        your_move: last.and_then(|round| round.actions.get(AGENT_SEAT).cloned()),
        your_payoff: last.and_then(|round| round.payoffs.get(AGENT_SEAT).copied()),
        cumulative_score: score,
        round_number: observation.current_round,
        max_rounds: observation.total_rounds,
        history: observation
            .history
            .iter()
            .map(|round| json!({ "round": round.round_number, "actions": round.actions, "payoffs": round.payoffs }))
            .collect(),
        message: status(observation.done),
        num_players: Some(observation.num_players),
        player_index: Some(observation.player_index),
        all_scores: Some(observation.scores.clone()),
        metadata: observation.metadata.clone(),
        ..KantBenchObservation::default()
    })
}
