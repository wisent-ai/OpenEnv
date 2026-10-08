//! The documents an episode exchanges: the agent's action, the observation it
//! gets back, one round's result and the episode's state. Their field names
//! are the OpenEnv wire names the Python environment served, so a client
//! written against it reads them unchanged.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// Rounds are numbered from one, as the environment's observation states:
// https://github.com/wisent-ai/OpenEnv/blob/main/README.md#environment-api
const FIRST_ROUND: usize = 1;

/// The number of the round that follows `played` finished rounds.
pub fn round_after(played: usize) -> usize {
    played + FIRST_ROUND
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundResult {
    pub round_number: usize,
    pub player_action: String,
    pub opponent_action: String,
    pub player_payoff: f64,
    pub opponent_payoff: f64,
    /// The free-form message the agent sent this round (free-chat games).
    #[serde(default)]
    pub player_message: String,
    /// The free-form message the opponent sent this round (free-chat games).
    #[serde(default)]
    pub opponent_message: String,
}

impl RoundResult {
    /// The same round seen from the opponent's seat.
    pub fn flipped(&self) -> Self {
        Self {
            round_number: self.round_number,
            player_action: self.opponent_action.clone(),
            opponent_action: self.player_action.clone(),
            player_payoff: self.opponent_payoff,
            opponent_payoff: self.player_payoff,
            player_message: self.opponent_message.clone(),
            opponent_message: self.player_message.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GameAction {
    pub action: String,
    /// Extra fields; `message` carries a free-chat message.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl GameAction {
    pub fn new(action: &str) -> Self {
        Self {
            action: action.to_owned(),
            metadata: Map::new(),
        }
    }

    /// The free-chat message this action carries, empty when it carries none.
    pub fn message(&self) -> String {
        match self.metadata.get("message") {
            Some(Value::String(text)) => text.clone(),
            _ => String::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GameObservation {
    pub done: bool,
    pub reward: f64,
    pub game_name: String,
    pub game_description: String,
    pub available_actions: Vec<String>,
    pub current_round: usize,
    pub total_rounds: usize,
    pub history: Vec<RoundResult>,
    pub player_score: f64,
    pub opponent_score: f64,
    pub opponent_strategy: String,
    pub last_round: Option<RoundResult>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub episode_id: String,
    pub step_count: usize,
    pub game_name: String,
    pub opponent_strategy: String,
    pub current_round: usize,
    pub total_rounds: usize,
    pub player_score: f64,
    pub opponent_score: f64,
    pub history: Vec<RoundResult>,
    pub is_done: bool,
}
