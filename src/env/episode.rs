//! One running episode: the game, who holds the opponent's seat, the state so
//! far, and the free-chat phase when the game has one.

use rand::RngCore;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::game::Game;
use crate::strategy::{Strategy, Turn, View};

use super::models::{round_after, GameAction, GameObservation, GameState, RoundResult};
use super::Agent;

/// Who holds the opponent's seat.
pub enum Opponent {
    Strategy { name: String, strategy: Box<dyn Strategy> },
    Agent(Box<dyn Agent>),
}

/// Where a free-chat round stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Message,
    Action,
}

impl Phase {
    pub(super) fn name(self) -> &'static str {
        match self {
            Phase::Message => "message",
            Phase::Action => "action",
        }
    }
}

/// What the opponent agent is told about this round's chat.
pub(super) struct Chat<'a> {
    pub phase: Phase,
    /// What the agent said this round, shown to the opponent.
    pub heard: &'a str,
    /// What the opponent itself said this round.
    pub said: &'a str,
}

pub(super) struct Episode {
    pub game: Game,
    pub opponent: Opponent,
    pub opponent_name: String,
    pub state: GameState,
    pub phase: Phase,
    pub pending_player: String,
    pub pending_opponent: String,
    /// One entry per step taken, naming its kind; the state's step count.
    pub steps: Vec<&'static str>,
}

impl Episode {
    pub fn start(game: Game, opponent: Opponent, game_key: &str, episode_id: String, rounds: usize) -> Self {
        let opponent_name = match &opponent {
            Opponent::Strategy { name, .. } => name.clone(),
            Opponent::Agent(_) => "agent".to_owned(),
        };
        let state = GameState {
            episode_id,
            game_name: game_key.to_owned(),
            opponent_strategy: opponent_name.clone(),
            total_rounds: rounds,
            ..GameState::default()
        };
        Self {
            game,
            opponent,
            opponent_name,
            state,
            phase: Phase::Message,
            pending_player: String::new(),
            pending_opponent: String::new(),
            steps: Vec::new(),
        }
    }

    pub fn check_move(&self, action: &str) -> Result<()> {
        if self.game.actions.iter().any(|allowed| allowed == action) {
            return Ok(());
        }
        Err(Error::InvalidAction {
            game: self.game.key.clone(),
            action: action.to_owned(),
            allowed: self.game.actions.join(", "),
        })
    }

    /// The opponent's move and message this round. A library strategy sends
    /// no message; an agent's message rides on its action's `message`.
    pub fn opponent_move(
        &mut self,
        player_action: &str,
        chat: Option<Chat<'_>>,
        rng: &mut dyn RngCore,
    ) -> Result<(String, String)> {
        let observation = self.opponent_observation(chat);
        let moves = self.game.opponent_moves().to_vec();
        match &mut self.opponent {
            Opponent::Strategy { strategy, .. } => {
                let history: Vec<Turn> = self
                    .state
                    .history
                    .iter()
                    .map(|round| Turn {
                        agent: round.player_action.clone(),
                        own: round.opponent_action.clone(),
                    })
                    .collect();
                let view = View {
                    game: &self.game,
                    moves: &moves,
                    history: &history,
                    answering: self.game.responds.then_some(player_action),
                };
                Ok((strategy.choose(&view, rng)?, String::new()))
            }
            Opponent::Agent(agent) => {
                let answer: GameAction = agent.act(&observation)?;
                if !moves.contains(&answer.action) {
                    return Err(Error::Opponent {
                        reason: format!(
                            "it played {}, which is not one of its moves: {}",
                            answer.action,
                            moves.join(", ")
                        ),
                    });
                }
                Ok((answer.action.clone(), answer.message()))
            }
        }
    }

    /// The opponent agent's message in a free-chat message phase; a library
    /// strategy has no language and says nothing.
    pub fn opponent_message(&mut self) -> Result<String> {
        let observation = self.opponent_observation(Some(Chat {
            phase: Phase::Message,
            heard: "",
            said: "",
        }));
        match &mut self.opponent {
            Opponent::Strategy { .. } => Ok(String::new()),
            Opponent::Agent(agent) => Ok(agent.act(&observation)?.message()),
        }
    }

    /// Pay a round, record it and advance the state.
    pub fn settle(
        &mut self,
        player_action: &str,
        opponent_action: &str,
        player_message: String,
        opponent_message: String,
        rng: &mut dyn RngCore,
    ) -> Result<RoundResult> {
        let (player_payoff, opponent_payoff) = self.game.pay(player_action, opponent_action, rng)?;
        let result = RoundResult {
            round_number: round_after(self.state.history.len()),
            player_action: player_action.to_owned(),
            opponent_action: opponent_action.to_owned(),
            player_payoff,
            opponent_payoff,
            player_message,
            opponent_message,
        };
        self.state.history.push(result.clone());
        self.steps.push("action");
        self.state.step_count = self.steps.len();
        self.state.current_round = self.state.history.len();
        self.state.player_score += player_payoff;
        self.state.opponent_score += opponent_payoff;
        self.state.is_done = self.state.current_round >= self.state.total_rounds;
        Ok(result)
    }

    pub fn after_round(&self, result: RoundResult) -> GameObservation {
        let reward = result.player_payoff;
        self.observation(reward, Some(result))
    }

    fn chat_metadata(&self) -> Map<String, Value> {
        let mut metadata = Map::new();
        if self.game.has_variant("free_chat") {
            metadata.insert("free_chat".to_owned(), Value::Bool(true));
            metadata.insert("phase".to_owned(), Value::String(self.phase.name().to_owned()));
        }
        metadata
    }

    pub fn observation(&self, reward: f64, last_round: Option<RoundResult>) -> GameObservation {
        let mut metadata = self.chat_metadata();
        if let Some(round) = &last_round {
            if !round.opponent_message.is_empty() {
                metadata.insert("last_opp_message".to_owned(), Value::String(round.opponent_message.clone()));
            }
            if !round.player_message.is_empty() {
                metadata.insert("last_player_message".to_owned(), Value::String(round.player_message.clone()));
            }
        }
        GameObservation {
            done: self.state.is_done,
            reward,
            game_name: self.state.game_name.clone(),
            game_description: self.game.description.clone(),
            available_actions: self.game.actions.clone(),
            current_round: self.state.current_round,
            total_rounds: self.state.total_rounds,
            history: self.state.history.clone(),
            player_score: self.state.player_score,
            opponent_score: self.state.opponent_score,
            opponent_strategy: self.opponent_name.clone(),
            last_round,
            metadata,
        }
    }

    /// The episode from the opponent's seat: history, scores and messages
    /// swapped so the opponent sees itself as the player.
    fn opponent_observation(&self, chat: Option<Chat<'_>>) -> GameObservation {
        let history: Vec<RoundResult> = self.state.history.iter().map(RoundResult::flipped).collect();
        let mut metadata = Map::new();
        if let Some(last) = history.last() {
            if !last.opponent_message.is_empty() {
                metadata.insert("last_opp_message".to_owned(), Value::String(last.opponent_message.clone()));
            }
            if !last.player_message.is_empty() {
                metadata.insert("last_player_message".to_owned(), Value::String(last.player_message.clone()));
            }
        }
        if let Some(chat) = chat {
            metadata.insert("free_chat".to_owned(), Value::Bool(true));
            metadata.insert("phase".to_owned(), Value::String(chat.phase.name().to_owned()));
            if chat.phase == Phase::Action {
                metadata.insert("last_opp_message".to_owned(), Value::String(chat.heard.to_owned()));
                metadata.insert("last_player_message".to_owned(), Value::String(chat.said.to_owned()));
            }
        }
        GameObservation {
            done: false,
            reward: Default::default(),
            game_name: self.state.game_name.clone(),
            game_description: self.game.description.clone(),
            available_actions: self.game.opponent_moves().to_vec(),
            current_round: self.state.current_round,
            total_rounds: self.state.total_rounds,
            history,
            player_score: self.state.opponent_score,
            opponent_score: self.state.player_score,
            opponent_strategy: "agent".to_owned(),
            last_round: None,
            metadata,
        }
    }
}
