//! A free-chat round in two steps. In the message step both seats speak (the
//! agent's `message`, then the opponent's) and both messages are revealed; in
//! the action step both seats move and the round is paid. The opponent hears
//! the agent's message within the same round, which is what cheap talk needs.
//! A library strategy has no language, so against one only the agent speaks.

use rand::RngCore;
use serde_json::{Map, Value};

use crate::error::Result;

use super::episode::{Chat, Episode, Phase};
use super::models::{round_after, GameAction, GameObservation};

pub(super) fn step(episode: &mut Episode, action: &GameAction, rng: &mut dyn RngCore) -> Result<GameObservation> {
    match episode.phase {
        Phase::Message => {
            episode.pending_player = action.message();
            episode.pending_opponent = episode.opponent_message()?;
            episode.phase = Phase::Action;
            episode.steps.push("message");
            episode.state.step_count = episode.steps.len();
            Ok(messages_revealed(episode))
        }
        Phase::Action => {
            episode.check_move(&action.action)?;
            let heard = episode.pending_player.clone();
            let said = episode.pending_opponent.clone();
            let chat = Chat {
                phase: Phase::Action,
                heard: &heard,
                said: &said,
            };
            let (opponent_action, _) = episode.opponent_move(&action.action, Some(chat), rng)?;
            episode.phase = Phase::Message;
            let player_message = std::mem::take(&mut episode.pending_player);
            let opponent_message = std::mem::take(&mut episode.pending_opponent);
            let result = episode.settle(&action.action, &opponent_action, player_message, opponent_message, rng)?;
            Ok(episode.after_round(result))
        }
    }
}

/// The observation after the message step: both messages are visible and the
/// agent's next step must be an action.
fn messages_revealed(episode: &Episode) -> GameObservation {
    let mut metadata = Map::new();
    metadata.insert("free_chat".to_owned(), Value::Bool(true));
    metadata.insert("phase".to_owned(), Value::String(Phase::Action.name().to_owned()));
    metadata.insert("last_opp_message".to_owned(), Value::String(episode.pending_opponent.clone()));
    metadata.insert("last_player_message".to_owned(), Value::String(episode.pending_player.clone()));
    let mut observation = episode.observation(Default::default(), None);
    observation.current_round = round_after(episode.state.current_round);
    observation.metadata = metadata;
    observation
}
