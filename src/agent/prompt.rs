//! The prompt a model seat reads, and how its answer becomes a move. The
//! prompt never names the opponent's strategy, so a model cannot shortcut
//! the game by recognising it. How many past rounds it shows is the run's
//! `agent.history_rounds`.

use crate::env::GameObservation;
use crate::error::{Error, Result};

/// The system turn: what the model is asked to do.
pub const SYSTEM: &str = "You are playing a game-theory game. Analyse the situation and choose the best action. Respond with ONLY the action name, nothing else.";

const MESSAGE_PHASE: &str = "Write ONE short message (one or two sentences) to your opponent. Reply with the message text only, no action and no labels. After both players' messages are revealed you will be asked to choose your action. The message is non-binding cheap talk.";

const ACTION_PHASE: &str = "Choose your action. Reply with EXACTLY ONE of the available actions listed above and nothing else. Your opponent's message is shown above; you may use it to inform your choice.";

/// Where a free-chat round stands, as the observation says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Message,
    Action,
    /// A game without a message step.
    Plain,
}

pub fn phase(observation: &GameObservation) -> Phase {
    match observation.metadata.get("phase").and_then(|phase| phase.as_str()) {
        Some("message") => Phase::Message,
        Some("action") => Phase::Action,
        _ => Phase::Plain,
    }
}

/// The prompt for `observation`, showing at most `history_rounds` past rounds.
pub fn build(observation: &GameObservation, history_rounds: usize) -> String {
    let mut sections = vec![format!("[Game]\n{}\n{}", observation.game_name, observation.game_description)];
    if let Some(said) = observation.metadata.get("last_opp_message").and_then(|said| said.as_str()) {
        if !said.is_empty() {
            sections.push(format!("[Opponent said]\n{said}"));
        }
    }
    let shown = observation.history.len().saturating_sub(history_rounds);
    let lines: Vec<String> = observation.history[shown..]
        .iter()
        .map(|round| {
            format!(
                "Round {} | You played: {} | Opponent played: {} | Your payoff: {} | Opp payoff: {}",
                round.round_number, round.player_action, round.opponent_action, round.player_payoff, round.opponent_payoff
            )
        })
        .collect();
    if !lines.is_empty() {
        sections.push(format!("[History]\n{}", lines.join("\n")));
    }
    sections.push(format!(
        "[Scores]\nYour score: {}\nOpponent score: {}\nRound: {} of {}",
        observation.player_score, observation.opponent_score, observation.current_round, observation.total_rounds
    ));
    let moves: Vec<String> = observation.available_actions.iter().map(|played| format!("- {played}")).collect();
    sections.push(format!("[Available Actions]\n{}", moves.join("\n")));
    let instruction = match phase(observation) {
        Phase::Message => MESSAGE_PHASE,
        Phase::Action => ACTION_PHASE,
        Phase::Plain => SYSTEM,
    };
    sections.push(format!("[Instruction]\n{instruction}"));
    sections.join("\n\n")
}

/// The move an answer names: exactly, ignoring case, or as the one move the
/// answer contains. An answer naming no move is refused with what it said,
/// never replaced by a move nobody chose; an answer containing several moves
/// is refused too, since nothing says which one the model meant.
pub fn parse(answer: &str, moves: &[String]) -> Result<String> {
    let said = answer.trim();
    if let Some(exact) = moves.iter().find(|played| played.as_str() == said) {
        return Ok(exact.clone());
    }
    let lower = said.to_lowercase();
    if let Some(same) = moves.iter().find(|played| played.to_lowercase() == lower) {
        return Ok(same.clone());
    }
    let contained: Vec<&String> = moves.iter().filter(|played| lower.contains(&played.to_lowercase())).collect();
    // The longest contained move wins when the others are parts of it
    // (`defect` inside `always_defect`).
    let longest = contained.iter().max_by_key(|played| played.len());
    match longest {
        Some(longest) if contained.iter().all(|played| longest.contains(played.as_str())) => Ok((*longest).clone()),
        _ => Err(Error::Unparsed {
            answer: answer.to_owned(),
            moves: moves.join(", "),
        }),
    }
}
