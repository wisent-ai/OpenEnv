//! Principal-agent games: moral hazard and gift exchange. The principal
//! offers an amount; the agent answers with effort after seeing it.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::{amount, amounts, moves, Entry, Game, Library, OpponentMoves};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new(
        "moral_hazard",
        "information",
        &["base_output", "effort_boost", "effort_cost", "most_bonus"],
        moral_hazard,
    ));
    library.add(Entry::new(
        "gift_exchange",
        "information",
        &["most_wage", "most_effort", "effort_cost", "productivity"],
        gift_exchange,
    ));
}

/// The principal offers a bonus from nothing to `most_bonus`; the agent works
/// or shirks. Output is `base_output`, plus `effort_boost` when the agent
/// works. The principal gets output less the bonus; the agent gets the bonus,
/// less `effort_cost` when it works.
fn moral_hazard(declared: &Declared<'_>) -> Result<Game> {
    let base = declared.number("base_output")?;
    let boost = declared.number("effort_boost")?;
    let effort_cost = declared.number("effort_cost")?;
    let most = declared.whole("most_bonus")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let bonus = amount(player)? as f64;
        if opponent == "work" {
            return Ok((base + boost - bonus, bonus - effort_cost));
        }
        Ok((base - bonus, bonus))
    });
    let mut game = Game::new(
        "Moral Hazard (Principal-Agent)",
        "A principal offers a bonus contract; an agent with unobservable effort decides whether to work or shirk. Tests optimal incentive design and the tradeoff between motivation and rent extraction.",
        "moral_hazard",
        amounts("bonus", most),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(moves(&["work", "shirk"]));
    game.responds = true;
    Ok(game)
}

/// The employer offers a wage from nothing to `most_wage`; the worker answers
/// with effort from nothing to `most_effort`. The employer gets
/// `productivity · effort − wage`; the worker `wage − effort_cost · effort`.
fn gift_exchange(declared: &Declared<'_>) -> Result<Game> {
    let most_wage = declared.whole("most_wage")?;
    let most_effort = declared.whole("most_effort")?;
    let effort_cost = declared.number("effort_cost")?;
    let productivity = declared.number("productivity")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let wage = amount(player)? as f64;
        let effort = amount(opponent)? as f64;
        Ok((productivity * effort - wage, wage - effort_cost * effort))
    });
    let mut game = Game::new(
        "Gift Exchange Game",
        "An employer offers a wage; a worker chooses effort. Nash prediction is minimal effort regardless of wage, but reciprocity often leads to higher wages eliciting higher effort. Tests fairness-driven behavior.",
        "gift_exchange",
        amounts("wage", most_wage),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(amounts("effort", most_effort));
    game.responds = true;
    Ok(game)
}
