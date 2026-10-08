//! Signaling games: a sender moves, a receiver responds with moves of its
//! own. Beer-Quiche, Spence signaling, cheap talk, Bayesian persuasion and
//! screening read their cells from `payoffs` (rows the sender's moves,
//! columns the receiver's); the lemon market is priced.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::{amount, amounts, matrix_between, moves, Entry, Game, Library, OpponentMoves, NONE};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    for entry in [
        matrix_between(
            "beer_quiche",
            "information",
            &["beer", "quiche"],
            &["challenge", "back_down"],
            "Beer-Quiche Game",
            "A signaling game: the sender chooses a meal (beer or quiche) to signal their type; the receiver decides whether to challenge. Tests reasoning about sequential equilibrium and belief refinement.",
        ),
        matrix_between(
            "spence_signaling",
            "information",
            &["educate", "no_educate"],
            &["high_wage", "low_wage"],
            "Spence Job Market Signaling",
            "A worker chooses whether to acquire education as a signal of ability; a firm responds with a wage offer. Tests understanding of separating versus pooling equilibria in labor markets.",
        ),
        matrix_between(
            "cheap_talk",
            "information",
            &["signal_left", "signal_right"],
            &["act_left", "act_right"],
            "Cheap Talk",
            "A sender observes a state and sends a costless message; the receiver chooses an action. Interests are partially aligned. Tests strategic communication and credibility.",
        ),
        matrix_between(
            "bayesian_persuasion",
            "information",
            &["reveal", "conceal"],
            &["act", "safe"],
            "Bayesian Persuasion",
            "A sender designs an information structure (reveal or conceal the state); a receiver takes an action based on the signal. Tests strategic information disclosure and commitment to information policies.",
        ),
        matrix_between(
            "screening",
            "information",
            &["offer_premium", "offer_basic"],
            &["choose_premium", "choose_basic"],
            "Screening Game",
            "An uninformed principal offers a menu of contracts; agents of different types self-select. Tests understanding of incentive compatibility and separating mechanisms as in Rothschild-Stiglitz insurance models.",
        ),
        Entry::new("lemon_market", "information", &["value", "cost", "most"], lemon_market),
    ] {
        library.add(entry);
    }
}

/// The seller names a price from nothing to `most`; the buyer buys or passes.
/// A sale pays the seller `price − cost` and the buyer `value − price`, where
/// `value` and `cost` are what each side expects over the qualities it cannot
/// tell apart. A pass pays both nothing.
fn lemon_market(declared: &Declared<'_>) -> Result<Game> {
    let value = declared.number("value")?;
    let cost = declared.number("cost")?;
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let price = amount(player)? as f64;
        if opponent == "pass" {
            return Ok((NONE, NONE));
        }
        Ok((price - cost, value - price))
    });
    let mut game = Game::new(
        "Lemon Market",
        "A seller with private quality information sets a price; the buyer decides whether to purchase. Adverse selection can cause market unraveling where only low-quality goods trade.",
        "lemon",
        amounts("price", most),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(moves(&["buy", "pass"]));
    game.responds = true;
    Ok(game)
}
