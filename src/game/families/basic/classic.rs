//! The classic games: the three canonical 2×2 dilemmas and the three
//! amount games (ultimatum, trust, public goods). Their payoffs are declared
//! by the run; the mechanics follow the KantBench paper's definitions:
//! https://github.com/wisent-ai/OpenEnv/blob/main/paper/sections/games/library.tex

use std::sync::Arc;

use crate::error::Result;
use crate::settings::Declared;

use crate::game::{amount, amounts, matrix_entry, moves, Entry, Game, Library, OpponentMoves, NOTHING};

pub(super) fn register(library: &mut Library) {
    library.add(matrix_entry(
        "prisoners_dilemma",
        "classic",
        &["cooperate", "defect"],
        "Prisoner's Dilemma",
        "Two players simultaneously choose to cooperate or defect. Mutual cooperation yields a moderate reward, mutual defection yields a low reward, and unilateral defection tempts with the highest individual payoff at the other player's expense.",
    ));
    library.add(matrix_entry(
        "stag_hunt",
        "classic",
        &["stag", "hare"],
        "Stag Hunt",
        "Two players choose between hunting stag (risky but rewarding if both participate) or hunting hare (safe but less rewarding). Coordination on stag yields the highest joint payoff.",
    ));
    // Dove is listed first, so a strategy that opens with its first (the
    // cooperative) move plays dove.
    library.add(matrix_entry(
        "hawk_dove",
        "classic",
        &["dove", "hawk"],
        "Hawk-Dove",
        "Two players choose between aggressive (hawk) and passive (dove) strategies over a shared resource. Two hawks suffer mutual harm; a hawk facing a dove claims the resource; two doves share it.",
    ));
    library.add(Entry::new("ultimatum", "classic", &["pot"], ultimatum));
    library.add(Entry::new("trust", "classic", &["endowment", "multiplier"], trust));
    library.add(Entry::new(
        "public_goods",
        "classic",
        &["endowment", "multiplier", "players"],
        public_goods,
    ));
}

/// The proposer offers part of a pot; an accepted offer pays
/// `(pot − offer, offer)`, a rejected one pays both nothing.
fn ultimatum(declared: &Declared<'_>) -> Result<Game> {
    let pot = declared.whole("pot")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn rand::RngCore| {
        let offer = amount(player)?;
        if opponent == "reject" {
            return Ok((NOTHING as f64, NOTHING as f64));
        }
        Ok(((pot - offer) as f64, offer as f64))
    });
    let mut game = Game::new(
        "Ultimatum Game",
        "The proposer offers a split of a fixed pot. The responder either accepts (both receive their shares) or rejects (both receive nothing).",
        "ultimatum",
        amounts("offer", pot),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(moves(&["accept", "reject"]));
    game.responds = true;
    Ok(game)
}

/// The investor sends `x` of an endowment `E`; the trustee receives `m·x` and
/// returns `y`. Payoffs `(E − x + y, m·x − y)`.
fn trust(declared: &Declared<'_>) -> Result<Game> {
    let endowment = declared.whole("endowment")?;
    let multiplier = declared.whole("multiplier")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn rand::RngCore| {
        let invested = amount(player)? as f64;
        let returned = amount(opponent)? as f64;
        Ok((
            endowment as f64 - invested + returned,
            invested * multiplier as f64 - returned,
        ))
    });
    let mut game = Game::new(
        "Trust Game",
        "The investor sends part of an endowment; the amount is multiplied and given to the trustee, who then decides how much to return.",
        "trust",
        amounts("invest", endowment),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(amounts("return", endowment * multiplier));
    game.responds = true;
    Ok(game)
}

/// Each participant contributes `c` of an endowment `E`; the pool is
/// multiplied by `m` and split among the declared number of players:
/// `u = E − c + m·Σc / N`.
fn public_goods(declared: &Declared<'_>) -> Result<Game> {
    let endowment = declared.whole("endowment")?;
    let multiplier = declared.number("multiplier")?;
    let players = declared.count("players")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn rand::RngCore| {
        let mine = amount(player)? as f64;
        let theirs = amount(opponent)? as f64;
        let share = (mine + theirs) * multiplier / players as f64;
        Ok((endowment as f64 - mine + share, endowment as f64 - theirs + share))
    });
    Ok(Game::new(
        "Public Goods Game",
        "Each participant decides how much of their endowment to contribute to a common pool. The pool is multiplied and distributed equally, creating tension between individual free-riding and collective benefit.",
        "public_goods",
        amounts("contribute", endowment),
        payoff,
    ))
}
