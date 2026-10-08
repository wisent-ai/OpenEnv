//! The classic games: the three canonical 2×2 dilemmas and the three
//! amount games (ultimatum, trust, public goods). Their payoffs are declared
//! by the run; the mechanics follow the KantBench paper's definitions:
//! https://github.com/wisent-ai/OpenEnv/blob/main/paper/sections/games/library.tex

use std::sync::Arc;

use crate::error::Result;
use crate::settings::Declared;

use super::{amount, amounts, matrix_payoff, Entry, Game, Library, Matrix, OpponentMoves, NOTHING};

const MATRIX_PARAMETERS: &[&str] = &["payoffs"];

pub(super) fn register(library: &mut Library) {
    for entry in [
        Entry {
            key: "prisoners_dilemma",
            family: "classic",
            parameters: MATRIX_PARAMETERS,
            build: prisoners_dilemma,
        },
        Entry {
            key: "stag_hunt",
            family: "classic",
            parameters: MATRIX_PARAMETERS,
            build: stag_hunt,
        },
        Entry {
            key: "hawk_dove",
            family: "classic",
            parameters: MATRIX_PARAMETERS,
            build: hawk_dove,
        },
        Entry {
            key: "ultimatum",
            family: "classic",
            parameters: &["pot"],
            build: ultimatum,
        },
        Entry {
            key: "trust",
            family: "classic",
            parameters: &["endowment", "multiplier"],
            build: trust,
        },
        Entry {
            key: "public_goods",
            family: "classic",
            parameters: &["endowment", "multiplier", "players"],
            build: public_goods,
        },
    ] {
        library.add(entry);
    }
}

fn moves(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// A 2×2 (or larger) game whose cells the declaration states.
pub(crate) fn declared_matrix(
    declared: &Declared<'_>,
    name: &str,
    description: &str,
    actions: Vec<String>,
) -> Result<Game> {
    let matrix = Matrix::declared(declared, &actions, &actions)?;
    Ok(Game::new(name, description, "matrix", actions, matrix_payoff(name, matrix)))
}

fn prisoners_dilemma(declared: &Declared<'_>) -> Result<Game> {
    declared_matrix(
        declared,
        "Prisoner's Dilemma",
        "Two players simultaneously choose to cooperate or defect. Mutual cooperation yields a moderate reward, mutual defection yields a low reward, and unilateral defection tempts with the highest individual payoff at the other player's expense.",
        moves(&["cooperate", "defect"]),
    )
}

fn stag_hunt(declared: &Declared<'_>) -> Result<Game> {
    declared_matrix(
        declared,
        "Stag Hunt",
        "Two players choose between hunting stag (risky but rewarding if both participate) or hunting hare (safe but less rewarding). Coordination on stag yields the highest joint payoff.",
        moves(&["stag", "hare"]),
    )
}

/// Dove is listed first, so a strategy that opens with its first move (the
/// cooperative one) plays dove.
fn hawk_dove(declared: &Declared<'_>) -> Result<Game> {
    declared_matrix(
        declared,
        "Hawk-Dove",
        "Two players choose between aggressive (hawk) and passive (dove) strategies over a shared resource. Two hawks suffer mutual harm; a hawk facing a dove claims the resource; two doves share it.",
        moves(&["dove", "hawk"]),
    )
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
