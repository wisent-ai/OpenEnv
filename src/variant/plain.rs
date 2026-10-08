//! Variants that change one thing about a game: an exit, a costly binding
//! commitment, noise on moves or payoffs, who holds the opponent's seat, or a
//! free-chat channel.

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::distributions::{Bernoulli, Distribution};
use rand::seq::SliceRandom;
use rand::RngCore;
use rand_distr::Normal;

use crate::error::{Error, Result};
use crate::game::{Game, OpponentMode, OpponentMoves, Seats, NONE};
use crate::settings::Declared;

fn marked(mut game: Game, variant: &str) -> Game {
    game.variants.push(variant.to_owned());
    game
}

fn pair_only(game: &Game, variant: &str) -> Result<()> {
    match game.seats {
        Seats::Pair => Ok(()),
        Seats::Group(_) => Err(Error::Unsupported {
            game: game.key.clone(),
            reason: format!("{variant} applies to a game of two seats, and this one seats a group"),
        }),
    }
}

/// Adds `exit` to both seats' moves; if either seat exits both get the
/// declared `payoff`.
pub(super) fn exit(base: Game, declared: &Declared<'_>) -> Result<Game> {
    let paid = declared.number("payoff")?;
    let inner = base.payoff.clone();
    let mut game = base;
    game.actions.push("exit".to_owned());
    if let OpponentMoves::Own(moves) = &mut game.opponent_actions {
        moves.push("exit".to_owned());
    }
    game.payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        if player == "exit" || opponent == "exit" {
            return Ok((paid, paid));
        }
        inner(player, opponent, rng)
    });
    Ok(marked(game, "exit"))
}

/// The first (cooperative) move gains a `commit_<move>` form that locks the
/// seat into it at the declared `cost`; every move gains a free `free_<move>`
/// form.
pub(super) fn binding_commitment(base: Game, declared: &Declared<'_>) -> Result<Game> {
    let cost = declared.number("cost")?;
    let committed = base.actions.first().cloned().ok_or_else(|| Error::Unsupported {
        game: base.key.clone(),
        reason: "binding_commitment needs a game with moves".to_owned(),
    })?;
    let mut parts = BTreeMap::new();
    let mut actions = vec![format!("commit_{committed}")];
    parts.insert(format!("commit_{committed}"), (committed.clone(), true));
    for played in &base.actions {
        actions.push(format!("free_{played}"));
        parts.insert(format!("free_{played}"), (played.clone(), false));
    }
    let name = base.key.clone();
    let inner = base.payoff.clone();
    let mut game = base;
    game.payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        let read = |action: &str| {
            parts.get(action).cloned().ok_or_else(|| Error::InvalidAction {
                game: name.clone(),
                action: action.to_owned(),
                allowed: parts.keys().cloned().collect::<Vec<_>>().join(", "),
            })
        };
        let ((mine, my_lock), (theirs, their_lock)) = (read(player)?, read(opponent)?);
        let (mut paid_mine, mut paid_theirs) = inner(&mine, &theirs, rng)?;
        if my_lock {
            paid_mine -= cost;
        }
        if their_lock {
            paid_theirs -= cost;
        }
        Ok((paid_mine, paid_theirs))
    });
    game.actions = actions;
    game.opponent_actions = OpponentMoves::Shared;
    Ok(marked(game, "binding_commitment"))
}

/// Each seat's move is replaced by a random one of its moves with the
/// declared `tremble` probability.
pub(super) fn noisy_actions(base: Game, declared: &Declared<'_>) -> Result<Game> {
    pair_only(&base, "noisy_actions")?;
    let tremble: Bernoulli = declared.probability("tremble")?;
    let agent_moves = base.actions.clone();
    let opponent_moves = base.opponent_moves().to_vec();
    let inner = base.payoff.clone();
    let mut game = base;
    game.payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        let mut played = |intended: &str, moves: &[String], rng: &mut dyn RngCore| -> String {
            if tremble.sample(rng) {
                if let Some(slipped) = moves.choose(rng) {
                    return slipped.clone();
                }
            }
            intended.to_owned()
        };
        let mine = played(player, &agent_moves, rng);
        let theirs = played(opponent, &opponent_moves, rng);
        inner(&mine, &theirs, rng)
    });
    Ok(marked(game, "noisy_actions"))
}

/// Each payoff gets independent Gaussian noise of mean zero and the declared
/// standard deviation `scale`.
pub(super) fn noisy_payoffs(base: Game, declared: &Declared<'_>) -> Result<Game> {
    pair_only(&base, "noisy_payoffs")?;
    let scale = declared.number("scale")?;
    let noise = Normal::new(NONE, scale).map_err(|refusal| Error::Malformed {
        scope: declared.scope().to_owned(),
        name: "scale".to_owned(),
        expected: "a standard deviation of zero or more".to_owned(),
        found: format!("{scale} ({refusal})"),
    })?;
    let inner = base.payoff.clone();
    let mut game = base;
    game.payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        let (mine, theirs) = inner(player, opponent, rng)?;
        Ok((mine + noise.sample(rng), theirs + noise.sample(rng)))
    });
    Ok(marked(game, "noisy_payoffs"))
}

/// Who holds the opponent's seat: the same model, or another one.
pub(super) fn opponent_mode(base: Game, variant: &str, mode: OpponentMode) -> Result<Game> {
    pair_only(&base, variant)?;
    let mut game = base;
    game.opponent_mode = mode;
    Ok(marked(game, variant))
}

/// A free-form message channel: the moves and payoffs are unchanged, and each
/// round gains a message step (see `env::free_chat`).
pub(super) fn free_chat(base: Game) -> Game {
    marked(base, "free_chat")
}
