//! Strategies for the seats of a group game the agent does not hold. Each
//! plays the game's first move as cooperation and its second as defection.

use rand::seq::SliceRandom;
use rand::RngCore;

use crate::error::{Error, Result};

use super::environment::GroupRound;
use super::GroupGame;

/// What a seat sees when it moves.
pub struct GroupView<'a> {
    pub game: &'a GroupGame,
    pub seat: usize,
    pub history: &'a [GroupRound],
}

pub trait GroupStrategy: Send {
    fn choose(&mut self, view: &GroupView<'_>, rng: &mut dyn RngCore) -> Result<String>;
}

pub const NAMES: &[&str] = &["random", "always_cooperate", "always_defect", "tit_for_tat", "adaptive"];

pub fn named(name: &str) -> Result<Box<dyn GroupStrategy>> {
    Ok(match name {
        "random" => Box::new(Random),
        "always_cooperate" => Box::new(Always { defect: false }),
        "always_defect" => Box::new(Always { defect: true }),
        "tit_for_tat" => Box::new(Majority),
        "adaptive" => Box::new(Adaptive),
        other => {
            return Err(Error::UnknownStrategy {
                name: other.to_owned(),
                known: NAMES.join(", "),
            })
        }
    })
}

fn cooperative(view: &GroupView<'_>) -> Result<String> {
    view.game.actions.first().cloned().ok_or_else(|| Error::NoMoves {
        strategy: "group strategy".to_owned(),
        game: view.game.key.clone(),
    })
}

fn defecting(view: &GroupView<'_>) -> Result<String> {
    let mut listed = view.game.actions.iter();
    listed.next();
    listed.next().cloned().ok_or_else(|| Error::Unsupported {
        game: view.game.key.clone(),
        reason: "a defecting strategy needs a second move and the game lists one".to_owned(),
    })
}

/// The other seats' moves in a round.
fn others<'a>(round: &'a GroupRound, seat: usize) -> impl Iterator<Item = &'a String> {
    round
        .actions
        .iter()
        .enumerate()
        .filter(move |(other, _)| *other != seat)
        .map(|(_, played)| played)
}

struct Random;

impl GroupStrategy for Random {
    fn choose(&mut self, view: &GroupView<'_>, rng: &mut dyn RngCore) -> Result<String> {
        view.game.actions.choose(rng).cloned().ok_or_else(|| Error::NoMoves {
            strategy: "random".to_owned(),
            game: view.game.key.clone(),
        })
    }
}

struct Always {
    defect: bool,
}

impl GroupStrategy for Always {
    fn choose(&mut self, view: &GroupView<'_>, _: &mut dyn RngCore) -> Result<String> {
        if self.defect {
            return defecting(view);
        }
        cooperative(view)
    }
}

/// Cooperates first, then plays what most other seats played last round
/// (cooperation on a tie).
struct Majority;

impl GroupStrategy for Majority {
    fn choose(&mut self, view: &GroupView<'_>, _: &mut dyn RngCore) -> Result<String> {
        let cooperate = cooperative(view)?;
        let Some(last) = view.history.last() else {
            return Ok(cooperate);
        };
        let defect = defecting(view)?;
        let defections = others(last, view.seat).filter(|played| **played == defect).count();
        let cooperations = others(last, view.seat).count() - defections;
        if cooperations >= defections {
            return Ok(cooperate);
        }
        Ok(defect)
    }
}

/// Cooperates while the other seats' cooperations over the whole episode
/// outnumber their other moves.
struct Adaptive;

impl GroupStrategy for Adaptive {
    fn choose(&mut self, view: &GroupView<'_>, _: &mut dyn RngCore) -> Result<String> {
        let cooperate = cooperative(view)?;
        if view.history.is_empty() {
            return Ok(cooperate);
        }
        let seen: Vec<&String> = view.history.iter().flat_map(|round| others(round, view.seat)).collect();
        let cooperations = seen.iter().filter(|played| ***played == cooperate).count();
        if cooperations > seen.len() - cooperations {
            return Ok(cooperate);
        }
        defecting(view)
    }
}
