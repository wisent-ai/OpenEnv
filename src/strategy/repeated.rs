//! The repeated-game strategies of Axelrod's tournaments and their relatives.
//! Each plays the game's first move as cooperation and its second as
//! defection, and reacts to the agent's earlier moves.

use rand::distributions::{Bernoulli, Distribution};
use rand::seq::SliceRandom;
use rand::RngCore;

use crate::error::{Error, Result};

use super::{cooperative, defecting, Strategy, Turn, View};

pub struct Random;

impl Strategy for Random {
    fn choose(&mut self, view: &View<'_>, rng: &mut dyn RngCore) -> Result<String> {
        view.moves.choose(rng).cloned().ok_or_else(|| Error::NoMoves {
            strategy: "random".to_owned(),
            game: view.game.key.clone(),
        })
    }
}

pub struct AlwaysCooperate;

impl Strategy for AlwaysCooperate {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        cooperative("always_cooperate", view)
    }
}

pub struct AlwaysDefect;

impl Strategy for AlwaysDefect {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        defecting("always_defect", view)
    }
}

/// What a mirroring strategy plays before the agent has moved.
pub enum Opening {
    Cooperate,
    Defect,
}

/// Opens as declared, then plays the agent's previous move (or cooperates
/// when that move is not one of its own).
pub struct TitForTat {
    pub opening: Opening,
}

impl Strategy for TitForTat {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let name = match self.opening {
            Opening::Cooperate => "tit_for_tat",
            Opening::Defect => "suspicious_tit_for_tat",
        };
        match view.history.last() {
            None => match self.opening {
                Opening::Cooperate => cooperative(name, view),
                Opening::Defect => defecting(name, view),
            },
            Some(last) if view.moves.contains(&last.agent) => Ok(last.agent.clone()),
            Some(_) => cooperative(name, view),
        }
    }
}

// Tit for Two Tats defects only after two defections in a row:
// https://en.wikipedia.org/wiki/Tit_for_tat#Tit_for_two_tats
const TATS: usize = 2;

pub struct TitForTwoTats;

impl Strategy for TitForTwoTats {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let defect = defecting("tit_for_two_tats", view)?;
        let recent: Vec<&Turn> = view.history.iter().rev().take(TATS).collect();
        if recent.len() == TATS && recent.iter().all(|turn| turn.agent == defect) {
            return Ok(defect);
        }
        cooperative("tit_for_two_tats", view)
    }
}

/// Cooperates until the agent defects once, then defects for good.
pub struct Grudger;

impl Strategy for Grudger {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let defect = defecting("grudger", view)?;
        if view.history.iter().any(|turn| turn.agent == defect) {
            return Ok(defect);
        }
        cooperative("grudger", view)
    }
}

/// Win-stay, lose-shift: cooperates after a round both seats played alike,
/// defects after one they did not.
pub struct Pavlov;

impl Strategy for Pavlov {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        match view.history.last() {
            Some(last) if last.own != last.agent => defecting("pavlov", view),
            _ => cooperative("pavlov", view),
        }
    }
}

/// Tit for Tat that answers a defection with cooperation at the declared
/// forgiveness probability.
pub struct GenerousTitForTat {
    pub forgive: Bernoulli,
}

impl Strategy for GenerousTitForTat {
    fn choose(&mut self, view: &View<'_>, rng: &mut dyn RngCore) -> Result<String> {
        let defect = defecting("generous_tit_for_tat", view)?;
        match view.history.last() {
            Some(last) if last.agent == defect && !self.forgive.sample(rng) => Ok(defect),
            _ => cooperative("generous_tit_for_tat", view),
        }
    }
}

/// Cooperates while the agent's cooperations outnumber its other moves.
pub struct Adaptive;

impl Strategy for Adaptive {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let cooperate = cooperative("adaptive", view)?;
        if view.history.is_empty() {
            return Ok(cooperate);
        }
        let cooperations = view.history.iter().filter(|turn| turn.agent == cooperate).count();
        if cooperations > view.history.len() - cooperations {
            return Ok(cooperate);
        }
        defecting("adaptive", view)
    }
}

/// Cooperates with the declared probability each round, whatever happened.
pub struct Mixed {
    pub cooperate: Bernoulli,
}

impl Strategy for Mixed {
    fn choose(&mut self, view: &View<'_>, rng: &mut dyn RngCore) -> Result<String> {
        if self.cooperate.sample(rng) {
            return cooperative("mixed", view);
        }
        defecting("mixed", view)
    }
}
