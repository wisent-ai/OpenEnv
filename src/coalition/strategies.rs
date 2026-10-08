//! How the seats the agent does not hold negotiate and move in a coalition
//! game. None of them proposes; they differ in what they accept and whether
//! they keep the agreed move. A seat with no agreement plays the game's
//! first move; breaking an agreement means playing the first move that is
//! not the agreed one.

use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

use crate::error::{Error, Result};

use super::models::{CoalitionObservation, CoalitionProposal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Accepts at even odds and moves at random.
    Random,
    /// Accepts everything and keeps every agreement.
    Loyal,
    /// Accepts everything and breaks every agreement.
    Betrayer,
    /// Keeps its agreement unless another seat defected last round.
    Conditional,
    /// Accepts and keeps its agreement unless another seat defected last
    /// round.
    TitForTat,
    /// Accepts and keeps agreements until any other seat ever defects, then
    /// refuses and breaks them for good.
    GrimTrigger,
}

pub const NAMES: &[&str] = &[
    "coalition_random",
    "coalition_loyal",
    "coalition_betrayer",
    "coalition_conditional",
    "coalition_tit_for_tat",
    "coalition_grim_trigger",
];

pub struct CoalitionStrategy {
    kind: Kind,
    triggered: bool,
}

impl CoalitionStrategy {
    pub fn named(name: &str) -> Result<Self> {
        let kind = match name {
            "coalition_random" => Kind::Random,
            "coalition_loyal" => Kind::Loyal,
            "coalition_betrayer" => Kind::Betrayer,
            "coalition_conditional" => Kind::Conditional,
            "coalition_tit_for_tat" => Kind::TitForTat,
            "coalition_grim_trigger" => Kind::GrimTrigger,
            other => {
                return Err(Error::UnknownStrategy {
                    name: other.to_owned(),
                    known: NAMES.join(", "),
                })
            }
        };
        Ok(Self { kind, triggered: false })
    }

    fn others_defected_last(observation: &CoalitionObservation) -> bool {
        let seat = observation.base.player_index;
        observation
            .coalition_history
            .last()
            .is_some_and(|round| round.defectors.iter().any(|defector| *defector != seat))
    }

    fn others_ever_defected(observation: &CoalitionObservation) -> bool {
        let seat = observation.base.player_index;
        observation
            .coalition_history
            .iter()
            .any(|round| round.defectors.iter().any(|defector| *defector != seat))
    }

    pub fn respond(&mut self, observation: &CoalitionObservation, _proposal: &CoalitionProposal, rng: &mut dyn RngCore) -> bool {
        match self.kind {
            Kind::Random => rng.gen::<bool>(),
            Kind::Loyal | Kind::Betrayer | Kind::Conditional => true,
            Kind::TitForTat => !Self::others_defected_last(observation),
            Kind::GrimTrigger => {
                self.triggered |= Self::others_ever_defected(observation);
                !self.triggered
            }
        }
    }

    pub fn choose(&mut self, observation: &CoalitionObservation, rng: &mut dyn RngCore) -> Result<String> {
        let moves = &observation.base.available_actions;
        let first = moves.first().cloned().ok_or_else(|| Error::NoMoves {
            strategy: "coalition strategy".to_owned(),
            game: observation.base.game_name.clone(),
        })?;
        let seat = observation.base.player_index;
        let agreed = observation
            .active_coalitions
            .iter()
            .find(|coalition| coalition.members.contains(&seat))
            .map(|coalition| coalition.agreed_action.clone());
        let other_than_agreed = agreed
            .as_ref()
            .and_then(|agreed| moves.iter().find(|played| *played != agreed).cloned());
        let breaking = match other_than_agreed {
            Some(other) => other,
            None => first.clone(),
        };
        let keeping = match agreed {
            Some(agreed) if moves.contains(&agreed) => agreed,
            _ => first,
        };
        Ok(match self.kind {
            Kind::Random => moves.choose(rng).cloned().ok_or_else(|| Error::NoMoves {
                strategy: "coalition_random".to_owned(),
                game: observation.base.game_name.clone(),
            })?,
            Kind::Loyal => keeping,
            Kind::Betrayer => breaking,
            Kind::Conditional | Kind::TitForTat if Self::others_defected_last(observation) => breaking,
            Kind::Conditional | Kind::TitForTat => keeping,
            Kind::GrimTrigger => {
                self.triggered |= Self::others_ever_defected(observation);
                if self.triggered {
                    breaking
                } else {
                    keeping
                }
            }
        })
    }
}
