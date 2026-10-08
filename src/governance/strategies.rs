//! How the seats the agent does not hold take part in governance. None of
//! them proposes; they differ in how they vote.

use rand::{Rng, RngCore};

use crate::error::{Error, Result};

use super::models::{GovernanceProposal, GovernanceVote};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernanceStrategy {
    /// Neither proposes nor votes.
    Passive,
    /// Approves or rejects each proposal at even odds.
    Random,
    /// Rejects every proposal.
    Conservative,
    /// Approves every proposal.
    Progressive,
}

pub const NAMES: &[&str] = &[
    "governance_passive",
    "governance_random",
    "governance_conservative",
    "governance_progressive",
];

impl GovernanceStrategy {
    pub fn named(name: &str) -> Result<Self> {
        Ok(match name {
            "governance_passive" => Self::Passive,
            "governance_random" => Self::Random,
            "governance_conservative" => Self::Conservative,
            "governance_progressive" => Self::Progressive,
            other => {
                return Err(Error::UnknownStrategy {
                    name: other.to_owned(),
                    known: NAMES.join(", "),
                })
            }
        })
    }

    pub fn propose(&self, _seat: usize) -> Vec<GovernanceProposal> {
        Vec::new()
    }

    pub fn vote(&self, seat: usize, pending: &[GovernanceProposal], rng: &mut dyn RngCore) -> Vec<GovernanceVote> {
        let decide = |rng: &mut dyn RngCore| match self {
            Self::Passive => None,
            Self::Random => Some(rng.gen::<bool>()),
            Self::Conservative => Some(false),
            Self::Progressive => Some(true),
        };
        pending
            .iter()
            .enumerate()
            .filter_map(|(proposal_index, _)| {
                decide(&mut *rng).map(|approve| GovernanceVote {
                    voter: seat,
                    proposal_index,
                    approve,
                })
            })
            .collect()
    }
}
