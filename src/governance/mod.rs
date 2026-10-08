//! Meta-governance over a coalition game: players propose changes to the
//! rules in force (enforcement, penalty, side payments, payoff mechanics,
//! registered custom modifiers), vote on them, and the changes a strict
//! majority of the active seats approves take effect. Its numbers come from
//! the settings document's `governance` section.

mod mechanics;
pub mod models;
pub mod strategies;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::group::{Enforcement, GroupGame};
use crate::settings::Declared;

pub use models::{Change, GovernanceProposal, GovernanceResult, GovernanceVote, Mechanic, Rules};

/// A payoff modifier a caller registers by key; governance can switch it on.
pub type Modifier = Arc<dyn Fn(&[f64], &BTreeSet<usize>) -> Result<Vec<f64>> + Send + Sync>;

pub struct Governance {
    rules: Rules,
    pending: Vec<GovernanceProposal>,
    modifiers: BTreeMap<String, Modifier>,
}

impl Governance {
    /// The rules a game starts with, and `config` (the settings document's
    /// `governance` section) for the numbers mechanics and limits read.
    pub fn new(game: &GroupGame, config: Map<String, Value>) -> Self {
        Self {
            rules: Rules {
                enforcement: game.enforcement,
                penalty: game.penalty,
                side_payments: game.side_payments,
                mechanics: Mechanic::ALL.iter().map(|mechanic| (*mechanic, false)).collect(),
                mechanic_config: config,
                custom_modifiers: Vec::new(),
                history: Vec::new(),
            },
            pending: Vec::new(),
            modifiers: BTreeMap::new(),
        }
    }

    pub fn rules(&self) -> &Rules {
        &self.rules
    }

    pub fn pending(&self) -> &[GovernanceProposal] {
        &self.pending
    }

    pub fn register(&mut self, key: &str, modifier: Modifier) {
        self.modifiers.insert(key.to_owned(), modifier);
    }

    pub fn unregister(&mut self, key: &str) {
        self.modifiers.remove(key);
        self.rules.custom_modifiers.retain(|active| active != key);
    }

    fn config(&self) -> Declared<'_> {
        Declared::over("governance", &self.rules.mechanic_config)
    }

    /// Queue the proposals of active seats, up to the declared
    /// `most_proposals` a round. A proposal from an inactive seat is passed
    /// over; one that names no change governance can make is refused.
    pub fn submit(&mut self, proposals: Vec<GovernanceProposal>, active: &BTreeSet<usize>) -> Result<()> {
        if proposals.is_empty() {
            return Ok(());
        }
        let most = self.config().count("most_proposals")?;
        for proposal in proposals {
            if self.pending.len() >= most {
                break;
            }
            if !active.contains(&proposal.proposer) {
                continue;
            }
            self.validate(&proposal)?;
            self.pending.push(proposal);
        }
        Ok(())
    }

    fn validate(&self, proposal: &GovernanceProposal) -> Result<()> {
        let refuse = |reason: String| Err(Error::Usage(format!("governance proposal by seat {}: {reason}", proposal.proposer)));
        match &proposal.change {
            Change::Parameter { name, value } => match (name.as_str(), value) {
                ("enforcement", Value::String(text)) => Enforcement::named(text).map(|_| ()),
                ("penalty", Value::Number(_)) | ("side_payments", Value::Bool(_)) => Ok(()),
                _ => refuse(format!(
                    "{name} = {value} is not a change governance makes; it sets enforcement (text), penalty (number) or side_payments (true or false)"
                )),
            },
            Change::Mechanic { .. } => Ok(()),
            Change::Custom { key, .. } if self.modifiers.contains_key(key) => Ok(()),
            Change::Custom { key, .. } => refuse(format!("no custom modifier is registered as {key}")),
        }
    }

    /// Count the votes of active seats; a pending proposal a strict majority
    /// of the active seats approves takes effect.
    pub fn tally(&mut self, votes: Vec<GovernanceVote>, active: &BTreeSet<usize>) -> Result<GovernanceResult> {
        let seats = active.len();
        let mut adopted = Vec::new();
        let mut rejected = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        for (index, proposal) in pending.iter().enumerate() {
            let approvals = votes
                .iter()
                .filter(|vote| vote.proposal_index == index && vote.approve && active.contains(&vote.voter))
                .count();
            if approvals > seats - approvals {
                self.adopt(proposal)?;
                adopted.push(index);
            } else {
                rejected.push(index);
            }
        }
        let result = GovernanceResult {
            proposals: pending,
            votes,
            adopted,
            rejected,
            enforcement: self.rules.enforcement,
            penalty: self.rules.penalty,
            mechanics: self.rules.mechanics.clone(),
        };
        self.rules.history.push(result.clone());
        Ok(result)
    }

    fn adopt(&mut self, proposal: &GovernanceProposal) -> Result<()> {
        match &proposal.change {
            Change::Parameter { name, value } => match (name.as_str(), value) {
                ("enforcement", Value::String(text)) => self.rules.enforcement = Enforcement::named(text)?,
                ("penalty", Value::Number(number)) => {
                    self.rules.penalty = number.as_f64().ok_or_else(|| Error::Usage(format!("penalty {number} is not a finite number")))?;
                }
                ("side_payments", Value::Bool(allowed)) => self.rules.side_payments = *allowed,
                _ => return self.validate(proposal),
            },
            Change::Mechanic { name, active, params } => {
                self.rules.mechanics.insert(*name, *active);
                for (key, value) in params {
                    self.rules.mechanic_config.insert(key.clone(), value.clone());
                }
            }
            Change::Custom { key, active } => {
                self.rules.custom_modifiers.retain(|on| on != key);
                if *active {
                    self.rules.custom_modifiers.push(key.clone());
                }
            }
        }
        Ok(())
    }

    /// The payoffs after every mechanic that is on, then every custom
    /// modifier that is on. A custom modifier may move a payoff by at most
    /// the declared `custom_clamp` times its size (or `custom_clamp` itself
    /// when the payoff is smaller than one unit).
    pub fn apply(&self, payoffs: &[f64], active: &BTreeSet<usize>) -> Result<Vec<f64>> {
        let mut result = mechanics::apply(payoffs, active, &self.rules)?;
        if self.rules.custom_modifiers.is_empty() {
            return Ok(result);
        }
        let clamp = self.config().number("custom_clamp")?;
        for key in &self.rules.custom_modifiers {
            let modifier = self.modifiers.get(key).ok_or_else(|| Error::Usage(format!("custom modifier {key} is on but no longer registered")))?;
            let modified = modifier(&result, active)?;
            for (before, after) in result.iter_mut().zip(modified) {
                let reach = (before.abs() * clamp).max(clamp);
                *before = after.clamp(*before - reach, *before + reach);
            }
        }
        Ok(result)
    }
}
