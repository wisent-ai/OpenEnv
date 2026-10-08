//! The documents of meta-governance: the rules in force, a proposed change,
//! a vote, and the record of one round of governance.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::group::Enforcement;

/// A payoff mechanic the players can switch on, applied in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mechanic {
    /// Each active seat pays `tax_rate` of its payoff into a pool shared
    /// equally.
    Taxation,
    /// `redistribution` `equal` gives every active seat the mean;
    /// `proportional` moves each payoff `damping` of the way to it.
    Redistribution,
    /// Each active seat pays `insurance_contribution` of its payoff into a
    /// pool shared by the seats below `insurance_threshold` times the mean.
    Insurance,
    /// No payoff passes `quota`; the excess goes equally to the seats below.
    Quota,
    /// Seats above `subsidy_floor` pay `subsidy_fund_rate` of their excess
    /// into a pool that lifts the seats below it toward the floor.
    Subsidy,
    /// When seat `veto_player` falls below the mean, every active seat gets
    /// the mean.
    Veto,
}

impl Mechanic {
    pub const ALL: &'static [Mechanic] = &[
        Mechanic::Taxation,
        Mechanic::Redistribution,
        Mechanic::Insurance,
        Mechanic::Quota,
        Mechanic::Subsidy,
        Mechanic::Veto,
    ];
}

/// The rules in force: the game's enforcement and penalty, which mechanics
/// are on with the numbers they read, and which custom modifiers run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rules {
    pub enforcement: Enforcement,
    pub penalty: f64,
    pub side_payments: bool,
    pub mechanics: BTreeMap<Mechanic, bool>,
    /// The numbers mechanics read: the settings document's `governance`
    /// section, with any a proposal changed.
    pub mechanic_config: Map<String, Value>,
    pub custom_modifiers: Vec<String>,
    pub history: Vec<GovernanceResult>,
}

/// One change a proposal asks for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "proposal_type", rename_all = "snake_case")]
pub enum Change {
    /// Set `enforcement` (text), `penalty` (number) or `side_payments`
    /// (true or false).
    Parameter { name: String, value: Value },
    /// Switch a mechanic on or off, optionally changing the numbers it reads.
    Mechanic {
        name: Mechanic,
        active: bool,
        #[serde(default)]
        params: Map<String, Value>,
    },
    /// Switch a registered custom modifier on or off.
    Custom { key: String, active: bool },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GovernanceProposal {
    pub proposer: usize,
    #[serde(flatten)]
    pub change: Change,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GovernanceVote {
    pub voter: usize,
    pub proposal_index: usize,
    pub approve: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GovernanceResult {
    pub proposals: Vec<GovernanceProposal>,
    pub votes: Vec<GovernanceVote>,
    pub adopted: Vec<usize>,
    pub rejected: Vec<usize>,
    /// The enforcement, penalty and mechanics in force after this round.
    pub enforcement: Enforcement,
    pub penalty: f64,
    pub mechanics: BTreeMap<Mechanic, bool>,
}
