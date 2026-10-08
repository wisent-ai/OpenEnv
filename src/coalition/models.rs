//! The documents of coalition play: a proposal, a response, a coalition in
//! force, one round's record, what a seat observes, and what the agent sends
//! in the negotiation step.

use serde::{Deserialize, Serialize};

use crate::governance::{GovernanceProposal, GovernanceResult, GovernanceVote, Rules};
use crate::group::environment::GroupObservation;
use crate::group::Enforcement;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoalitionProposal {
    pub proposer: usize,
    /// Every seat in the coalition, the proposer included.
    pub members: Vec<usize>,
    pub agreed_action: String,
    /// What the proposer pays each other member; nothing when absent.
    #[serde(default)]
    pub side_payment: Option<f64>,
    /// A seat the coalition removes from play when the proposal is accepted.
    #[serde(default)]
    pub exclude_target: Option<usize>,
    /// A seat the coalition brings back into play when it is accepted.
    #[serde(default)]
    pub include_target: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoalitionResponse {
    pub responder: usize,
    pub proposal_index: usize,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveCoalition {
    pub members: Vec<usize>,
    pub agreed_action: String,
    pub side_payment: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoalitionRound {
    pub round_number: usize,
    pub proposals: Vec<CoalitionProposal>,
    pub responses: Vec<CoalitionResponse>,
    pub active_coalitions: Vec<ActiveCoalition>,
    pub defectors: Vec<usize>,
    pub penalties: Vec<f64>,
    pub side_payments: Vec<f64>,
}

/// Where a coalition round stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Negotiate,
    Action,
    Done,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoalitionObservation {
    pub base: GroupObservation,
    pub phase: Phase,
    pub active_coalitions: Vec<ActiveCoalition>,
    /// Proposals from the other seats awaiting the agent's response.
    pub pending_proposals: Vec<CoalitionProposal>,
    pub coalition_history: Vec<CoalitionRound>,
    pub enforcement: Enforcement,
    /// Scores after penalties, side payments and governance.
    pub adjusted_scores: Vec<f64>,
    pub active_players: Vec<usize>,
    pub current_rules: Rules,
    pub pending_governance: Vec<GovernanceProposal>,
    pub governance_history: Vec<GovernanceResult>,
}

/// The agent's negotiation step: its responses to pending proposals, its own
/// proposals, and its part in governance.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CoalitionAction {
    #[serde(default)]
    pub proposals: Vec<CoalitionProposal>,
    #[serde(default)]
    pub responses: Vec<CoalitionResponse>,
    #[serde(default)]
    pub governance_proposals: Vec<GovernanceProposal>,
    #[serde(default)]
    pub governance_votes: Vec<GovernanceVote>,
}
