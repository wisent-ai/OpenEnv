//! The negotiation step: proposals and responses form the round's
//! coalitions, accepted proposals move seats in or out of play, and
//! governance runs.

use crate::error::{Error, Result};
use crate::group::{AGENT_SEAT, AGENT_SEATS};

use super::super::models::{ActiveCoalition, CoalitionAction, CoalitionObservation, CoalitionProposal, Phase};
use super::CoalitionEnvironment;

impl CoalitionEnvironment {
    /// Answer the pending proposals, make the agent's own, and take part in
    /// governance. A pending proposal that names the agent forms only if the
    /// agent accepts it; the agent's proposal forms only if every other
    /// member's strategy accepts it.
    pub fn negotiate(&mut self, action: &CoalitionAction) -> Result<CoalitionObservation> {
        if self.phase != Phase::Negotiate {
            return Err(Error::Usage(
                "the episode is not negotiating: play the action step first, or reset".to_owned(),
            ));
        }
        let mut formed = Vec::new();
        for (index, proposal) in self.pending.iter().enumerate() {
            let asks_agent = proposal.members.contains(&AGENT_SEAT) && proposal.proposer != AGENT_SEAT;
            let agent_accepted = action
                .responses
                .iter()
                .any(|response| response.proposal_index == index && response.accepted);
            if !asks_agent || agent_accepted {
                formed.push(proposal.clone());
            }
        }
        for proposal in &action.proposals {
            if self.others_accept(proposal)? {
                formed.push(proposal.clone());
            }
        }
        let mut proposals = std::mem::take(&mut self.pending);
        proposals.extend(action.proposals.iter().cloned());
        for proposal in &formed {
            if let Some(target) = proposal.exclude_target {
                self.active.remove(&target);
            }
            if let Some(target) = proposal.include_target {
                self.active.insert(target);
            }
        }
        self.coalitions = formed
            .into_iter()
            .map(|proposal| ActiveCoalition {
                members: proposal.members,
                agreed_action: proposal.agreed_action,
                side_payment: proposal.side_payment,
            })
            .collect();
        self.round_proposals = proposals;
        self.round_responses = action.responses.clone();
        self.govern(action)?;
        self.phase = Phase::Action;
        self.observation(AGENT_SEAT, None)
    }

    fn others_accept(&mut self, proposal: &CoalitionProposal) -> Result<bool> {
        for member in &proposal.members {
            if *member == proposal.proposer || *member == AGENT_SEAT {
                continue;
            }
            let seen = self.observation(*member, None)?;
            let Some(strategy) = member
                .checked_sub(AGENT_SEATS)
                .and_then(|index| self.strategies.get_mut(index))
            else {
                return Ok(false);
            };
            if !strategy.respond(&seen, proposal, &mut self.rng) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn govern(&mut self, action: &CoalitionAction) -> Result<()> {
        let active = self.active.clone();
        let governance = self.governance.as_mut().ok_or(Error::NotStarted)?;
        let mut proposals = action.governance_proposals.clone();
        for (index, voter) in self.voters.iter().enumerate() {
            let seat = index + AGENT_SEATS;
            if active.contains(&seat) {
                proposals.extend(voter.propose(seat));
            }
        }
        governance.submit(proposals, &active)?;
        let mut votes = action.governance_votes.clone();
        for (index, voter) in self.voters.iter().enumerate() {
            let seat = index + AGENT_SEATS;
            if active.contains(&seat) {
                votes.extend(voter.vote(seat, governance.pending(), &mut self.rng));
            }
        }
        governance.tally(votes, &active)?;
        Ok(())
    }

    /// Take a seat out of play (negotiation only); it is paid nothing while out.
    pub fn remove_player(&mut self, seat: usize) -> Result<()> {
        self.check_seat(seat)?;
        if !self.active.remove(&seat) {
            return Err(Error::Usage(format!("seat {seat} is already out of play")));
        }
        Ok(())
    }

    /// Bring a seat back into play (negotiation only).
    pub fn add_player(&mut self, seat: usize) -> Result<()> {
        self.check_seat(seat)?;
        if !self.active.insert(seat) {
            return Err(Error::Usage(format!("seat {seat} is already in play")));
        }
        Ok(())
    }

    fn check_seat(&self, seat: usize) -> Result<()> {
        if self.phase != Phase::Negotiate {
            return Err(Error::Usage("seats are added and removed only while negotiating".to_owned()));
        }
        let players = self.game.as_ref().ok_or(Error::NotStarted)?.players;
        if seat >= players {
            return Err(Error::Usage(format!("seat {seat} is not one of the {players} seats")));
        }
        Ok(())
    }
}
