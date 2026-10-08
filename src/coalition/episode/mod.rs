//! The coalition environment's episode: negotiation (`negotiation`),
//! governance and the action step over a group environment whose other
//! seats it moves itself.

mod negotiation;

use std::collections::BTreeSet;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::Map;

use crate::env::{round_after, GameAction};
use crate::error::{Error, Result};
use crate::governance::strategies::GovernanceStrategy;
use crate::governance::Governance;
use crate::group::environment::{GroupEnvironment, GroupObservation, Seat};
use crate::group::{Enforcement, GroupGame, GroupLibrary, AGENT_SEAT, AGENT_SEATS};
use crate::settings::Settings;

use super::models::{ActiveCoalition, CoalitionObservation, CoalitionProposal, CoalitionResponse, CoalitionRound, Phase};
use super::payoffs::settle;
use super::strategies::CoalitionStrategy;

/// What to play: a coalition game, and for the other seats their coalition
/// strategies and governance strategies (one name for all, or one each).
#[derive(Clone, Debug, Default)]
pub struct CoalitionReset {
    pub game: String,
    pub strategies: Vec<String>,
    pub governance: Vec<String>,
    pub rounds: Option<usize>,
    pub episode_id: Option<String>,
}

fn per_seat(names: &[String], others: usize, what: &str) -> Result<Vec<String>> {
    match names {
        [only] => Ok(vec![only.clone(); others]),
        many if many.len() == others => Ok(many.to_vec()),
        _ => Err(Error::Usage(format!(
            "name one {what} for every other seat ({others}) or one for all of them, not {}",
            names.len()
        ))),
    }
}

pub struct CoalitionEnvironment {
    group: GroupEnvironment,
    settings: Arc<Settings>,
    rng: StdRng,
    game: Option<GroupGame>,
    strategies: Vec<CoalitionStrategy>,
    voters: Vec<GovernanceStrategy>,
    governance: Option<Governance>,
    phase: Phase,
    coalitions: Vec<ActiveCoalition>,
    pending: Vec<CoalitionProposal>,
    history: Vec<CoalitionRound>,
    adjustments: Vec<f64>,
    active: BTreeSet<usize>,
    round_proposals: Vec<CoalitionProposal>,
    round_responses: Vec<CoalitionResponse>,
    last: Option<GroupObservation>,
}

impl CoalitionEnvironment {
    pub fn new(library: Arc<GroupLibrary>, settings: Arc<Settings>) -> Result<Self> {
        let group = GroupEnvironment::new(library, settings.clone())?;
        let rng = StdRng::seed_from_u64(group.seed());
        Ok(Self {
            group,
            settings,
            rng,
            game: None,
            strategies: Vec::new(),
            voters: Vec::new(),
            governance: None,
            phase: Phase::Done,
            coalitions: Vec::new(),
            pending: Vec::new(),
            history: Vec::new(),
            adjustments: Vec::new(),
            active: BTreeSet::new(),
            round_proposals: Vec::new(),
            round_responses: Vec::new(),
            last: None,
        })
    }

    pub fn seed(&self) -> u64 {
        self.group.seed()
    }

    pub fn governance(&mut self) -> Result<&mut Governance> {
        self.governance.as_mut().ok_or(Error::NotStarted)
    }

    pub fn reset(&mut self, request: &CoalitionReset) -> Result<CoalitionObservation> {
        let probe = GroupLibrary::standard().build(&request.game, &self.settings)?;
        if !probe.is_coalition() {
            return Err(Error::Usage(format!("{} is not a coalition game", request.game)));
        }
        let others = probe.players.saturating_sub(AGENT_SEATS);
        self.strategies = per_seat(&request.strategies, others, "coalition strategy")?
            .iter()
            .map(|name| CoalitionStrategy::named(name))
            .collect::<Result<_>>()?;
        self.voters = per_seat(&request.governance, others, "governance strategy")?
            .iter()
            .map(|name| GovernanceStrategy::named(name))
            .collect::<Result<_>>()?;
        let seats = (AGENT_SEATS..probe.players).map(|_| Seat::Given).collect();
        let first = self.group.reset(&request.game, seats, request.rounds, request.episode_id.clone())?;
        let game = self.group.game().cloned().ok_or(Error::NotStarted)?;
        // Governance numbers are read when a proposal or mechanic needs one,
        // so a run that never governs declares none.
        let config = match self.settings.document().get("governance") {
            Some(_) => self.settings.section("governance")?.values().clone(),
            None => Map::new(),
        };
        self.governance = Some(Governance::new(&game, config));
        self.adjustments = vec![Default::default(); game.players];
        self.active = (AGENT_SEAT..game.players).collect();
        self.coalitions.clear();
        self.history.clear();
        self.pending.clear();
        self.last = Some(first);
        self.game = Some(game);
        self.phase = Phase::Negotiate;
        self.observation(AGENT_SEAT, None)
    }

    /// The action step: every seat moves, agreements bind as the rules say,
    /// and the round is paid after penalties, side payments and governance.
    pub fn act(&mut self, action: &GameAction) -> Result<CoalitionObservation> {
        if self.phase != Phase::Action {
            return Err(Error::Usage("the episode is not at its action step: negotiate first".to_owned()));
        }
        let game = self.game.clone().ok_or(Error::NotStarted)?;
        let rules = self.governance.as_ref().ok_or(Error::NotStarted)?.rules().clone();
        let bound = |seat: usize, coalitions: &[ActiveCoalition]| {
            coalitions
                .iter()
                .find(|coalition| coalition.members.contains(&seat))
                .map(|coalition| coalition.agreed_action.clone())
        };
        let mut moves = vec![match (rules.enforcement, bound(AGENT_SEAT, &self.coalitions)) {
            (Enforcement::Binding, Some(agreed)) => agreed,
            _ => action.action.clone(),
        }];
        let first = game.actions.first().cloned().ok_or(Error::NotStarted)?;
        for index in AGENT_SEAT..self.strategies.len() {
            let seat = index + AGENT_SEATS;
            if !self.active.contains(&seat) {
                moves.push(first.clone());
                continue;
            }
            let seen = self.observation(seat, None)?;
            let chosen = self.strategies[index].choose(&seen, &mut self.rng)?;
            moves.push(match (rules.enforcement, bound(seat, &self.coalitions)) {
                (Enforcement::Binding, Some(agreed)) => agreed,
                _ => chosen,
            });
        }
        let messages = moves.iter().map(|_| String::new()).collect();
        let played = self.group.play(moves.clone(), messages)?;
        let round = played.last_round.clone().ok_or(Error::NotStarted)?;
        let settled = settle(&round.payoffs, &moves, &self.coalitions, rules.enforcement, rules.penalty);
        let governance = self.governance.as_ref().ok_or(Error::NotStarted)?;
        let mut adjusted = governance.apply(&settled.adjusted, &self.active)?;
        for (seat, paid) in adjusted.iter_mut().enumerate() {
            if !self.active.contains(&seat) {
                *paid = crate::game::NONE;
            }
        }
        for ((adjustment, after), before) in self.adjustments.iter_mut().zip(&adjusted).zip(&round.payoffs) {
            *adjustment += after - before;
        }
        self.history.push(CoalitionRound {
            round_number: round_after(self.history.len()),
            proposals: std::mem::take(&mut self.round_proposals),
            responses: std::mem::take(&mut self.round_responses),
            active_coalitions: self.coalitions.clone(),
            defectors: settled.defectors,
            penalties: settled.penalties,
            side_payments: settled.side_payments,
        });
        let reward = adjusted.get(AGENT_SEAT).copied();
        let done = played.done;
        self.last = Some(played);
        if done {
            self.phase = Phase::Done;
        } else {
            self.coalitions.clear();
            self.phase = Phase::Negotiate;
        }
        self.observation(AGENT_SEAT, reward)
    }

    /// The episode as `seat` sees it; `reward` replaces the group reward
    /// after an action step.
    pub fn observation(&self, seat: usize, reward: Option<f64>) -> Result<CoalitionObservation> {
        let last = self.last.as_ref().ok_or(Error::NotStarted)?;
        let mut base = self.group.observation(seat, last.reward, last.last_round.clone())?;
        if let Some(reward) = reward {
            base.reward = reward;
        }
        let governance = self.governance.as_ref().ok_or(Error::NotStarted)?;
        let rules = governance.rules().clone();
        let adjusted_scores = base
            .scores
            .iter()
            .zip(&self.adjustments)
            .map(|(score, adjustment)| score + adjustment)
            .collect();
        Ok(CoalitionObservation {
            base,
            phase: self.phase,
            active_coalitions: self.coalitions.clone(),
            pending_proposals: self.pending.clone(),
            coalition_history: self.history.clone(),
            enforcement: rules.enforcement,
            adjusted_scores,
            active_players: self.active.iter().copied().collect(),
            governance_history: rules.history.clone(),
            pending_governance: governance.pending().to_vec(),
            current_rules: rules,
        })
    }
}
