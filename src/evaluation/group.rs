//! Group and coalition tournaments: the agent's seat, held by a strategy,
//! plays every named group game with every other seat held by one named
//! strategy, `evaluation.episodes` times each.
//!
//! In a group game the agent and the other seats play group strategies, and
//! each pairing reports the agent's total score and how often it played the
//! game's first, cooperative move. In a coalition game they play coalition
//! strategies (the agent accepts or refuses the proposals naming it as its
//! strategy would, and moves as it would), the other seats vote with the
//! named governance strategy, and each pairing also reports how many rounds
//! formed a coalition, how many saw a defection, and how much governance was
//! proposed and adopted.

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::coalition::{CoalitionAction, CoalitionEnvironment, CoalitionReset, CoalitionResponse, Phase};
use crate::coalition::strategies::CoalitionStrategy;
use crate::env::GameAction;
use crate::error::{Error, Result};
use crate::group::environment::{GroupEnvironment, Seat};
use crate::group::strategies::{self, GroupView};
use crate::group::{GroupLibrary, AGENT_SEAT, AGENT_SEATS};
use crate::settings::Settings;

#[derive(Clone, Debug, Serialize)]
pub struct GroupEpisode {
    pub player_score: f64,
    pub all_scores: Vec<f64>,
    pub rounds_played: usize,
    pub cooperation_rate: f64,
    /// Coalition games only: the share of rounds in which a coalition formed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coalition_rate: Option<f64>,
    /// Coalition games only: the share of rounds in which a member defected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defection_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub governance_proposed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub governance_adopted: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupTournament {
    pub seed: u64,
    pub agent: String,
    /// game → other seats' strategy → episodes.
    pub games: BTreeMap<String, BTreeMap<String, Vec<GroupEpisode>>>,
}

fn share(count: usize, of: usize) -> f64 {
    count as f64 / of as f64
}

pub fn run(settings: Arc<Settings>, agent: &str, games: &[String], strategies: &[String], governance: Option<&str>) -> Result<GroupTournament> {
    if games.is_empty() || strategies.is_empty() {
        return Err(Error::Usage("a group tournament needs at least one game and one strategy for the other seats".to_owned()));
    }
    let episodes = settings.section("evaluation")?.count("episodes")?;
    let library = Arc::new(GroupLibrary::standard());
    let mut results = BTreeMap::new();
    let mut seed = None;
    for key in games {
        let probe = library.build(key, &settings)?;
        let mut by_strategy = BTreeMap::new();
        for strategy in strategies {
            let mut played = Vec::new();
            for _ in std::iter::repeat(()).take(episodes) {
                let (episode, used) = match probe.is_coalition() {
                    true => {
                        let governance = governance.ok_or_else(|| {
                            Error::Usage(format!("{key} is a coalition game: name the other seats' governance strategy with --governance"))
                        })?;
                        coalition_episode(&library, &settings, key, agent, strategy, governance)?
                    }
                    false => group_episode(&library, &settings, key, agent, strategy, probe.players)?,
                };
                seed = Some(used);
                played.push(episode);
            }
            by_strategy.insert(strategy.clone(), played);
        }
        results.insert(key.clone(), by_strategy);
    }
    Ok(GroupTournament {
        seed: seed.ok_or(Error::NotStarted)?,
        agent: agent.to_owned(),
        games: results,
    })
}

fn group_episode(library: &Arc<GroupLibrary>, settings: &Arc<Settings>, key: &str, agent: &str, strategy: &str, players: usize) -> Result<(GroupEpisode, u64)> {
    let mut environment = GroupEnvironment::new(library.clone(), settings.clone())?;
    let seed = environment.seed();
    let seats = Seat::strategies(&[strategy.to_owned()], players.saturating_sub(AGENT_SEATS))?;
    let mut own = strategies::named(agent)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut observation = environment.reset(key, seats, None, None)?;
    while !observation.done {
        let game = environment.game().ok_or(Error::NotStarted)?.clone();
        let view = GroupView { game: &game, seat: AGENT_SEAT, history: &observation.history };
        let played = own.choose(&view, &mut rng)?;
        observation = environment.step(&GameAction::new(&played))?;
    }
    let game = environment.game().ok_or(Error::NotStarted)?;
    let first = game.actions.first().cloned().ok_or(Error::NotStarted)?;
    let rounds = observation.history.len();
    let cooperated = observation.history.iter().filter(|round| round.actions.get(AGENT_SEAT) == Some(&first)).count();
    let player_score = observation.scores.get(AGENT_SEAT).copied().ok_or(Error::NotStarted)?;
    Ok((
        GroupEpisode {
            player_score,
            all_scores: observation.scores,
            rounds_played: rounds,
            cooperation_rate: share(cooperated, rounds),
            coalition_rate: None,
            defection_rate: None,
            governance_proposed: None,
            governance_adopted: None,
        },
        seed,
    ))
}

fn coalition_episode(library: &Arc<GroupLibrary>, settings: &Arc<Settings>, key: &str, agent: &str, strategy: &str, governance: &str) -> Result<(GroupEpisode, u64)> {
    let mut environment = CoalitionEnvironment::new(library.clone(), settings.clone())?;
    let seed = environment.seed();
    let mut own = CoalitionStrategy::named(agent)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let request = CoalitionReset {
        game: key.to_owned(),
        strategies: vec![strategy.to_owned()],
        governance: vec![governance.to_owned()],
        rounds: None,
        episode_id: None,
    };
    let mut observation = environment.reset(&request)?;
    while observation.phase != Phase::Done {
        let responses = observation
            .pending_proposals
            .iter()
            .enumerate()
            .filter(|(_, proposal)| proposal.members.contains(&AGENT_SEAT))
            .map(|(proposal_index, proposal)| CoalitionResponse {
                responder: AGENT_SEAT,
                proposal_index,
                accepted: own.respond(&observation, proposal, &mut rng),
            })
            .collect();
        let negotiated = environment.negotiate(&CoalitionAction { responses, ..CoalitionAction::default() })?;
        let played = own.choose(&negotiated, &mut rng)?;
        observation = environment.act(&GameAction::new(&played))?;
    }
    let history = &observation.coalition_history;
    let rounds = history.len();
    let first = observation.base.available_actions.first().cloned().ok_or(Error::NotStarted)?;
    let cooperated = observation.base.history.iter().filter(|round| round.actions.get(AGENT_SEAT) == Some(&first)).count();
    let proposed = observation.governance_history.iter().map(|round| round.proposals.len()).sum();
    let adopted = observation.governance_history.iter().map(|round| round.adopted.len()).sum();
    let player_score = observation.adjusted_scores.get(AGENT_SEAT).copied().ok_or(Error::NotStarted)?;
    Ok((
        GroupEpisode {
            player_score,
            all_scores: observation.adjusted_scores.clone(),
            rounds_played: rounds,
            cooperation_rate: share(cooperated, rounds),
            coalition_rate: Some(share(history.iter().filter(|round| !round.active_coalitions.is_empty()).count(), rounds)),
            defection_rate: Some(share(history.iter().filter(|round| !round.defectors.is_empty()).count(), rounds)),
            governance_proposed: Some(proposed),
            governance_adopted: Some(adopted),
        },
        seed,
    ))
}
