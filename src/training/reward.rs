//! The reward of an answer to a training prompt: the move it names, paid in
//! the state the prompt shows, averaged over every move the opponent could
//! answer with. A game whose payoffs move with the episode (an adaptive game)
//! is replayed using each recorded round's payoff seed, independently of the
//! random draws that selected opponents' moves or belonged to earlier episodes.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::agent::prompt;
use crate::error::{Error, Result};
use crate::game::{Library, NONE};
use crate::settings::Settings;

use super::State;

/// The expected self-payoff of `played` in `state`.
pub fn expected(
    library: &Library,
    settings: &Settings,
    state: &State,
    played: &str,
) -> Result<f64> {
    let probe = library.build(&state.game, settings)?;
    let support = probe.opponent_moves().to_vec();
    if support.is_empty() {
        return Err(Error::NoMoves {
            strategy: "the opponent".to_owned(),
            game: state.game.clone(),
        });
    }
    let mut total = NONE;
    for answer in &support {
        let game = library.build(&state.game, settings)?;
        for round in &state.history {
            let mut rng = StdRng::seed_from_u64(round.payoff_seed);
            game.pay(&round.player_action, &round.opponent_action, &mut rng)?;
        }
        let mut rng = StdRng::seed_from_u64(state.payoff_seed);
        let (paid, _) = game.pay(played, answer, &mut rng)?;
        if !paid.is_finite() {
            return Err(Error::Usage(format!(
                "game {} produced a non-finite training payoff for {played:?} against {answer:?}: {paid}",
                state.game
            )));
        }
        total += paid;
        if !total.is_finite() {
            return Err(Error::Usage(format!(
                "game {} training payoff sum cannot be represented for {played:?} after opponent move {answer:?}",
                state.game
            )));
        }
    }
    Ok(total / support.len() as f64)
}

/// Every move of `state` with its expected payoff, best first.
pub fn ranked(library: &Library, settings: &Settings, state: &State) -> Result<Vec<(String, f64)>> {
    let game = library.build(&state.game, settings)?;
    let mut scored = game
        .actions
        .iter()
        .map(|played| expected(library, settings, state, played).map(|paid| (played.clone(), paid)))
        .collect::<Result<Vec<_>>>()?;
    scored.sort_by(|(_, better), (_, worse)| worse.total_cmp(better));
    Ok(scored)
}

#[derive(Clone, Debug, Serialize)]
pub struct Scored {
    pub reward: f64,
    /// The move the answer named; absent when it named none.
    #[serde(rename = "move", skip_serializing_if = "Option::is_none")]
    pub played: Option<String>,
}

/// Rewards for answers to the prompts of a dataset.
pub struct Scorer {
    library: Library,
    settings: std::sync::Arc<Settings>,
    states: BTreeMap<String, State>,
    unparsed: f64,
}

impl Scorer {
    /// A scorer over `states`; an answer that names no move earns the run's
    /// declared `training.unparsed_reward`.
    pub fn new(settings: std::sync::Arc<Settings>, states: Vec<State>) -> Result<Self> {
        let unparsed = settings.section("training")?.number("unparsed_reward")?;
        let mut indexed: BTreeMap<String, State> = BTreeMap::new();
        for state in states {
            match indexed.entry(state.prompt.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(state);
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    let previous = entry.get();
                    if previous.game != state.game
                        || previous.history != state.history
                        || previous.seed != state.seed
                        || previous.payoff_seed != state.payoff_seed
                    {
                        return Err(Error::Usage(format!(
                            "training dataset repeats a prompt with different replay state: games {:?} and {:?}, environment seeds {} and {}, payoff seeds {} and {}; a reward request cannot identify which state to score",
                            previous.game, state.game, previous.seed, state.seed,
                            previous.payoff_seed, state.payoff_seed
                        )));
                    }
                }
            }
        }
        Ok(Self {
            library: Library::standard(),
            states: indexed,
            settings,
            unparsed,
        })
    }

    /// The reward for `text` answering `prompt`; a prompt the dataset does
    /// not hold is refused, since nothing says which game it showed.
    pub fn score(&self, prompt_text: &str, text: &str) -> Result<Scored> {
        let state = self.states.get(prompt_text).ok_or_else(|| {
            Error::Usage("the prompt is not one of this dataset's states; score answers to prompts kant dataset wrote".to_owned())
        })?;
        let game = self.library.build(&state.game, &self.settings)?;
        match prompt::parse(text, &game.actions) {
            Ok(played) => Ok(Scored {
                reward: expected(&self.library, &self.settings, state, &played)?,
                played: Some(played),
            }),
            Err(Error::Unparsed { .. }) => Ok(Scored {
                reward: self.unparsed,
                played: None,
            }),
            Err(other) => Err(other),
        }
    }
}
