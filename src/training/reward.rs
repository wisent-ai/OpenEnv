//! The reward of an answer to a training prompt: the move it names, paid in
//! the state the prompt shows, averaged over every move the opponent could
//! answer with. A game whose payoffs move with the episode (an adaptive game)
//! is replayed through the state's earlier rounds first, on the state's own
//! seed, so the move is paid as it would be at that point.

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
pub fn expected(library: &Library, settings: &Settings, state: &State, played: &str) -> Result<f64> {
    let probe = library.build(&state.game, settings)?;
    let support = probe.opponent_moves().to_vec();
    if support.is_empty() {
        return Err(Error::NoMoves { strategy: "the opponent".to_owned(), game: state.game.clone() });
    }
    let mut total = NONE;
    for answer in &support {
        let game = library.build(&state.game, settings)?;
        let mut rng = StdRng::seed_from_u64(state.seed);
        for (mine, theirs) in &state.history {
            game.pay(mine, theirs, &mut rng)?;
        }
        let (paid, _) = game.pay(played, answer, &mut rng)?;
        total += paid;
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
        Ok(Self {
            library: Library::standard(),
            states: states.into_iter().map(|state| (state.prompt.clone(), state)).collect(),
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
            Err(Error::Unparsed { .. }) => Ok(Scored { reward: self.unparsed, played: None }),
            Err(other) => Err(other),
        }
    }
}
