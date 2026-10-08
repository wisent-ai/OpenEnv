//! Reputation across episodes: a store of what is known about each opponent
//! (a cooperation score blended over its episodes, how many there were, the
//! gossip ratings it received), kept in a file the run names so it carries
//! from one run to the next. The environment wrapper shows the agent its
//! opponent's reputation before and during an episode, records any gossip the
//! agent's moves carry, and records the episode when it ends.
//!
//! The store's numbers come from the settings document's `reputation`
//! section: `prior`, the score of an opponent with no record, and `decay`,
//! the weight an old score keeps when an episode is blended in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::env::{Agent, Environment, GameAction, GameObservation, Reset};
use crate::error::{Error, Result};
use crate::settings::Settings;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gossip {
    pub rater: String,
    pub rating: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reputation {
    pub score: f64,
    pub cooperation_rate: f64,
    pub interaction_count: usize,
    pub gossip_history: Vec<Gossip>,
    /// Each recorded episode's cooperation rate, oldest first.
    #[serde(default)]
    pub episodes: Vec<f64>,
}

pub struct ReputationStore {
    path: PathBuf,
    prior: f64,
    decay: f64,
    known: BTreeMap<String, Reputation>,
}

impl ReputationStore {
    /// The store kept at `path` (empty when the file does not exist yet).
    pub fn open(settings: &Settings, path: &Path) -> Result<Self> {
        let declared = settings.section("reputation")?;
        let known = match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(|source| Error::Json {
                origin: path.display().to_string(),
                source,
            })?,
            Err(missing) if missing.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(source) => return Err(Error::Io { path: path.to_path_buf(), source }),
        };
        Ok(Self {
            path: path.to_path_buf(),
            prior: declared.number("prior")?,
            decay: declared.number("decay")?,
            known,
        })
    }

    /// What is known of `opponent`, or the prior when nothing is.
    pub fn get(&self, opponent: &str) -> Reputation {
        match self.known.get(opponent) {
            Some(known) => known.clone(),
            None => Reputation {
                score: self.prior,
                cooperation_rate: self.prior,
                interaction_count: Default::default(),
                gossip_history: Vec::new(),
                episodes: Vec::new(),
            },
        }
    }

    pub fn record_gossip(&mut self, rater: &str, target: &str, rating: &str) {
        let mut known = self.get(target);
        known.gossip_history.push(Gossip { rater: rater.to_owned(), rating: rating.to_owned() });
        self.known.insert(target.to_owned(), known);
    }

    /// Blend one episode's cooperation rate into `opponent`'s score by
    /// exponential smoothing, the old score keeping `decay` of its weight.
    pub fn record_episode(&mut self, opponent: &str, cooperation_rate: f64) {
        let mut known = self.get(opponent);
        let blended = known.cooperation_rate * self.decay + cooperation_rate * (WHOLE - self.decay);
        known.score = blended;
        known.cooperation_rate = blended;
        known.episodes.push(cooperation_rate);
        known.interaction_count = known.episodes.len();
        self.known.insert(opponent.to_owned(), known);
    }

    pub fn save(&self) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.known).map_err(|source| Error::Json {
            origin: self.path.display().to_string(),
            source,
        })?;
        std::fs::write(&self.path, text).map_err(|source| Error::Io { path: self.path.clone(), source })
    }
}

// Exponential smoothing weighs the new observation α and the old score
// (1 − α); the two weights make one whole:
// https://en.wikipedia.org/wiki/Exponential_smoothing
const WHOLE: f64 = 1.0;

/// An environment that knows its opponents' reputations.
pub struct ReputationEnvironment {
    env: Environment,
    store: ReputationStore,
    agent: String,
    opponent: String,
}

impl ReputationEnvironment {
    pub fn new(env: Environment, store: ReputationStore) -> Self {
        Self { env, store, agent: String::new(), opponent: String::new() }
    }

    fn with_reputation(&self, mut observation: GameObservation) -> Result<GameObservation> {
        let known = self.store.get(&self.opponent);
        observation.metadata.insert("interaction_count".to_owned(), Value::from(known.interaction_count));
        let value = serde_json::to_value(known).map_err(|source| Error::Json { origin: "reputation".to_owned(), source })?;
        observation.metadata.insert("opponent_reputation".to_owned(), value);
        Ok(observation)
    }

    /// Start an episode as `agent`; the opponent is known by its strategy name.
    pub fn reset(&mut self, request: &Reset, agent: &str, opponent: Option<Box<dyn Agent>>) -> Result<GameObservation> {
        self.agent = agent.to_owned();
        self.opponent = match &request.strategy {
            Some(name) => name.clone(),
            None => "agent".to_owned(),
        };
        let observation = self.env.reset(request, opponent)?;
        self.with_reputation(observation)
    }

    pub fn step(&mut self, action: &GameAction) -> Result<GameObservation> {
        let game = self.env.game().cloned().ok_or(Error::NotStarted)?;
        if let (Some(tagged), Some(base)) = (action.action.strip_prefix("gossip_"), game.base_move(&action.action)) {
            if let Some(rating) = tagged.strip_suffix(&format!("_{base}")) {
                self.store.record_gossip(&self.agent, &self.opponent, rating);
            }
        }
        let observation = self.env.step(action)?;
        if observation.done {
            let rounds = observation.history.len();
            let cooperated = observation.history.iter().filter(|round| game.cooperated(&round.player_action)).count();
            self.store.record_episode(&self.opponent, cooperated as f64 / rounds as f64);
            self.store.save()?;
        }
        self.with_reputation(observation)
    }
}
