//! A game: its moves, how a pair of moves pays, and how many rounds an episode
//! lasts. The library holds every game KantBench can build; a game is built
//! only from the numbers the run's settings document declares for it
//! (`games.<key>`), and the built game keeps those numbers so a result can
//! record what was played.

mod classic;
mod matrix;

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::RngCore;
use serde_json::{Map, Value};

pub use matrix::{matrix_payoff, Matrix};

use crate::error::{Error, Result};
use crate::settings::{Declared, Settings};

/// How a pair of moves pays: `(player, opponent)`. The random source is the
/// episode's own, so a variant that adds noise draws from the seeded stream.
pub type Payoff = Arc<dyn Fn(&str, &str, &mut dyn RngCore) -> Result<(f64, f64)> + Send + Sync>;

/// Who plays a game: an agent and one opponent, or a group of a declared size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Seats {
    Pair,
    Group(usize),
}

/// Who answers the agent's move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpponentMode {
    /// A named opponent strategy from the library.
    Strategy,
    /// The same model plays both seats.
    SelfPlay,
    /// A different model plays the opponent's seat.
    CrossModel,
}

/// The opponent's moves: the agent's own, or a list of its own (a responder
/// accepts or rejects; a trustee returns an amount).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpponentMoves {
    Shared,
    Own(Vec<String>),
}

/// A penalty a governed game charges, as a fraction of the payoff.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Penalty {
    pub numerator: i64,
    pub denominator: i64,
}

#[derive(Clone)]
pub struct Game {
    pub key: String,
    pub name: String,
    pub description: String,
    pub actions: Vec<String>,
    /// The game's mechanic, as strategies and prompts read it: `matrix`,
    /// `ultimatum`, `trust`, `public_goods`, `auction`, …
    pub kind: String,
    pub rounds: usize,
    pub payoff: Payoff,
    pub seats: Seats,
    pub variants: Vec<String>,
    pub base: String,
    pub enforcement: String,
    pub penalty: Option<Penalty>,
    pub side_payments: bool,
    pub opponent_mode: OpponentMode,
    pub opponent_actions: OpponentMoves,
    /// The values the settings document declared for this game.
    pub parameters: Map<String, Value>,
    /// The opponent moves after seeing the agent's move this round (a
    /// responder sees the offer, a trustee the investment).
    pub responds: bool,
}

impl Game {
    /// A two-seat game played against a strategy, with no variant applied.
    pub fn new(name: &str, description: &str, kind: &str, actions: Vec<String>, payoff: Payoff) -> Self {
        Self {
            key: String::new(),
            name: name.to_owned(),
            description: description.to_owned(),
            actions,
            kind: kind.to_owned(),
            rounds: Default::default(),
            payoff,
            seats: Seats::Pair,
            variants: Vec::new(),
            base: String::new(),
            enforcement: String::new(),
            penalty: None,
            side_payments: false,
            opponent_mode: OpponentMode::Strategy,
            opponent_actions: OpponentMoves::Shared,
            parameters: Map::new(),
            responds: false,
        }
    }

    pub fn has_variant(&self, variant: &str) -> bool {
        self.variants.iter().any(|applied| applied == variant)
    }

    pub fn pay(&self, player: &str, opponent: &str, rng: &mut dyn RngCore) -> Result<(f64, f64)> {
        (self.payoff)(player, opponent, rng)
    }

    /// The moves the opponent's seat may answer with.
    pub fn opponent_moves(&self) -> &[String] {
        match &self.opponent_actions {
            OpponentMoves::Shared => &self.actions,
            OpponentMoves::Own(moves) => moves,
        }
    }

    pub fn summary(&self) -> Value {
        serde_json::json!({
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "actions": self.actions,
            "opponent_actions": self.opponent_moves(),
            "rounds": self.rounds,
            "seats": self.seats,
            "variants": self.variants,
            "base": self.base,
            "opponent_mode": self.opponent_mode,
            "parameters": self.parameters,
        })
    }
}

/// How one library entry turns its declared numbers into a game.
pub type Build = fn(&Declared<'_>) -> Result<Game>;

/// One game the library can build: its family, the names it reads from its
/// declaration besides `rounds`, and its builder.
#[derive(Clone, Copy)]
pub struct Entry {
    pub key: &'static str,
    pub family: &'static str,
    pub parameters: &'static [&'static str],
    pub build: Build,
}

/// Every game KantBench can build, keyed by the name a run asks for.
#[derive(Clone, Default)]
pub struct Library {
    entries: BTreeMap<&'static str, Entry>,
}

impl Library {
    /// The full library: each family registers its games.
    pub fn standard() -> Self {
        let mut library = Self::default();
        classic::register(&mut library);
        library
    }

    pub fn add(&mut self, entry: Entry) {
        self.entries.insert(entry.key, entry);
    }

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.entries.values()
    }

    pub fn entry(&self, key: &str) -> Result<&Entry> {
        self.entries.get(key).ok_or_else(|| Error::UnknownGame {
            key: key.to_owned(),
            known: self.entries.keys().copied().collect::<Vec<_>>().join(", "),
        })
    }

    /// Build `key` from what `settings` declares under `games.<key>`.
    pub fn build(&self, key: &str, settings: &Settings) -> Result<Game> {
        let entry = self.entry(key)?;
        let declared = settings.entry("games", key)?;
        let mut game = (entry.build)(&declared)?;
        game.key = key.to_owned();
        if game.base.is_empty() {
            game.base = key.to_owned();
        }
        game.rounds = declared.count("rounds")?;
        game.parameters = declared.values().clone();
        Ok(game)
    }
}

// The smallest amount a contribution, offer or investment can be: the
// KantBench paper's games take an amount from nothing up to the endowment,
// `x ∈ [0, E]`:
// https://github.com/wisent-ai/OpenEnv/blob/main/paper/sections/games/library.tex
pub const NOTHING: u64 = 0;

/// The moves `<prefix>_<amount>` for every amount from nothing to `most`.
pub fn amounts(prefix: &str, most: u64) -> Vec<String> {
    (NOTHING..=most).map(|amount| format!("{prefix}_{amount}")).collect()
}

/// The amount a move like `offer_5` carries after its last underscore.
pub fn amount(action: &str) -> Result<u64> {
    action
        .rsplit_once('_')
        .and_then(|(_, amount)| amount.parse().ok())
        .ok_or_else(|| Error::NoAmount {
            action: action.to_owned(),
        })
}
