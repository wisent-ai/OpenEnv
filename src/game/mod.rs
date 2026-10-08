//! A game: its moves, how a pair of moves pays, and how many rounds an episode
//! lasts. The library holds every game KantBench can build; a game is built
//! only from the numbers the run's settings document declares for it
//! (`games.<key>`), and the built game keeps those numbers so a result can
//! record what was played.

mod amounts;
pub(crate) mod families;
mod matrix;

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::RngCore;
use serde_json::{Map, Value};

pub use amounts::{amount, amounts, mean, NONE, NOTHING};
pub use matrix::{declared_matrix, matrix_between, matrix_entry, matrix_payoff, moves, Matrix};

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
    /// The base game's own moves, before any variant tagged them.
    pub base_actions: Vec<String>,
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
            base_actions: actions.clone(),
            actions,
            kind: kind.to_owned(),
            rounds: Default::default(),
            payoff,
            seats: Seats::Pair,
            variants: Vec::new(),
            base: String::new(),
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

    /// The base move a played move carries: the move itself, or the base move
    /// a variant tagged (`gossip_trustworthy_cooperate` carries `cooperate`).
    /// The longest base move the played move ends with wins, since base moves
    /// may share an ending.
    pub fn base_move<'a>(&'a self, played: &str) -> Option<&'a str> {
        self.base_actions
            .iter()
            .filter(|base| played == base.as_str() || played.ends_with(&format!("_{base}")))
            .max_by_key(|base| base.len())
            .map(String::as_str)
    }

    /// Whether a played move carries the base game's cooperative (first) move.
    pub fn cooperated(&self, played: &str) -> bool {
        self.base_move(played)
            .is_some_and(|base| self.base_actions.first().is_some_and(|first| first == base))
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
pub type Build = Arc<dyn Fn(&Declared<'_>) -> Result<Game> + Send + Sync>;

/// One game the library can build: its family, the names it reads from its
/// declaration besides `rounds`, and its builder.
#[derive(Clone)]
pub struct Entry {
    pub key: &'static str,
    pub family: &'static str,
    pub parameters: &'static [&'static str],
    pub build: Build,
}

impl Entry {
    pub fn new(
        key: &'static str,
        family: &'static str,
        parameters: &'static [&'static str],
        build: impl Fn(&Declared<'_>) -> Result<Game> + Send + Sync + 'static,
    ) -> Self {
        Self {
            key,
            family,
            parameters,
            build: Arc::new(build),
        }
    }
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
        families::register(&mut library);
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

    /// Build `key`: a library game from what `settings` declares under
    /// `games.<key>`; a game the settings declare whole under
    /// `custom_games.<key>`; or a composed key `<variant>_<game>`, the game
    /// built as its own key says and the variant applied with its numbers
    /// from `variants.<variant>`. A key that is both a library game and a
    /// custom game is refused, since nothing could say which one the run
    /// meant.
    pub fn build(&self, key: &str, settings: &Settings) -> Result<Game> {
        let custom = settings.declares("custom_games", key);
        let (declared, mut game) = match (self.entries.get(key), custom) {
            (Some(_), true) => {
                return Err(Error::Usage(format!(
                    "{key} is a library game and custom_games declares it too; rename the custom game"
                )))
            }
            (Some(entry), false) => {
                let declared = settings.entry("games", key)?;
                let game = (entry.build)(&declared)?;
                (declared, game)
            }
            (None, true) => {
                let declared = settings.entry("custom_games", key)?;
                let game = families::made::custom::build(key, &declared)?;
                (declared, game)
            }
            (None, false) => return self.composed(key, settings),
        };
        game.key = key.to_owned();
        if game.base.is_empty() {
            game.base = key.to_owned();
        }
        game.rounds = declared.count("rounds")?;
        game.parameters = declared.values().clone();
        Ok(game)
    }

    fn unknown(&self, key: &str) -> Error {
        Error::UnknownGame {
            key: key.to_owned(),
            known: self.entries.keys().copied().collect::<Vec<_>>().join(", "),
        }
    }

    fn composed(&self, key: &str, settings: &Settings) -> Result<Game> {
        let Some((variant, inner)) = crate::variant::outermost(key) else {
            return Err(self.unknown(key));
        };
        let base = self.build(inner, settings)?;
        let mut game = crate::variant::apply(base, variant, &|| settings.entry("variants", variant))?;
        if variant == "free_chat" {
            game.name = format!("Free-chat {}", game.name);
            game.description = format!(
                "{} In this variant each player additionally sends a free-form natural-language message to the opponent before acting; the opponent sees the verbatim message. Messages are non-binding cheap talk and do not affect payoff.",
                game.description
            );
        }
        game.key = key.to_owned();
        Ok(game)
    }
}
