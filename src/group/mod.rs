//! Games of a group: every seat moves at once and each is paid from the
//! whole vector of moves. Seat zero is the agent; the others are played by
//! group strategies or by other agents. A game's size, rounds and payoff
//! numbers come from `games.<key>` in the settings document.

mod coalitions;
pub mod environment;
mod games;
pub mod strategies;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::settings::{Declared, Settings};

/// Every seat's payoff from every seat's move, in seat order.
pub type GroupPayoff = Arc<dyn Fn(&[String]) -> Result<Vec<f64>> + Send + Sync>;

// The agent holds exactly one seat, seat zero; every other seat is played
// for it (README, "Group games"):
// https://github.com/wisent-ai/OpenEnv/blob/main/README.md#group-games
pub const AGENT_SEATS: usize = 1;
// The agent's seat is seat zero (README, "Group games"):
// https://github.com/wisent-ai/OpenEnv/blob/main/README.md#group-games
pub const AGENT_SEAT: usize = 0;

/// How a coalition's agreement binds its members.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enforcement {
    /// Agreements are talk; a defector keeps its payoff.
    CheapTalk,
    /// A defector loses the declared share of its payoff.
    Penalty,
    /// Members are made to play the agreed move.
    Binding,
}

impl Enforcement {
    pub fn named(name: &str) -> Result<Self> {
        serde_json::from_value(Value::String(name.to_owned())).map_err(|_| {
            Error::Usage(format!("enforcement {name} is not one of cheap_talk, penalty, binding"))
        })
    }
}

#[derive(Clone)]
pub struct GroupGame {
    pub key: String,
    pub name: String,
    pub description: String,
    pub actions: Vec<String>,
    pub kind: String,
    pub players: usize,
    pub rounds: usize,
    pub payoff: GroupPayoff,
    pub variants: Vec<String>,
    pub enforcement: Enforcement,
    /// The share of a defector's payoff a penalty takes.
    pub penalty: f64,
    pub side_payments: bool,
    pub parameters: Map<String, Value>,
}

impl GroupGame {
    pub fn new(name: &str, description: &str, kind: &str, actions: Vec<String>, players: usize, payoff: GroupPayoff) -> Self {
        Self {
            key: String::new(),
            name: name.to_owned(),
            description: description.to_owned(),
            actions,
            kind: kind.to_owned(),
            players,
            rounds: Default::default(),
            payoff,
            variants: Vec::new(),
            enforcement: Enforcement::CheapTalk,
            penalty: Default::default(),
            side_payments: false,
            parameters: Map::new(),
        }
    }

    pub fn pay(&self, moves: &[String]) -> Result<Vec<f64>> {
        (self.payoff)(moves)
    }

    pub fn has_variant(&self, variant: &str) -> bool {
        self.variants.iter().any(|applied| applied == variant)
    }

    pub fn is_coalition(&self) -> bool {
        self.kind == "coalition"
    }
}

pub type GroupBuild = Arc<dyn Fn(&Declared<'_>, usize) -> Result<GroupGame> + Send + Sync>;

/// One group game the library can build, with the names it reads besides
/// `players` and `rounds`.
#[derive(Clone)]
pub struct GroupEntry {
    pub key: &'static str,
    pub family: &'static str,
    pub parameters: &'static [&'static str],
    pub build: GroupBuild,
}

impl GroupEntry {
    pub fn new(
        key: &'static str,
        family: &'static str,
        parameters: &'static [&'static str],
        build: impl Fn(&Declared<'_>, usize) -> Result<GroupGame> + Send + Sync + 'static,
    ) -> Self {
        Self {
            key,
            family,
            parameters,
            build: Arc::new(build),
        }
    }
}

#[derive(Clone, Default)]
pub struct GroupLibrary {
    entries: BTreeMap<&'static str, GroupEntry>,
}

impl GroupLibrary {
    pub fn standard() -> Self {
        let mut library = Self::default();
        games::register(&mut library);
        coalitions::register(&mut library);
        library
    }

    pub fn add(&mut self, entry: GroupEntry) {
        self.entries.insert(entry.key, entry);
    }

    pub fn entries(&self) -> impl Iterator<Item = &GroupEntry> {
        self.entries.values()
    }

    /// Build `key` from `games.<key>`: its `players`, `rounds` and payoff
    /// numbers. `free_chat_<key>` builds `key` with a message step each round.
    pub fn build(&self, key: &str, settings: &Settings) -> Result<GroupGame> {
        if let Some(inner) = key.strip_prefix("free_chat_") {
            let mut game = self.build(inner, settings)?;
            game.variants.push("free_chat".to_owned());
            game.name = format!("Free-chat {}", game.name);
            game.description = format!(
                "{} Each player additionally sends a free-form natural-language message every round; messages from all other players appear in the next round verbatim. Messages are non-binding and do not affect payoff.",
                game.description
            );
            game.key = key.to_owned();
            return Ok(game);
        }
        let entry = self.entries.get(key).ok_or_else(|| Error::UnknownGame {
            key: key.to_owned(),
            known: self.entries.keys().copied().collect::<Vec<_>>().join(", "),
        })?;
        let declared = settings.entry("games", key)?;
        let players = declared.count("players")?;
        let mut game = (entry.build)(&declared, players)?;
        game.key = key.to_owned();
        game.rounds = declared.count("rounds")?;
        game.parameters = declared.values().clone();
        Ok(game)
    }
}

/// How many seats played `played`.
pub(crate) fn count(moves: &[String], played: &str) -> usize {
    moves.iter().filter(|chosen| *chosen == played).count()
}
