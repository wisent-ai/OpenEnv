//! Every refusal KantBench answers with. Each one names what was asked for,
//! what was found instead, and where it was looked for, so a caller can repair
//! the request without reading the source.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{origin} is not JSON: {source}")]
    Json {
        origin: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{origin} must be a JSON object at its top level")]
    NotAnObject { origin: String },
    #[error("{scope} declares no {name}; the settings document must declare it")]
    Undeclared { scope: String, name: String },
    #[error("{scope}.{name} must be {expected}, not {found}")]
    Malformed {
        scope: String,
        name: String,
        expected: String,
        found: String,
    },
    #[error("no game is named {key}; the library holds {known}")]
    UnknownGame { key: String, known: String },
    #[error("no opponent strategy is named {name}; the library holds {known}")]
    UnknownStrategy { name: String, known: String },
    #[error("no variant is named {name}; the library holds {known}")]
    UnknownVariant { name: String, known: String },
    #[error("{game} has no payoff for {player} against {opponent}")]
    NoPayoff {
        game: String,
        player: String,
        opponent: String,
    },
    #[error("{action} is not a move of {game}; its moves are {allowed}")]
    InvalidAction {
        game: String,
        action: String,
        allowed: String,
    },
    #[error("{action} carries no amount after its last underscore")]
    NoAmount { action: String },
    #[error("{game}: {reason}")]
    Unsupported { game: String, reason: String },
    #[error("no episode is running: reset the environment before stepping it")]
    NotStarted,
    #[error("episode {episode} is over after {rounds} rounds: reset the environment to play again")]
    Finished { episode: String, rounds: usize },
    #[error("the opponent agent failed: {reason}")]
    Opponent { reason: String },
    #[error("{strategy} cannot choose from an empty move list in {game}")]
    NoMoves { strategy: String, game: String },
    #[error("{0}")]
    Usage(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
