//! KantBench: game-theory environments, opponent strategies, variants and
//! evaluation for language-model agents. Every number that decides how a game
//! pays or how a run is scored comes from the settings document the run
//! declares (`settings`), never from this crate.

pub mod cli;
pub mod env;
pub mod error;
pub mod game;
pub mod settings;
pub mod strategy;

pub use error::{Error, Result};
