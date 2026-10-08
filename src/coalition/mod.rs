//! Coalition play over a group game: each round opens with a negotiation step
//! (proposals, responses, governance) and closes with an action step in which
//! every seat moves. Agreements bind as the game's enforcement (which
//! governance may change) says: binding forces the agreed move, penalty fines
//! a defector, cheap talk leaves it be.

mod episode;
pub mod models;
mod payoffs;
pub mod strategies;

pub use episode::{CoalitionEnvironment, CoalitionReset};
pub use models::{ActiveCoalition, CoalitionAction, CoalitionObservation, CoalitionProposal, CoalitionResponse, CoalitionRound, Phase};
