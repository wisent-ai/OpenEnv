//! Information, signaling, contract, communication and network games.

mod beliefs;
mod communication;
mod contracts;
mod signaling;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    signaling::register(library);
    contracts::register(library);
    communication::register(library);
    beliefs::register(library);
}
