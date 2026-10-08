//! Cooperative, social choice, repeated and evolutionary games.

mod amounts;
mod repeated;
mod voting;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    voting::register(library);
    repeated::register(library);
    amounts::register(library);
}
