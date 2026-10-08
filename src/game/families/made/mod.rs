//! Games made rather than taken from the literature: generated from a seed,
//! adapting to the episode's history, or declared whole by a run.

mod adaptive;
pub(crate) mod custom;
mod generated;

pub(crate) use generated::labels;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    generated::register(library);
    adaptive::register(library);
}
