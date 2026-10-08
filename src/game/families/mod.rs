//! Every family of games the library holds. Each family registers its games
//! with the names they read from a settings document.

pub(crate) mod basic;
mod information;
pub(crate) mod made;
mod market;
mod cooperation;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    basic::register(library);
    information::register(library);
    market::register(library);
    cooperation::register(library);
    made::register(library);
}
