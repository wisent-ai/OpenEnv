//! The basic families: the classic dilemmas and amount games, further
//! normal-form games, sequential games, and auctions with the commons.

mod auction;
mod classic;
mod extended;
mod sequential;

pub(crate) use auction::bids;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    classic::register(library);
    extended::register(library);
    sequential::register(library);
    auction::register(library);
}
