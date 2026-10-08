//! Market, contest and further dilemma games.

mod contests;
mod dilemmas;
mod oligopoly;

use crate::game::Library;

pub(super) fn register(library: &mut Library) {
    oligopoly::register(library);
    contests::register(library);
    dilemmas::register(library);
}
