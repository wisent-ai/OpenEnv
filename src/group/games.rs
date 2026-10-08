//! The group social dilemmas: public goods, volunteer's dilemma and the El
//! Farol bar, each for the declared number of `players`.

use std::sync::Arc;

use crate::error::Result;
use crate::game::{amount, amounts, moves};
use crate::settings::Declared;

use super::{count, GroupEntry, GroupGame, GroupLibrary};

pub(super) fn register(library: &mut GroupLibrary) {
    library.add(GroupEntry::new(
        "nplayer_public_goods",
        "group",
        &["endowment", "multiplier"],
        public_goods,
    ));
    library.add(GroupEntry::new(
        "nplayer_volunteer_dilemma",
        "group",
        &["benefit", "cost", "nobody"],
        volunteer,
    ));
    library.add(GroupEntry::new(
        "nplayer_el_farol",
        "group",
        &["capacity", "attend", "crowded", "home"],
        el_farol,
    ));
}

/// Each seat keeps what it does not contribute of `endowment`; the pool of
/// contributions is multiplied by `multiplier` and split among all seats.
fn public_goods(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let endowment = declared.whole("endowment")?;
    let multiplier = declared.number("multiplier")?;
    let kept = endowment as f64;
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let given: Vec<f64> = moves
            .iter()
            .map(|played| amount(played).map(|given| given as f64))
            .collect::<Result<_>>()?;
        let share = given.iter().sum::<f64>() * multiplier / given.len() as f64;
        Ok(given.iter().map(|mine| kept - mine + share).collect())
    });
    Ok(GroupGame::new(
        "N-Player Public Goods",
        "Each player contributes from an endowment. The total pot is multiplied and split equally among all players.",
        "public_goods",
        amounts("contribute", endowment),
        players,
        payoff,
    ))
}

/// When anyone volunteers every seat gets `benefit` and each volunteer pays
/// `cost`; when nobody does every seat gets `nobody`.
fn volunteer(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (benefit, cost, nobody) = (declared.number("benefit")?, declared.number("cost")?, declared.number("nobody")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        if !moves.iter().any(|played| played == "volunteer") {
            return Ok(moves.iter().map(|_| nobody).collect());
        }
        Ok(moves
            .iter()
            .map(|played| if played == "volunteer" { benefit - cost } else { benefit })
            .collect())
    });
    Ok(GroupGame::new(
        "N-Player Volunteer's Dilemma",
        "Players choose to volunteer or abstain. If at least one volunteers, everyone benefits but volunteers pay a cost. If nobody volunteers, everyone gets nothing.",
        "matrix",
        moves(&["volunteer", "abstain"]),
        players,
        payoff,
    ))
}

/// Attending pays `attend` while attendance stays within `capacity` and
/// `crowded` once it passes it; staying home pays `home`.
fn el_farol(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let capacity = declared.whole("capacity")?;
    let (attend, crowded, home) = (declared.number("attend")?, declared.number("crowded")?, declared.number("home")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let over = count(moves, "attend") as u64 > capacity;
        Ok(moves
            .iter()
            .map(|played| match (played == "attend", over) {
                (false, _) => home,
                (true, true) => crowded,
                (true, false) => attend,
            })
            .collect())
    });
    Ok(GroupGame::new(
        "N-Player El Farol Bar",
        "Players decide whether to attend a bar. The bar is fun when not crowded but unpleasant when too many people show up.",
        "matrix",
        moves(&["attend", "stay_home"]),
        players,
        payoff,
    ))
}
