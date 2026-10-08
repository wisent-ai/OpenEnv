//! Contests, conflict and fair division: Colonel Blotto, war of attrition,
//! Tullock contest, inspection, Rubinstein bargaining and divide-and-choose.

use std::sync::Arc;

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::{amount, amounts, matrix_between, mean, moves, Entry, Game, Library, OpponentMoves, NONE, NOTHING};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("colonel_blotto", "market", &["battlefields", "troops"], blotto));
    library.add(Entry::new("war_of_attrition", "market", &["prize", "cost", "most"], attrition));
    library.add(Entry::new("tullock_contest", "market", &["prize", "most"], tullock));
    library.add(matrix_between(
        "inspection_game",
        "market",
        &["violate", "comply"],
        &["inspect", "no_inspect"],
        "Inspection Game",
        "A potential violator chooses to comply or violate; an inspector chooses whether to inspect. Mixed-strategy equilibrium models compliance, auditing, and arms control verification.",
    ));
    library.add(Entry::new(
        "rubinstein_bargaining",
        "market",
        &["surplus", "discount", "grace"],
        rubinstein,
    ));
    library.add(Entry::new("divide_and_choose", "market", &["endowment"], divide_and_choose));
}

/// Every way to place exactly `troops` across `fields` battlefields.
fn allocations(fields: usize, troops: u64) -> Vec<Vec<u64>> {
    let mut partial: Vec<(Vec<u64>, u64)> = vec![(Vec::new(), troops)];
    for _ in std::iter::repeat(()).take(fields) {
        partial = partial
            .into_iter()
            .flat_map(|(placed, left)| {
                (NOTHING..=left).map(move |here| {
                    let mut next = placed.clone();
                    next.push(here);
                    (next, left - here)
                })
            })
            .collect();
    }
    partial
        .into_iter()
        .filter(|(_, left)| *left == NOTHING)
        .map(|(placed, _)| placed)
        .collect()
}

/// Each seat places its troops as `alloc_<first>_<second>_…`; whoever has more
/// on a battlefield wins it, and each seat is paid the battlefields it won.
fn blotto(declared: &Declared<'_>) -> Result<Game> {
    let fields = declared.count("battlefields")?;
    let troops = declared.whole("troops")?;
    let actions: Vec<String> = allocations(fields, troops)
        .into_iter()
        .map(|placed| {
            let spelled: Vec<String> = placed.iter().map(u64::to_string).collect();
            format!("alloc_{}", spelled.join("_"))
        })
        .collect();
    let payoff = Arc::new(|player: &str, opponent: &str, _: &mut dyn RngCore| {
        let read = |action: &str| -> Result<Vec<u64>> {
            let Some(placed) = action.strip_prefix("alloc_") else {
                return Err(Error::NoAmount { action: action.to_owned() });
            };
            placed
                .split('_')
                .map(|troops| troops.parse().map_err(|_| Error::NoAmount { action: action.to_owned() }))
                .collect()
        };
        let (mine, theirs) = (read(player)?, read(opponent)?);
        let won = mine.iter().zip(&theirs).filter(|(a, b)| a > b).count() as f64;
        let lost = mine.iter().zip(&theirs).filter(|(a, b)| a < b).count() as f64;
        Ok((won, lost))
    });
    Ok(Game::new(
        "Colonel Blotto",
        "Two players allocate limited troops across multiple battlefields. The player with more troops wins each field. Tests multi-dimensional strategic resource allocation.",
        "blotto",
        actions,
        payoff,
    ))
}

/// Each seat names how long it persists, paying `cost` per round of it; the
/// longer one wins `prize`, an equal persistence splits it.
fn attrition(declared: &Declared<'_>) -> Result<Game> {
    let (prize, cost) = (declared.number("prize")?, declared.number("cost")?);
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let won = if mine > theirs {
            (prize, NONE)
        } else if theirs > mine {
            (NONE, prize)
        } else {
            let shared = mean(&[prize, NONE]);
            (shared, shared)
        };
        Ok((won.0 - mine * cost, won.1 - theirs * cost))
    });
    Ok(Game::new(
        "War of Attrition",
        "Both players choose how long to persist. The survivor wins a prize but both pay costs for duration. Tests endurance strategy and rent dissipation reasoning.",
        "war_of_attrition",
        amounts("persist", most),
        payoff,
    ))
}

/// Each seat spends effort; it wins `prize` in proportion to its share of the
/// total effort (equally when nobody spends), less its own effort.
fn tullock(declared: &Declared<'_>) -> Result<Game> {
    let prize = declared.number("prize")?;
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let total = mine + theirs;
        if total == NONE {
            let shared = mean(&[prize, NONE]);
            return Ok((shared, shared));
        }
        Ok((mine / total * prize - mine, theirs / total * prize - theirs))
    });
    Ok(Game::new(
        "Tullock Contest",
        "Players invest effort to win a prize. Win probability is proportional to relative effort. Models lobbying, rent-seeking, and competitive R&D spending.",
        "tullock",
        amounts("effort", most),
        payoff,
    ))
}

/// Demands summing to at most `surplus` are paid in full; demands over it by
/// no more than `grace` are paid times `discount`; larger ones pay nothing.
fn rubinstein(declared: &Declared<'_>) -> Result<Game> {
    let surplus = declared.whole("surplus")?;
    let (discount, grace) = (declared.number("discount")?, declared.number("grace")?);
    let total = surplus as f64;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        if mine + theirs <= total {
            return Ok((mine, theirs));
        }
        if mine + theirs <= total + grace {
            return Ok((mine * discount, theirs * discount));
        }
        Ok((NONE, NONE))
    });
    Ok(Game::new(
        "Rubinstein Bargaining",
        "Players make simultaneous demands over a surplus. Compatible demands yield immediate payoff; excessive demands are discounted. Models alternating-offers bargaining with time preference.",
        "rubinstein",
        amounts("demand", surplus),
        payoff,
    ))
}

/// The divider splits `endowment` into a left piece of its move's amount and a
/// right piece of the rest; the chooser, seeing the split, takes one and the
/// divider keeps the other.
fn divide_and_choose(declared: &Declared<'_>) -> Result<Game> {
    let endowment = declared.whole("endowment")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let left = amount(player)?;
        let right = (endowment - left) as f64;
        let left = left as f64;
        if opponent == "choose_left" {
            return Ok((right, left));
        }
        Ok((left, right))
    });
    let mut game = Game::new(
        "Divide-and-Choose",
        "The divider splits a resource into two portions; the chooser takes their preferred portion. The optimal strategy for the divider is an even split. Tests envy-free fair division reasoning.",
        "divide_choose",
        amounts("split", endowment),
        payoff,
    );
    game.opponent_actions = OpponentMoves::Own(moves(&["choose_left", "choose_right"]));
    game.responds = true;
    Ok(game)
}
