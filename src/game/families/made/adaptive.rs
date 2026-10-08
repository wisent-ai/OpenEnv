//! Games whose payoffs move with the episode's history. Each build starts a
//! fresh state, so two episodes never share one. The base cells are declared
//! as `payoffs`; how the state moves is declared beside them.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::{moves, Entry, Game, Library, Matrix, Payoff, NONE};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new(
        "adaptive_prisoners_dilemma",
        "adaptive",
        &["payoffs", "start", "lowest", "highest", "step"],
        adaptive_dilemma,
    ));
    library.add(Entry::new(
        "arms_race",
        "adaptive",
        &["payoffs", "escalation", "most_cost", "relief"],
        arms_race,
    ));
    library.add(Entry::new(
        "trust_erosion",
        "adaptive",
        &["payoffs", "multiplier", "decay", "recovery"],
        trust_erosion,
    ));
    library.add(Entry::new(
        "market_dynamics",
        "adaptive",
        &["outputs", "costs", "intercept", "floor", "crowded_above", "shift"],
        market_dynamics,
    ));
    library.add(Entry::new("reputation_payoffs", "adaptive", &["payoffs", "bonus"], reputation_payoffs));
}

fn poisoned(game: &str) -> Error {
    Error::Unsupported {
        game: game.to_owned(),
        reason: "its payoff state was left broken by a panic in an earlier round".to_owned(),
    }
}

/// A payoff over a state the closure owns for this game's lifetime.
fn stateful<S: Send + 'static>(
    game: &'static str,
    state: S,
    pay: impl Fn(&mut S, &str, &str) -> Result<(f64, f64)> + Send + Sync + 'static,
) -> Payoff {
    let state = Mutex::new(state);
    Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let mut state = state.lock().map_err(|_| poisoned(game))?;
        pay(&mut state, player, opponent)
    })
}

fn cooperation_moves() -> Vec<String> {
    moves(&["cooperate", "defect"])
}

/// A dilemma whose cells are scaled by a multiplier that starts at `start`,
/// rises by `step` (up to `highest`) after mutual cooperation and falls by
/// `step` (down to `lowest`) after mutual defection.
fn adaptive_dilemma(declared: &Declared<'_>) -> Result<Game> {
    let actions = cooperation_moves();
    let base = Matrix::declared(declared, &actions, &actions)?;
    let start = declared.number("start")?;
    let (lowest, highest, step) = (declared.number("lowest")?, declared.number("highest")?, declared.number("step")?);
    let name = "Adaptive Prisoner's Dilemma";
    let payoff = stateful(name, start, move |multiplier, player, opponent| {
        let (mine, theirs) = base.cell(name, player, opponent)?;
        let paid = (mine * *multiplier, theirs * *multiplier);
        if player == opponent && player == "cooperate" {
            *multiplier = highest.min(*multiplier + step);
        } else if player == opponent && player == "defect" {
            *multiplier = lowest.max(*multiplier - step);
        }
        Ok(paid)
    });
    Ok(Game::new(name, "A Prisoner's Dilemma where mutual cooperation increases future payoffs via a growing multiplier, while mutual defection decreases it. Mixed outcomes leave it unchanged.", "adaptive", actions, payoff))
}

/// Hawk-Dove where every hawk-hawk round adds `escalation` to a conflict cost
/// (up to `most_cost`) that both hawks pay; any other round relieves it by
/// `relief`, never below nothing.
fn arms_race(declared: &Declared<'_>) -> Result<Game> {
    let actions = moves(&["hawk", "dove"]);
    let base = Matrix::declared(declared, &actions, &actions)?;
    let (escalation, most, relief) = (
        declared.number("escalation")?,
        declared.number("most_cost")?,
        declared.number("relief")?,
    );
    let name = "Arms Race";
    let payoff = stateful(name, NONE, move |cost, player, opponent| {
        let (mine, theirs) = base.cell(name, player, opponent)?;
        if player == opponent && player == "hawk" {
            let paid = (mine - *cost, theirs - *cost);
            *cost = most.min(*cost + escalation);
            return Ok(paid);
        }
        *cost = (*cost - relief).max(NONE);
        Ok((mine, theirs))
    });
    Ok(Game::new(name, "A Hawk-Dove game where mutual hawk play incurs escalating costs each round. Non-hawk rounds de-escalate the accumulated conflict cost.", "adaptive", actions, payoff))
}

/// A dilemma whose cells are scaled by a trust multiplier that starts at
/// `multiplier`, is multiplied by `decay` after mutual defection and recovers
/// by `recovery` (up to its start) after mutual cooperation.
fn trust_erosion(declared: &Declared<'_>) -> Result<Game> {
    let actions = cooperation_moves();
    let base = Matrix::declared(declared, &actions, &actions)?;
    let start = declared.number("multiplier")?;
    let (decay, recovery) = (declared.number("decay")?, declared.number("recovery")?);
    let name = "Trust Erosion";
    let payoff = stateful(name, start, move |multiplier, player, opponent| {
        let (mine, theirs) = base.cell(name, player, opponent)?;
        let paid = (mine * *multiplier, theirs * *multiplier);
        if player == opponent && player == "defect" {
            *multiplier *= decay;
        } else if player == opponent && player == "cooperate" {
            *multiplier = start.min(*multiplier + recovery);
        }
        Ok(paid)
    });
    Ok(Game::new(name, "A Prisoner's Dilemma where a trust multiplier amplifies all payoffs. Mutual defection erodes trust, while mutual cooperation slowly rebuilds it.", "adaptive", actions, payoff))
}

/// A duopoly over output levels `low`, `medium` and `high`, each with a
/// declared output and cost. Price is the current intercept less total output
/// (never below nothing); a round whose total passes `crowded_above` lowers
/// the intercept by `shift` (down to `floor`), any other raises it by `shift`
/// (up to its start).
fn market_dynamics(declared: &Declared<'_>) -> Result<Game> {
    let actions = moves(&["low", "medium", "high"]);
    let outputs = declared.nested("outputs")?;
    let costs = declared.nested("costs")?;
    let mut levels = BTreeMap::new();
    for level in &actions {
        levels.insert(level.clone(), (outputs.number(level)?, costs.number(level)?));
    }
    let start = declared.number("intercept")?;
    let floor = declared.number("floor")?;
    let crowded = declared.number("crowded_above")?;
    let shift = declared.number("shift")?;
    let name = "Market Dynamics";
    let payoff = stateful(name, start, move |intercept, player, opponent| {
        let level = |action: &str| {
            levels.get(action).copied().ok_or_else(|| Error::NoPayoff {
                game: name.to_owned(),
                player: player.to_owned(),
                opponent: opponent.to_owned(),
            })
        };
        let ((mine, my_cost), (theirs, their_cost)) = (level(player)?, level(opponent)?);
        let total = mine + theirs;
        let price = (*intercept - total).max(NONE);
        let paid = (price * mine - my_cost, price * theirs - their_cost);
        *intercept = if total > crowded {
            floor.max(*intercept - shift)
        } else {
            start.min(*intercept + shift)
        };
        Ok(paid)
    });
    Ok(Game::new(name, "A Cournot-like duopoly where each player chooses output level. The demand curve shifts based on past total output: high output depresses future demand, restraint recovers it.", "adaptive", actions, payoff))
}

/// A dilemma where both seats get `bonus` times the agent's cooperation rate
/// over the rounds before this one.
fn reputation_payoffs(declared: &Declared<'_>) -> Result<Game> {
    let actions = cooperation_moves();
    let base = Matrix::declared(declared, &actions, &actions)?;
    let bonus = declared.number("bonus")?;
    let name = "Reputation Payoffs";
    let payoff = stateful(name, Vec::<bool>::new(), move |record, player, opponent| {
        let (mine, theirs) = base.cell(name, player, opponent)?;
        let rate = if record.is_empty() {
            NONE
        } else {
            record.iter().filter(|cooperated| **cooperated).count() as f64 / record.len() as f64
        };
        record.push(player == "cooperate");
        Ok((mine + rate * bonus, theirs + rate * bonus))
    });
    Ok(Game::new(name, "A Prisoner's Dilemma where both players receive a bonus proportional to the player's historical cooperation rate. Building a cooperative reputation pays future dividends.", "adaptive", actions, payoff))
}
