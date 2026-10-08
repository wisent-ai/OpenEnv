//! Cooperation games over amounts: the beauty contest, the continuous
//! Prisoner's Dilemma and threshold public goods.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::{amount, amounts, mean, Entry, Game, Library};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new(
        "beauty_contest",
        "cooperation",
        &["most", "fraction", "win", "lose", "tie"],
        beauty_contest,
    ));
    library.add(Entry::new("continuous_pd", "cooperation", &["most", "benefit", "cost"], continuous_pd));
    library.add(Entry::new(
        "threshold_public_goods",
        "cooperation",
        &["endowment", "threshold", "bonus"],
        threshold_public_goods,
    ));
}

/// Each seat guesses from nothing to `most`; the guess nearer `fraction` of
/// the two guesses' mean pays `win` and the other `lose`; equally near
/// guesses both pay `tie`.
fn beauty_contest(declared: &Declared<'_>) -> Result<Game> {
    let most = declared.whole("most")?;
    let fraction = declared.number("fraction")?;
    let (win, lose, tie) = (declared.number("win")?, declared.number("lose")?, declared.number("tie")?);
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let target = mean(&[mine, theirs]) * fraction;
        let (near, far) = ((mine - target).abs(), (theirs - target).abs());
        if near < far {
            return Ok((win, lose));
        }
        if far < near {
            return Ok((lose, win));
        }
        Ok((tie, tie))
    });
    Ok(Game::new(
        "Keynesian Beauty Contest",
        "Each player picks a number. The winner is closest to a target fraction of the average. Tests depth of strategic reasoning and level-k thinking. The unique Nash equilibrium is zero, reached through iterated elimination.",
        "beauty_contest",
        amounts("guess", most),
        payoff,
    ))
}

/// Each seat picks a cooperation level from nothing to `most`; a level pays
/// the other seat `benefit` per unit and costs its own seat `cost` per unit.
fn continuous_pd(declared: &Declared<'_>) -> Result<Game> {
    let most = declared.whole("most")?;
    let (benefit, cost) = (declared.number("benefit")?, declared.number("cost")?);
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        Ok((theirs * benefit - mine * cost, mine * benefit - theirs * cost))
    });
    Ok(Game::new(
        "Continuous Prisoner's Dilemma",
        "A generalization of the Prisoner's Dilemma with variable cooperation levels instead of binary choices. Each unit of cooperation costs the player but benefits the opponent more. Tests whether agents find intermediate cooperation strategies in continuous action spaces.",
        "continuous_pd",
        amounts("level", most),
        payoff,
    ))
}

/// Each seat keeps what it does not contribute of `endowment`; when the
/// contributions together reach `threshold` both also get `bonus`.
fn threshold_public_goods(declared: &Declared<'_>) -> Result<Game> {
    let endowment = declared.whole("endowment")?;
    let (threshold, bonus) = (declared.number("threshold")?, declared.number("bonus")?);
    let kept = endowment as f64;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let (mut paid_mine, mut paid_theirs) = (kept - mine, kept - theirs);
        if mine + theirs >= threshold {
            paid_mine += bonus;
            paid_theirs += bonus;
        }
        Ok((paid_mine, paid_theirs))
    });
    Ok(Game::new(
        "Threshold Public Goods Game",
        "A public goods game with a provision threshold. Each player contributes from an endowment. If total contributions meet the threshold a bonus is provided to all. Otherwise contributions are spent without the bonus. Tests coordination on provision.",
        "threshold_public_goods",
        amounts("contribute", endowment),
        payoff,
    ))
}
