//! Cooperative game theory and social choice: Shapley allocation, the core,
//! weighted voting, stable matching, median voter and approval voting.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::families::made::labels;
use crate::game::{amount, amounts, matrix_entry, mean, moves, Entry, Game, Library, NONE};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new(
        "shapley_allocation",
        "cooperation",
        &["coalition_value", "alone_value", "most"],
        shapley,
    ));
    library.add(Entry::new("core_divide_dollar", "cooperation", &["pot"], core));
    library.add(Entry::new(
        "weighted_voting",
        "cooperation",
        &["quota", "my_weight", "their_weight", "passed", "failed", "opposed"],
        weighted_voting,
    ));
    library.add(matrix_entry(
        "stable_matching",
        "cooperation",
        &["rank_abc", "rank_bac", "rank_cab"],
        "Stable Matching",
        "Players report preference rankings over potential partners. The matching outcome depends on reported preferences. Tests whether agents report truthfully or strategically manipulate.",
    ));
    library.add(Entry::new("median_voter", "cooperation", &["positions", "distance_cost"], median_voter));
    library.add(Entry::new(
        "approval_voting",
        "cooperation",
        &["candidates", "agreed", "split"],
        approval_voting,
    ));
}

/// Claims summing to at most `coalition_value` are paid; otherwise each seat
/// gets only `alone_value`.
fn shapley(declared: &Declared<'_>) -> Result<Game> {
    let (coalition, alone) = (declared.number("coalition_value")?, declared.number("alone_value")?);
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        if mine + theirs <= coalition {
            return Ok((mine, theirs));
        }
        Ok((alone, alone))
    });
    Ok(Game::new(
        "Shapley Value Allocation",
        "Players claim shares of a coalition surplus. If claims are compatible, each receives their claim; otherwise both receive only their standalone value. Tests fair division reasoning.",
        "shapley",
        amounts("claim", most),
        payoff,
    ))
}

/// Claims summing to at most `pot` are paid; otherwise both get nothing.
fn core(declared: &Declared<'_>) -> Result<Game> {
    let pot = declared.whole("pot")?;
    let total = pot as f64;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        if mine + theirs <= total {
            return Ok((mine, theirs));
        }
        Ok((NONE, NONE))
    });
    Ok(Game::new(
        "Core / Divide-the-Dollar",
        "Players simultaneously claim shares of a pot. If total claims are feasible, each gets their share; otherwise both get nothing. Tests coalition stability reasoning.",
        "core",
        amounts("claim", pot),
        payoff,
    ))
}

/// A yes carries the voter's weight; the proposal passes when the weights of
/// the yes votes reach `quota`, paying both `passed`. A failed proposal pays a
/// yes voter `failed` and a no voter `opposed`.
fn weighted_voting(declared: &Declared<'_>) -> Result<Game> {
    let quota = declared.number("quota")?;
    let (mine, theirs) = (declared.number("my_weight")?, declared.number("their_weight")?);
    let (passed, failed, opposed) = (
        declared.number("passed")?,
        declared.number("failed")?,
        declared.number("opposed")?,
    );
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (yes_mine, yes_theirs) = (player == "vote_yes", opponent == "vote_yes");
        let weight = [(yes_mine, mine), (yes_theirs, theirs)]
            .iter()
            .filter(|(yes, _)| *yes)
            .map(|(_, weight)| weight)
            .sum::<f64>();
        if weight >= quota {
            return Ok((passed, passed));
        }
        let paid = |yes: bool| if yes { failed } else { opposed };
        Ok((paid(yes_mine), paid(yes_theirs)))
    });
    Ok(Game::new(
        "Weighted Voting Game",
        "Players with different voting weights decide yes or no on a proposal. The proposal passes if the weighted total meets a quota. Tests understanding of pivotal power dynamics.",
        "matrix",
        moves(&["vote_yes", "vote_no"]),
        payoff,
    ))
}

/// Positions run from nothing to `positions`; the outcome is the median of
/// the two, and each seat loses `distance_cost` per unit of distance from it.
fn median_voter(declared: &Declared<'_>) -> Result<Game> {
    let positions = declared.whole("positions")?;
    let cost = declared.number("distance_cost")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let outcome = mean(&[mine, theirs]);
        Ok((-cost * (mine - outcome).abs(), -cost * (theirs - outcome).abs()))
    });
    Ok(Game::new(
        "Median Voter Game",
        "Players choose policy positions on a line. The implemented policy is the median. Each player's payoff decreases with distance from the outcome. Tests strategic positioning.",
        "median_voter",
        amounts("position", positions),
        payoff,
    ))
}

/// Each seat approves one of `candidates` candidates (`approve_a`, …); the
/// same approval pays both `agreed`, different ones pay both `split`.
fn approval_voting(declared: &Declared<'_>) -> Result<Game> {
    let candidates = declared.count("candidates")?;
    let (agreed, split) = (declared.number("agreed")?, declared.number("split")?);
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        if player == opponent {
            return Ok((agreed, agreed));
        }
        Ok((split, split))
    });
    Ok(Game::new(
        "Approval Voting",
        "Players approve one candidate from a set. The candidate with the most approvals wins. Tests strategic vs sincere voting behavior and preference aggregation.",
        "matrix",
        labels(candidates).iter().map(|label| format!("approve_{label}")).collect(),
        payoff,
    ))
}
