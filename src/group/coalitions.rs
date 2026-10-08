//! Coalition games: group games played with a negotiation phase before each
//! move (`crate::coalition`). Each declares its payoff numbers and the share
//! of a defector's payoff a `penalty` takes; how agreements bind is the
//! game's design.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::error::Result;
use crate::game::{amount, moves};
use crate::settings::Declared;

use super::{count, Enforcement, GroupEntry, GroupGame, GroupLibrary};

pub(super) fn register(library: &mut GroupLibrary) {
    library.add(GroupEntry::new(
        "coalition_cartel",
        "coalition",
        &["penalty", "holds_at", "colluding_held", "colluding_broken", "competing_held", "competing_broken"],
        cartel,
    ));
    library.add(GroupEntry::new(
        "coalition_alliance",
        "coalition",
        &["penalty", "pool", "betrayal", "unsupported"],
        alliance,
    ));
    library.add(GroupEntry::new("coalition_voting", "coalition", &["penalty", "winner", "loser"], voting));
    library.add(GroupEntry::new(
        "coalition_ostracism",
        "coalition",
        &["penalty", "bonus_pool", "excluded", "kept"],
        ostracism,
    ));
    library.add(GroupEntry::new(
        "coalition_resource_trading",
        "coalition",
        &["penalty", "diverse", "uniform", "minority_bonus"],
        trading,
    ));
    library.add(GroupEntry::new(
        "coalition_rule_voting",
        "coalition",
        &["penalty", "equal", "winner_high", "winner_low"],
        rule_voting,
    ));
    library.add(GroupEntry::new(
        "coalition_commons",
        "coalition",
        &["penalty", "sustainable_most", "low_kept", "high_kept", "low_depleted", "high_depleted"],
        commons,
    ));
}

fn coalition(
    declared: &Declared<'_>,
    game: (&str, &str),
    actions: Vec<String>,
    players: usize,
    enforcement: Enforcement,
    payoff: super::GroupPayoff,
) -> Result<GroupGame> {
    let (name, description) = game;
    let mut built = GroupGame::new(name, description, "coalition", actions, players, payoff);
    built.enforcement = enforcement;
    built.penalty = declared.number("penalty")?;
    Ok(built)
}

/// When at least `holds_at` seats collude the cartel holds; colluders and
/// competitors are paid by whether it held.
fn cartel(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let holds_at = declared.whole("holds_at")?;
    let (colluding_held, colluding_broken) = (declared.number("colluding_held")?, declared.number("colluding_broken")?);
    let (competing_held, competing_broken) = (declared.number("competing_held")?, declared.number("competing_broken")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let held = count(moves, "collude") as u64 >= holds_at;
        Ok(moves
            .iter()
            .map(|played| match (played == "collude", held) {
                (true, true) => colluding_held,
                (true, false) => colluding_broken,
                (false, true) => competing_held,
                (false, false) => competing_broken,
            })
            .collect())
    });
    coalition(declared, ("Cartel", "Players collude or compete. If enough collude the cartel holds. Defectors who promised to collude are fined under penalty enforcement."), moves(&["collude", "compete"]), players, Enforcement::Penalty, payoff)
}

/// Supporters split `pool`; a betrayer takes `betrayal`; with no supporter
/// every seat gets `unsupported`.
fn alliance(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (pool, betrayal, unsupported) = (declared.number("pool")?, declared.number("betrayal")?, declared.number("unsupported")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let supporters = count(moves, "support");
        if !moves.iter().any(|played| played == "support") {
            return Ok(moves.iter().map(|_| unsupported).collect());
        }
        Ok(moves
            .iter()
            .map(|played| if played == "support" { pool / supporters as f64 } else { betrayal })
            .collect())
    });
    coalition(declared, ("Alliance Formation", "Form non-binding alliances. Supporters split a shared pool; betrayers take a fixed gain. Cheap-talk: no enforcement."), moves(&["support", "betray"]), players, Enforcement::CheapTalk, payoff)
}

/// The side with more votes wins (A on a tie); its voters get `winner`, the
/// others `loser`.
fn voting(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (winner, loser) = (declared.number("winner")?, declared.number("loser")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let for_a = count(moves, "vote_A");
        let won = if for_a >= moves.len() - for_a { "vote_A" } else { "vote_B" };
        Ok(moves.iter().map(|played| if played == won { winner } else { loser }).collect())
    });
    coalition(declared, ("Coalition Voting", "Form voting blocs bound to vote together. Majority earns a winner payoff. Binding enforcement overrides defectors to their agreed vote."), moves(&["vote_A", "vote_B"]), players, Enforcement::Binding, payoff)
}

/// Each seat votes to exclude a seat (`exclude_<seat>`) or nobody. A seat a
/// strict majority votes against gets `excluded` and the others split
/// `bonus_pool`; otherwise every seat gets `kept`.
fn ostracism(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (pool, excluded_paid, kept) = (declared.number("bonus_pool")?, declared.number("excluded")?, declared.number("kept")?);
    let mut actions: Vec<String> = (crate::game::NOTHING..players as u64).map(|seat| format!("exclude_{seat}")).collect();
    actions.push("exclude_none".to_owned());
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let targets: BTreeSet<&String> = moves.iter().collect();
        let seats = moves.len();
        let ousted = targets
            .into_iter()
            .filter(|target| *target != "exclude_none")
            .find(|target| {
                let against = count(moves, target);
                against > seats - against
            })
            .map(|target| amount(target))
            .transpose()?;
        let Some(ousted) = ousted else {
            return Ok(moves.iter().map(|_| kept).collect());
        };
        let everyone = crate::game::NOTHING..seats as u64;
        let remaining = everyone.clone().filter(|seat| *seat != ousted).count() as f64;
        Ok(everyone
            .map(|seat| if seat == ousted { excluded_paid } else { pool / remaining })
            .collect())
    });
    coalition(declared, ("Ostracism", "Vote to exclude a player. Excluded gets zero; others split a bonus. Penalty enforcement fines defectors who break exclusion agreements."), actions, players, Enforcement::Penalty, payoff)
}

/// Each seat produces A or B. When both are produced every seat gets
/// `diverse`, and producers of the scarcer resource add `minority_bonus`;
/// when one resource is all that is produced every seat gets `uniform`.
fn trading(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (diverse, uniform, bonus) = (declared.number("diverse")?, declared.number("uniform")?, declared.number("minority_bonus")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let (a, b) = (count(moves, "produce_A"), count(moves, "produce_B"));
        let both = moves.iter().any(|played| played == "produce_A") && moves.iter().any(|played| played == "produce_B");
        if !both {
            return Ok(moves.iter().map(|_| uniform).collect());
        }
        Ok(moves
            .iter()
            .map(|played| {
                let scarce = (played == "produce_A" && a < b) || (played == "produce_B" && b < a);
                if scarce { diverse + bonus } else { diverse }
            })
            .collect())
    });
    let mut game = coalition(declared, ("Resource Trading", "Produce resource A or B. Diversity is rewarded; minority producers get a bonus. Cheap-talk lets players agree on production but renegotiate freely."), moves(&["produce_A", "produce_B"]), players, Enforcement::CheapTalk, payoff)?;
    game.side_payments = true;
    Ok(game)
}

/// The rule with the most votes wins (ties go to the equal split). Under the
/// equal split every seat gets `equal`; under winner-take-all its voters get
/// `winner_high` and the others `winner_low`.
fn rule_voting(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let (equal, high, low) = (declared.number("equal")?, declared.number("winner_high")?, declared.number("winner_low")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        if count(moves, "rule_winner") <= count(moves, "rule_equal") {
            return Ok(moves.iter().map(|_| equal).collect());
        }
        Ok(moves.iter().map(|played| if played == "rule_winner" { high } else { low }).collect())
    });
    coalition(declared, ("Rule Voting", "Vote on payoff rule: equal split or winner-take-all. Binding enforcement locks coalition members to their agreed vote."), moves(&["rule_equal", "rule_winner"]), players, Enforcement::Binding, payoff)
}

/// While at most `sustainable_most` seats extract high the resource holds;
/// each extraction level is paid by whether it held.
fn commons(declared: &Declared<'_>, players: usize) -> Result<GroupGame> {
    let most = declared.whole("sustainable_most")?;
    let (low_kept, high_kept) = (declared.number("low_kept")?, declared.number("high_kept")?);
    let (low_depleted, high_depleted) = (declared.number("low_depleted")?, declared.number("high_depleted")?);
    let payoff = Arc::new(move |moves: &[String]| -> Result<Vec<f64>> {
        let held = count(moves, "extract_high") as u64 <= most;
        Ok(moves
            .iter()
            .map(|played| match (played == "extract_high", held) {
                (true, true) => high_kept,
                (true, false) => high_depleted,
                (false, true) => low_kept,
                (false, false) => low_depleted,
            })
            .collect())
    });
    coalition(declared, ("Commons Governance", "Extract from a shared resource. Over-extraction degrades payoffs. Penalty enforcement fines coalition members who exceed agreed limits."), moves(&["extract_low", "extract_high"]), players, Enforcement::Penalty, payoff)
}
