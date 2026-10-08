//! The rules a meta-game's players can propose: each turns a round's payoffs
//! into new ones. A rule that needs a number reads it from its declaration
//! (`rules.<rule>`), and the declared rules are the ones the players may
//! name. The cooperative move is the base game's first, as strategies read
//! it.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::settings::Declared;

/// One rule and the numbers it was declared with.
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// Payoffs unchanged.
    Unchanged,
    /// Both seats get the mean of the two payoffs.
    EqualSplit,
    /// A cooperative move earns `bonus` on top.
    CooperationBonus(f64),
    /// A move that is not cooperative loses `penalty`.
    DefectionPenalty(f64),
    /// No payoff falls below `floor`.
    MinimumGuarantee(f64),
    /// A move that is not cooperative loses `penalty`, set high enough to ban it.
    DefectionBan(f64),
}

impl Rule {
    fn read(name: &str, declared: &Declared<'_>) -> Result<Self> {
        Ok(match name {
            "none" => Rule::Unchanged,
            "equalsplit" => Rule::EqualSplit,
            "coopbonus" => Rule::CooperationBonus(declared.number("bonus")?),
            "defectpenalty" => Rule::DefectionPenalty(declared.number("penalty")?),
            "minguarantee" => Rule::MinimumGuarantee(declared.number("floor")?),
            "bandefect" => Rule::DefectionBan(declared.number("penalty")?),
            other => {
                return Err(Error::Malformed {
                    scope: declared.scope().to_owned(),
                    name: other.to_owned(),
                    expected: "one of none, equalsplit, coopbonus, defectpenalty, minguarantee, bandefect".to_owned(),
                    found: other.to_owned(),
                })
            }
        })
    }

    /// The round's payoffs under this rule.
    pub fn apply(self, paid: (f64, f64), moves: (&str, &str), cooperative: &str) -> (f64, f64) {
        let (mine, theirs) = paid;
        let (player, opponent) = moves;
        let charged = |payoff: f64, played: &str, amount: f64| {
            if played == cooperative {
                payoff
            } else {
                payoff - amount
            }
        };
        match self {
            Rule::Unchanged => paid,
            Rule::EqualSplit => {
                let pair = [mine, theirs];
                let mean = pair.iter().sum::<f64>() / pair.len() as f64;
                (mean, mean)
            }
            Rule::CooperationBonus(bonus) => {
                let earned = |payoff: f64, played: &str| {
                    if played == cooperative {
                        payoff + bonus
                    } else {
                        payoff
                    }
                };
                (earned(mine, player), earned(theirs, opponent))
            }
            Rule::DefectionPenalty(penalty) | Rule::DefectionBan(penalty) => {
                (charged(mine, player, penalty), charged(theirs, opponent, penalty))
            }
            Rule::MinimumGuarantee(floor) => (mine.max(floor), theirs.max(floor)),
        }
    }
}

/// The rules a variant's declaration offers: `rules.<name>` for each one,
/// with the numbers that rule reads.
pub fn declared(declared: &Declared<'_>) -> Result<BTreeMap<String, Rule>> {
    let rules = declared.nested("rules")?;
    let mut offered = BTreeMap::new();
    for name in rules.values().keys() {
        let numbers = rules.nested(name)?;
        offered.insert(name.clone(), Rule::read(name, &numbers)?);
    }
    if offered.is_empty() {
        return Err(Error::Malformed {
            scope: declared.scope().to_owned(),
            name: "rules".to_owned(),
            expected: "at least one rule".to_owned(),
            found: "none".to_owned(),
        });
    }
    Ok(offered)
}
