//! The payoff mechanics governance can switch on. Each reads its numbers by
//! name from the rules' mechanic configuration (the settings document's
//! `governance` section with any change a proposal made); a mechanic that is
//! on and whose number is not declared is refused by name.

use std::collections::BTreeSet;

use crate::error::{Error, Result};
use crate::game::NONE;
use crate::settings::Declared;

use super::models::{Mechanic, Rules};

/// Every mechanic that is on, in `Mechanic::ALL` order.
pub fn apply(payoffs: &[f64], active: &BTreeSet<usize>, rules: &Rules) -> Result<Vec<f64>> {
    let config = Declared::over("governance", &rules.mechanic_config);
    let mut result = payoffs.to_vec();
    for mechanic in Mechanic::ALL {
        if rules.mechanics.get(mechanic) == Some(&true) {
            result = one(*mechanic, &result, active, &config)?;
        }
    }
    Ok(result)
}

fn mean_of(payoffs: &[f64], active: &BTreeSet<usize>) -> f64 {
    active.iter().map(|seat| payoffs[*seat]).sum::<f64>() / active.len() as f64
}

fn one(mechanic: Mechanic, payoffs: &[f64], active: &BTreeSet<usize>, config: &Declared<'_>) -> Result<Vec<f64>> {
    let mut result = payoffs.to_vec();
    if active.is_empty() {
        return Ok(result);
    }
    match mechanic {
        Mechanic::Taxation => {
            let rate = config.number("tax_rate")?;
            let share = active.iter().map(|seat| payoffs[*seat] * rate).sum::<f64>() / active.len() as f64;
            for seat in active {
                result[*seat] = payoffs[*seat] - payoffs[*seat] * rate + share;
            }
        }
        Mechanic::Redistribution => {
            let mean = mean_of(payoffs, active);
            match config.text("redistribution")? {
                "equal" => active.iter().for_each(|seat| result[*seat] = mean),
                "proportional" => {
                    let damping = config.number("damping")?;
                    active.iter().for_each(|seat| result[*seat] += damping * (mean - result[*seat]));
                }
                other => {
                    return Err(Error::Malformed {
                        scope: config.scope().to_owned(),
                        name: "redistribution".to_owned(),
                        expected: "equal or proportional".to_owned(),
                        found: other.to_owned(),
                    })
                }
            }
        }
        Mechanic::Insurance => {
            let rate = config.number("insurance_contribution")?;
            let threshold = mean_of(payoffs, active) * config.number("insurance_threshold")?;
            let mut pool = NONE;
            for seat in active {
                let paid = result[*seat] * rate;
                pool += paid;
                result[*seat] -= paid;
            }
            let claimants: Vec<usize> = active.iter().copied().filter(|seat| payoffs[*seat] < threshold).collect();
            for seat in &claimants {
                result[*seat] += pool / claimants.len() as f64;
            }
        }
        Mechanic::Quota => {
            let cap = config.number("quota")?;
            let mut excess = NONE;
            let mut below = Vec::new();
            for seat in active {
                if result[*seat] > cap {
                    excess += result[*seat] - cap;
                    result[*seat] = cap;
                } else {
                    below.push(*seat);
                }
            }
            if excess > NONE {
                for seat in &below {
                    result[*seat] += excess / below.len() as f64;
                }
            }
        }
        Mechanic::Subsidy => {
            let floor = config.number("subsidy_floor")?;
            let rate = config.number("subsidy_fund_rate")?;
            let mut pool = NONE;
            for seat in active {
                if result[*seat] > floor {
                    let paid = (result[*seat] - floor) * rate;
                    pool += paid;
                    result[*seat] -= paid;
                }
            }
            let below: Vec<usize> = active.iter().copied().filter(|seat| payoffs[*seat] < floor).collect();
            let needed: f64 = below.iter().map(|seat| floor - payoffs[*seat]).sum();
            if pool > NONE && needed > NONE {
                for seat in below {
                    let need = floor - payoffs[seat];
                    result[seat] += need.min(pool * need / needed);
                }
            }
        }
        Mechanic::Veto => {
            let holder = usize::try_from(config.whole("veto_player")?).map_err(|_| Error::Malformed {
                scope: config.scope().to_owned(),
                name: "veto_player".to_owned(),
                expected: "a seat".to_owned(),
                found: "a number past every seat".to_owned(),
            })?;
            let mean = mean_of(payoffs, active);
            if active.contains(&holder) && payoffs[holder] < mean {
                active.iter().for_each(|seat| result[*seat] = mean);
            }
        }
    }
    Ok(result)
}
