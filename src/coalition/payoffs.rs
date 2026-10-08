//! A round's payoffs after its coalitions: a member that did not play the
//! agreed move is a defector, fined `penalty` of its payoff under penalty
//! enforcement; each coalition's first member pays its side payment to every
//! other member.

use crate::game::NONE;
use crate::group::Enforcement;

use super::models::ActiveCoalition;

pub struct Settled {
    pub adjusted: Vec<f64>,
    pub defectors: Vec<usize>,
    pub penalties: Vec<f64>,
    pub side_payments: Vec<f64>,
}

pub fn settle(
    base: &[f64],
    moves: &[String],
    coalitions: &[ActiveCoalition],
    enforcement: Enforcement,
    penalty: f64,
) -> Settled {
    let mut adjusted = base.to_vec();
    let mut penalties = vec![NONE; base.len()];
    let mut side_payments = vec![NONE; base.len()];
    let mut defectors: Vec<usize> = Vec::new();
    for coalition in coalitions {
        for member in &coalition.members {
            let broke = moves.get(*member).is_some_and(|played| *played != coalition.agreed_action);
            if broke && !defectors.contains(member) {
                defectors.push(*member);
            }
        }
    }
    if enforcement == Enforcement::Penalty {
        for defector in &defectors {
            let fine = base[*defector] * penalty;
            penalties[*defector] = fine;
            adjusted[*defector] = base[*defector] - fine;
        }
    }
    for coalition in coalitions {
        let Some(payment) = coalition.side_payment.filter(|payment| *payment > NONE) else {
            continue;
        };
        let mut members = coalition.members.iter().copied().filter(|member| *member < base.len());
        let Some(payer) = members.next() else {
            continue;
        };
        for member in members {
            side_payments[payer] -= payment;
            adjusted[payer] -= payment;
            side_payments[member] += payment;
            adjusted[member] += payment;
        }
    }
    Settled {
        adjusted,
        defectors,
        penalties,
        side_payments,
    }
}
