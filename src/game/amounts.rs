//! Amount moves (`offer_5`, `contribute_10`) and the arithmetic games share.

use crate::error::{Error, Result};

/// The mean of some values: an equal split, a midpoint. Nothing has no mean.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

// The smallest amount a contribution, offer or investment can be: the
// KantBench paper's games take an amount from nothing up to the endowment,
// `x ∈ [0, E]`:
// https://github.com/wisent-ai/OpenEnv/blob/main/paper/sections/games/library.tex
pub const NOTHING: u64 = 0;

/// The same nothing as a payoff: what a rejected offer pays, what a cost is
/// before it accrues.
pub const NONE: f64 = NOTHING as f64;

/// The moves `<prefix>_<amount>` for every amount from nothing to `most`.
pub fn amounts(prefix: &str, most: u64) -> Vec<String> {
    (NOTHING..=most).map(|amount| format!("{prefix}_{amount}")).collect()
}

/// The amount a move like `offer_5` carries after its last underscore.
pub fn amount(action: &str) -> Result<u64> {
    action
        .rsplit_once('_')
        .and_then(|(_, amount)| amount.parse().ok())
        .ok_or_else(|| Error::NoAmount {
            action: action.to_owned(),
        })
}
