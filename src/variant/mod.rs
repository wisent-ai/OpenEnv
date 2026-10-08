//! Variants: transforms that turn one game into another and compose
//! (`exit` over `cheap_talk` over a base game). A variant that needs a number
//! reads it from `variants.<name>`, or from the game's own declaration when
//! the library registers the composed game under its own key.
//!
//! A run asks for a composed game by key, `<variant>_<game>`
//! (`free_chat_stag_hunt`, `gossip_prisoners_dilemma`,
//! `exit_cheap_talk_hawk_dove`), or by naming variants beside a game.

mod plain;
mod rules;
mod tagged;

use crate::error::{Error, Result};
use crate::game::{Game, OpponentMode};
use crate::settings::Declared;

pub use rules::Rule;

/// Every variant name, in the order a listing shows.
pub const NAMES: &[&str] = &[
    "cheap_talk",
    "exit",
    "binding_commitment",
    "noisy_actions",
    "noisy_payoffs",
    "self_play",
    "cross_model",
    "free_chat",
    "rule_proposal",
    "rule_signal",
    "constitutional",
    "proposer_responder",
    "gossip",
];

/// The names each variant reads from its declaration.
pub fn parameters(name: &str) -> &'static [&'static str] {
    match name {
        "exit" => &["payoff"],
        "binding_commitment" => &["cost"],
        "noisy_actions" => &["tremble"],
        "noisy_payoffs" => &["scale"],
        "rule_proposal" | "rule_signal" | "constitutional" | "proposer_responder" => &["rules"],
        "gossip" => &["ratings"],
        _ => &[],
    }
}

/// `base` with the variant `name` applied. `numbers` yields the declaration
/// the variant reads; it is asked only by a variant that needs a number.
pub fn apply<'s>(base: Game, name: &str, numbers: &dyn Fn() -> Result<Declared<'s>>) -> Result<Game> {
    match name {
        "cheap_talk" => {
            let said = base.actions.clone();
            tagged::compose(base, name, "msg", &said, tagged::Settle::Talk)
        }
        "gossip" => {
            let ratings = numbers()?.texts("ratings")?;
            tagged::compose(base, name, "gossip", &ratings, tagged::Settle::Talk)
        }
        "rule_signal" => {
            let offered = rules::declared(&numbers()?)?;
            let names: Vec<String> = offered.keys().cloned().collect();
            tagged::compose(base, name, "sig", &names, tagged::Settle::Talk)
        }
        "rule_proposal" => {
            let offered = rules::declared(&numbers()?)?;
            let names: Vec<String> = offered.keys().cloned().collect();
            tagged::compose(base, name, "prop", &names, tagged::Settle::Proposal(offered))
        }
        "constitutional" => {
            let offered = rules::declared(&numbers()?)?;
            let names: Vec<String> = offered.keys().cloned().collect();
            tagged::compose(base, name, "const", &names, tagged::Settle::Constitution(offered))
        }
        "proposer_responder" => tagged::proposer_responder(base, rules::declared(&numbers()?)?),
        "exit" => plain::exit(base, &numbers()?),
        "binding_commitment" => plain::binding_commitment(base, &numbers()?),
        "noisy_actions" => plain::noisy_actions(base, &numbers()?),
        "noisy_payoffs" => plain::noisy_payoffs(base, &numbers()?),
        "self_play" => plain::opponent_mode(base, name, OpponentMode::SelfPlay),
        "cross_model" => plain::opponent_mode(base, name, OpponentMode::CrossModel),
        "free_chat" => Ok(plain::free_chat(base)),
        other => Err(Error::UnknownVariant {
            name: other.to_owned(),
            known: NAMES.join(", "),
        }),
    }
}

/// A composed key read as its outermost variant and the key it applies to:
/// `exit_cheap_talk_hawk_dove` is `exit` over `cheap_talk_hawk_dove`. The
/// longest variant name that prefixes the key wins, so `noisy_payoffs_…` is
/// never read as a variant named `noisy`.
pub fn outermost(key: &str) -> Option<(&'static str, &str)> {
    NAMES
        .iter()
        .filter_map(|variant| {
            key.strip_prefix(variant)
                .and_then(|rest| rest.strip_prefix('_'))
                .map(|inner| (*variant, inner))
        })
        .max_by_key(|(variant, _)| variant.len())
}
