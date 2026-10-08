//! Opponent strategies: what the opponent's seat plays when no model holds it.
//! The repeated-game strategies read only the history; the ones that need a
//! number (how often to forgive, how much to offer) read it from the run's
//! settings document under `strategies.<name>` and are refused when it is not
//! declared.

mod amounts;
mod repeated;

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::Game;
use crate::settings::Settings;

/// One past round as the strategy sees it: what the agent played and what the
/// strategy itself played.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub agent: String,
    pub own: String,
}

/// What a strategy sees when it moves.
pub struct View<'a> {
    pub game: &'a Game,
    /// The moves its seat may play.
    pub moves: &'a [String],
    pub history: &'a [Turn],
    /// The agent's move this round, when the game lets the opponent answer it
    /// (a responder sees the offer, a trustee sees the investment).
    pub answering: Option<&'a str>,
}

pub trait Strategy: Send {
    fn choose(&mut self, view: &View<'_>, rng: &mut dyn RngCore) -> Result<String>;
}

/// Every strategy name the library answers to, in the order a listing shows.
pub const NAMES: &[&str] = &[
    "random",
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "tit_for_two_tats",
    "grudger",
    "pavlov",
    "suspicious_tit_for_tat",
    "generous_tit_for_tat",
    "adaptive",
    "mixed",
    "ultimatum_fair",
    "ultimatum_low",
    "trust_fair",
    "trust_generous",
    "public_goods_fair",
    "public_goods_free_rider",
];

/// The names each strategy reads from `strategies.<name>`.
pub fn parameters(name: &str) -> &'static [&'static str] {
    match name {
        "generous_tit_for_tat" => &["forgive"],
        "mixed" => &["cooperate"],
        "ultimatum_fair" => &["offer", "accept_at_least"],
        "ultimatum_low" => &["offer"],
        "trust_fair" | "trust_generous" => &["invest", "return_share"],
        "public_goods_fair" | "public_goods_free_rider" => &["contribute"],
        _ => &[],
    }
}

/// The strategy `name`, with any numbers it needs read from `settings`.
pub fn named(name: &str, settings: &Settings) -> Result<Box<dyn Strategy>> {
    let declared = || settings.entry("strategies", name);
    Ok(match name {
        "random" => Box::new(repeated::Random),
        "always_cooperate" => Box::new(repeated::AlwaysCooperate),
        "always_defect" => Box::new(repeated::AlwaysDefect),
        "tit_for_tat" => Box::new(repeated::TitForTat {
            opening: repeated::Opening::Cooperate,
        }),
        "suspicious_tit_for_tat" => Box::new(repeated::TitForTat {
            opening: repeated::Opening::Defect,
        }),
        "tit_for_two_tats" => Box::new(repeated::TitForTwoTats),
        "grudger" => Box::new(repeated::Grudger),
        "pavlov" => Box::new(repeated::Pavlov),
        "generous_tit_for_tat" => Box::new(repeated::GenerousTitForTat {
            forgive: declared()?.probability("forgive")?,
        }),
        "adaptive" => Box::new(repeated::Adaptive),
        "mixed" => Box::new(repeated::Mixed {
            cooperate: declared()?.probability("cooperate")?,
        }),
        "ultimatum_fair" => {
            let declared = declared()?;
            Box::new(amounts::Ultimatum {
                name: "ultimatum_fair",
                offer: declared.whole("offer")?,
                accept_at_least: Some(declared.whole("accept_at_least")?),
            })
        }
        "ultimatum_low" => Box::new(amounts::Ultimatum {
            name: "ultimatum_low",
            offer: declared()?.whole("offer")?,
            accept_at_least: None,
        }),
        "trust_fair" | "trust_generous" => {
            let declared = declared()?;
            Box::new(amounts::Trust {
                name: if name == "trust_fair" { "trust_fair" } else { "trust_generous" },
                invest: declared.whole("invest")?,
                return_share: declared.number("return_share")?,
            })
        }
        "public_goods_fair" | "public_goods_free_rider" => Box::new(amounts::Contribute {
            name: if name == "public_goods_fair" { "public_goods_fair" } else { "public_goods_free_rider" },
            amount: declared()?.whole("contribute")?,
        }),
        _ => {
            return Err(Error::UnknownStrategy {
                name: name.to_owned(),
                known: NAMES.join(", "),
            })
        }
    })
}

/// The cooperative move: a repeated game lists it first.
pub(crate) fn cooperative<'a>(strategy: &str, view: &View<'a>) -> Result<String> {
    view.moves.first().cloned().ok_or_else(|| Error::NoMoves {
        strategy: strategy.to_owned(),
        game: view.game.key.clone(),
    })
}

/// The defecting move: a repeated game lists it second. A game with a single
/// move has nothing to defect with, and is refused.
pub(crate) fn defecting<'a>(strategy: &str, view: &View<'a>) -> Result<String> {
    let mut listed = view.moves.iter();
    listed.next();
    listed.next().cloned().ok_or_else(|| Error::Unsupported {
        game: view.game.key.clone(),
        reason: format!("{strategy} needs a second, defecting move and the game lists one move"),
    })
}

/// The move `wanted` when the seat may play it; a strategy whose declared
/// amount the game does not offer is refused by name rather than played as
/// some other move.
pub(crate) fn listed(strategy: &str, view: &View<'_>, wanted: String) -> Result<String> {
    if view.moves.contains(&wanted) {
        return Ok(wanted);
    }
    Err(Error::Unsupported {
        game: view.game.key.clone(),
        reason: format!(
            "{strategy} plays {wanted}, which this seat cannot play; its moves are {}",
            view.moves.join(", ")
        ),
    })
}
