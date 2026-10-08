//! Sealed-bid auctions and the two-seat commons. Bids run from nothing to
//! `most` in steps of `increment`; the item is worth `value` to either bidder.
//! A tie splits what is at stake equally between the tied bidders.

use std::sync::Arc;

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::{amount, amounts, Entry, Game, Library, NOTHING};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("first_price_auction", "auction", &["value", "most", "increment"], |declared| {
        auction(declared, Rule::FirstPrice)
    }));
    library.add(Entry::new("vickrey_auction", "auction", &["value", "most", "increment"], |declared| {
        auction(declared, Rule::SecondPrice)
    }));
    library.add(Entry::new("allpay_auction", "auction", &["value", "most", "increment"], |declared| {
        auction(declared, Rule::AllPay)
    }));
    library.add(Entry::new(
        "tragedy_of_commons",
        "auction",
        &["capacity", "most", "depletion_payoff"],
        commons,
    ));
}

#[derive(Clone, Copy)]
enum Rule {
    /// The winner pays its own bid.
    FirstPrice,
    /// The winner pays the losing bid.
    SecondPrice,
    /// Both pay their bids; the winner gets the item.
    AllPay,
}

/// The moves `bid_<amount>` from nothing to `most` in steps of `increment`.
pub(crate) fn bids(declared: &Declared<'_>) -> Result<Vec<String>> {
    let most = declared.whole("most")?;
    let increment = declared.whole("increment")?;
    let step = usize::try_from(increment)
        .ok()
        .and_then(std::num::NonZeroUsize::new)
        .ok_or_else(|| Error::Malformed {
            scope: declared.scope().to_owned(),
            name: "increment".to_owned(),
            expected: "a whole number above zero".to_owned(),
            found: increment.to_string(),
        })?;
    Ok((NOTHING..=most)
        .step_by(step.get())
        .map(|bid| format!("bid_{bid}"))
        .collect())
}

fn auction(declared: &Declared<'_>, rule: Rule) -> Result<Game> {
    let value = declared.number("value")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let mine = amount(player)? as f64;
        let theirs = amount(opponent)? as f64;
        let top = mine.max(theirs);
        let tied = [mine, theirs].iter().filter(|bid| **bid == top).count() as f64;
        let won = |bid: f64, other: f64| -> f64 {
            match rule {
                Rule::FirstPrice | Rule::AllPay => value - bid,
                Rule::SecondPrice => value - other,
            }
        };
        let lost = |bid: f64| -> f64 {
            match rule {
                Rule::AllPay => -bid,
                Rule::FirstPrice | Rule::SecondPrice => NOTHING as f64,
            }
        };
        if mine > theirs {
            return Ok((won(mine, theirs), lost(theirs)));
        }
        if theirs > mine {
            return Ok((lost(mine), won(theirs, mine)));
        }
        Ok(match rule {
            Rule::AllPay => (value / tied - mine, value / tied - theirs),
            Rule::FirstPrice | Rule::SecondPrice => ((value - mine) / tied, (value - theirs) / tied),
        })
    });
    let (name, description) = match rule {
        Rule::FirstPrice => (
            "First-Price Sealed-Bid Auction",
            "Two bidders simultaneously submit sealed bids for an item. The highest bidder wins and pays their own bid. Strategic bidding requires shading below true value to maximize surplus while still winning.",
        ),
        Rule::SecondPrice => (
            "Second-Price (Vickrey) Auction",
            "Two bidders submit sealed bids. The highest bidder wins but pays the second-highest bid. The dominant strategy is to bid one's true valuation, making this a strategy-proof mechanism.",
        ),
        Rule::AllPay => (
            "All-Pay Auction",
            "Two bidders submit sealed bids. Both pay their bids regardless of outcome, but only the highest bidder receives the item. Models contests, lobbying, and rent-seeking where effort is spent whether or not you win.",
        ),
    };
    Ok(Game::new(name, description, "auction", bids(declared)?, payoff))
}

/// Each seat extracts from a shared pool; when the total passes `capacity` the
/// pool collapses and both get `depletion_payoff`, otherwise each keeps what
/// it extracted.
fn commons(declared: &Declared<'_>) -> Result<Game> {
    let capacity = declared.whole("capacity")?;
    let most = declared.whole("most")?;
    let depleted = declared.number("depletion_payoff")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let mine = amount(player)?;
        let theirs = amount(opponent)?;
        if mine + theirs > capacity {
            return Ok((depleted, depleted));
        }
        Ok((mine as f64, theirs as f64))
    });
    Ok(Game::new(
        "Tragedy of the Commons",
        "Players extract resources from a shared pool. Individual incentive is to extract more, but if total extraction exceeds the sustainable capacity, the resource collapses and everyone suffers. Models environmental and resource management dilemmas.",
        "commons",
        amounts("extract", most),
        payoff,
    ))
}
