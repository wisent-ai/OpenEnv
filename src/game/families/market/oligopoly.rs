//! Market competition and bargaining: Cournot, Bertrand, Hotelling, entry
//! deterrence, the Nash demand game and the double auction.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::{amount, amounts, matrix_between, mean, Entry, Game, Library, NONE};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("cournot", "market", &["intercept", "slope", "cost", "most"], cournot));
    library.add(Entry::new("bertrand", "market", &["most", "cost", "market_size"], bertrand));
    library.add(Entry::new("hotelling", "market", &["line", "transport", "market_value"], hotelling));
    library.add(matrix_between(
        "entry_deterrence",
        "market",
        &["enter", "stay_out"],
        &["accommodate", "fight"],
        "Entry Deterrence",
        "A potential entrant decides whether to enter a market; the incumbent decides whether to fight or accommodate. Tests credible commitment and limit pricing reasoning.",
    ));
    library.add(Entry::new("nash_demand", "market", &["surplus"], nash_demand));
    library.add(Entry::new(
        "double_auction",
        "market",
        &["buyer_value", "seller_cost", "most"],
        double_auction,
    ));
}

type Pay = dyn Fn(f64, f64) -> (f64, f64) + Send + Sync;

/// A game over amount moves `<prefix>_0 … <prefix>_<most>` paid by `pay`.
fn priced(name: &str, description: &str, kind: &str, moves: Vec<String>, pay: Arc<Pay>) -> Game {
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        Ok(pay(amount(player)? as f64, amount(opponent)? as f64))
    });
    Game::new(name, description, kind, moves, payoff)
}

/// Price `intercept − slope · (q₁ + q₂)`; each firm earns `(price − cost) · q`.
fn cournot(declared: &Declared<'_>) -> Result<Game> {
    let (intercept, slope, cost) = (declared.number("intercept")?, declared.number("slope")?, declared.number("cost")?);
    let most = declared.whole("most")?;
    Ok(priced(
        "Cournot Duopoly",
        "Two firms simultaneously choose production quantities. Market price decreases with total output. Tests Nash equilibrium reasoning in quantity competition.",
        "cournot",
        amounts("produce", most),
        Arc::new(move |mine, theirs| {
            let price = intercept - slope * (mine + theirs);
            ((price - cost) * mine, (price - cost) * theirs)
        }),
    ))
}

/// The lower price takes the market of `market_size − price` buyers (never
/// fewer than none) at `price − cost` each; equal prices share it equally.
fn bertrand(declared: &Declared<'_>) -> Result<Game> {
    let (cost, size) = (declared.number("cost")?, declared.number("market_size")?);
    let most = declared.whole("most")?;
    Ok(priced(
        "Bertrand Competition",
        "Two firms simultaneously set prices. The lower-price firm captures the market. The Bertrand paradox predicts pricing at marginal cost even with only two competitors.",
        "bertrand",
        amounts("price", most),
        Arc::new(move |mine, theirs| {
            let profit = |price: f64| (price - cost) * (size - price).max(NONE);
            if mine < theirs {
                return (profit(mine), NONE);
            }
            if theirs < mine {
                return (NONE, profit(theirs));
            }
            let shared = mean(&[profit(mine), NONE]);
            (shared, shared)
        }),
    ))
}

/// Two firms locate on a line of length `line`; each serves the customers
/// nearer to it, earning `transport` per unit of line it serves. Firms at the
/// same spot split `market_value` equally.
fn hotelling(declared: &Declared<'_>) -> Result<Game> {
    let line = declared.whole("line")?;
    let (transport, value) = (declared.number("transport")?, declared.number("market_value")?);
    let length = line as f64;
    Ok(priced(
        "Hotelling Location Game",
        "Two firms choose locations on a line. Consumers visit the nearest firm. Tests the principle of minimum differentiation and spatial competition dynamics.",
        "hotelling",
        amounts("locate", line),
        Arc::new(move |mine, theirs| {
            if mine == theirs {
                let shared = mean(&[value, NONE]);
                return (shared, shared);
            }
            let middle = mean(&[mine, theirs]);
            let served = if mine < theirs { middle } else { length - middle };
            (served * transport, (length - served) * transport)
        }),
    ))
}

/// Compatible demands (summing to at most `surplus`) are each paid; otherwise
/// both get nothing.
fn nash_demand(declared: &Declared<'_>) -> Result<Game> {
    let surplus = declared.whole("surplus")?;
    let total = surplus as f64;
    Ok(priced(
        "Nash Demand Game",
        "Two players simultaneously demand shares of a surplus. If demands are compatible (sum within surplus), both receive their demand; otherwise both get nothing.",
        "nash_demand",
        amounts("demand", surplus),
        Arc::new(move |mine, theirs| {
            if mine + theirs <= total {
                return (mine, theirs);
            }
            (NONE, NONE)
        }),
    ))
}

/// The buyer bids, the seller asks; a bid at or above the ask trades at their
/// midpoint, paying the buyer `buyer_value − price` and the seller
/// `price − seller_cost`.
fn double_auction(declared: &Declared<'_>) -> Result<Game> {
    let (value, cost) = (declared.number("buyer_value")?, declared.number("seller_cost")?);
    let most = declared.whole("most")?;
    Ok(priced(
        "Double Auction",
        "A buyer submits a bid and a seller submits an ask. Trade occurs at the midpoint if bid exceeds ask. Tests price discovery and competitive market behavior.",
        "double_auction",
        amounts("bid", most),
        Arc::new(move |bid, ask| {
            if bid >= ask {
                let price = mean(&[bid, ask]);
                return (value - price, price - cost);
            }
            (NONE, NONE)
        }),
    ))
}
