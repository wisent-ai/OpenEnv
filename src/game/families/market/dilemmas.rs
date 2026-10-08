//! Further dilemmas and timing games: traveler's dilemma, dollar auction,
//! penalty shootout, and the normal forms unscrupulous diner, minority game,
//! rock-paper-scissors-lizard-Spock, preemption and war of gifts.

use std::sync::Arc;

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::{amount, amounts, matrix_entry, mean, moves, Entry, Game, Library};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("travelers_dilemma", "market", &["lowest", "highest", "bonus"], travelers));
    library.add(Entry::new("dollar_auction", "market", &["prize", "most"], dollar_auction));
    library.add(Entry::new("penalty_shootout", "market", &["save", "score", "center_bonus"], penalty));
    for entry in [
        matrix_entry(
            "unscrupulous_diner",
            "market",
            &["order_cheap", "order_expensive"],
            "Unscrupulous Diner's Dilemma",
            "Diners at a restaurant independently order cheap or expensive meals and split the bill equally. Each prefers expensive food but shared costs create a free-rider problem. A multiplayer generalization of the Prisoner's Dilemma in social settings.",
        ),
        matrix_entry(
            "minority_game",
            "market",
            &["choose_a", "choose_b", "choose_c"],
            "Minority Game",
            "Players independently choose from three options. With two players, matching choices yield a low tie payoff while different choices yield a high payoff for both. Tests anti-coordination and contrarian strategic reasoning.",
        ),
        matrix_entry(
            "rpsls",
            "market",
            &["rock", "paper", "scissors", "lizard", "spock"],
            "Rock-Paper-Scissors-Lizard-Spock",
            "An extended zero-sum game with five actions. Each action beats two others and loses to two others. The unique Nash equilibrium is uniform randomization. Tests strategic reasoning in larger zero-sum action spaces.",
        ),
        matrix_entry(
            "preemption_game",
            "market",
            &["enter_early", "enter_late", "stay_out"],
            "Preemption Game",
            "A timing game with first-mover advantage. Players choose to enter a market early (risky if both enter) or late (safer but second-mover disadvantage) or stay out entirely for a safe payoff. Early entry against a late opponent captures the market. Tests preemption incentives and entry deterrence.",
        ),
        matrix_entry(
            "war_of_gifts",
            "market",
            &["gift_large", "gift_small", "no_gift"],
            "War of Gifts",
            "A competitive generosity game. Players choose to give a large gift or small gift or no gift. The largest giver wins prestige but at material cost. Mutual large gifts cancel prestige gains. No gift is safe but earns no prestige. Tests competitive signaling through costly generosity.",
        ),
        matrix_entry(
            "parameterized_chicken",
            "market",
            &["hawk", "dove"],
            "Chicken",
            "A Chicken / Hawk-Dove game whose resource value and fight cost the run declares through its cells. Tests anti-coordination behavior under varied incentive parameters.",
        ),
    ] {
        library.add(entry);
    }
}

/// Claims run from `lowest` to `highest`. Equal claims are paid; otherwise
/// both are paid the lower claim, the lower claimant plus `bonus` and the
/// higher minus it.
fn travelers(declared: &Declared<'_>) -> Result<Game> {
    let (lowest, highest) = (declared.whole("lowest")?, declared.whole("highest")?);
    if lowest > highest {
        return Err(Error::Malformed {
            scope: declared.scope().to_owned(),
            name: "highest".to_owned(),
            expected: format!("at least lowest ({lowest})"),
            found: highest.to_string(),
        });
    }
    let bonus = declared.number("bonus")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        if mine == theirs {
            return Ok((mine, theirs));
        }
        if mine < theirs {
            return Ok((mine + bonus, mine - bonus));
        }
        Ok((theirs - bonus, theirs + bonus))
    });
    Ok(Game::new(
        "Traveler's Dilemma",
        "Two travelers submit claims. The lower claim sets the base payout with a bonus for the lower claimant and a penalty for the higher. Nash equilibrium is the minimum claim but experimental subjects often claim high. Tests the rationality paradox in iterative dominance reasoning.",
        "travelers_dilemma",
        (lowest..=highest).map(|claim| format!("claim_{claim}")).collect(),
        payoff,
    ))
}

/// Both bidders pay their bids; the higher wins `prize`, a tie splits it.
fn dollar_auction(declared: &Declared<'_>) -> Result<Game> {
    let prize = declared.number("prize")?;
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let (mine, theirs) = (amount(player)? as f64, amount(opponent)? as f64);
        let none = crate::game::NONE;
        let won = if mine > theirs {
            (prize, none)
        } else if theirs > mine {
            (none, prize)
        } else {
            let shared = mean(&[prize, none]);
            (shared, shared)
        };
        Ok((won.0 - mine, won.1 - theirs))
    });
    Ok(Game::new(
        "Dollar Auction",
        "An escalation game: both players bid and both pay their bids but only the highest bidder wins the prize. Ties split the prize. Models sunk cost escalation and commitment traps. Tests resistance to escalation bias.",
        "dollar_auction",
        amounts("bid", most),
        payoff,
    ))
}

/// The kicker (agent) and keeper pick a side; a match is a save worth `save`
/// to the keeper, a miss a goal worth `score` to the kicker, plus
/// `center_bonus` when the kick went to the center. Zero-sum.
fn penalty(declared: &Declared<'_>) -> Result<Game> {
    let (save, score, bonus) = (
        declared.number("save")?,
        declared.number("score")?,
        declared.number("center_bonus")?,
    );
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        if player == opponent {
            return Ok((-save, save));
        }
        let scored = if player == "center" { score + bonus } else { score };
        Ok((scored, -scored))
    });
    Ok(Game::new(
        "Penalty Shootout",
        "A zero-sum mismatch game modeling penalty kicks. The kicker chooses left or center or right; the goalkeeper dives. Matching means a save. Mismatching means a goal. Center kicks score a bonus when the goalkeeper guesses wrong. Tests mixed-strategy reasoning in adversarial settings.",
        "penalty_shootout",
        moves(&["left", "center", "right"]),
        payoff,
    ))
}
