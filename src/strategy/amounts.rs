//! Strategies for the amount games: what to offer, invest, return or
//! contribute, each amount declared by the run. A responder answers the move
//! it is shown this round, never an earlier round's.

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::amount;
use crate::settings::Declared;

use super::{listed, Strategy, View};

fn answering<'a>(strategy: &str, view: &View<'a>) -> Result<&'a str> {
    view.answering.ok_or_else(|| Error::Unsupported {
        game: view.game.key.clone(),
        reason: format!("{strategy} answers the agent's move, and this game does not show it to the opponent"),
    })
}

/// Proposes the declared offer; as the responder, accepts an offer of at least
/// the declared amount, or every offer when no amount is declared.
pub struct Ultimatum {
    pub name: &'static str,
    pub offer: u64,
    pub accept_at_least: Option<u64>,
}

impl Strategy for Ultimatum {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let proposal = format!("offer_{}", self.offer);
        if view.moves.contains(&proposal) {
            return Ok(proposal);
        }
        let accepts = match self.accept_at_least {
            None => true,
            Some(least) => amount(answering(self.name, view)?)? >= least,
        };
        let answer = if accepts { "accept" } else { "reject" };
        listed(self.name, view, answer.to_owned())
    }
}

/// Invests the declared amount; as the trustee, returns the declared share of
/// what it received (the investment times the game's multiplier), rounded
/// down to a whole amount.
pub struct Trust {
    pub name: &'static str,
    pub invest: u64,
    pub return_share: f64,
}

impl Strategy for Trust {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        let investment = format!("invest_{}", self.invest);
        if view.moves.contains(&investment) {
            return Ok(investment);
        }
        let invested = amount(answering(self.name, view)?)?;
        let scope = format!("games.{}", view.game.key);
        let multiplier = Declared::over(&scope, &view.game.parameters).whole("multiplier")?;
        let received = invested * multiplier;
        let returned = (received as f64 * self.return_share).floor() as u64;
        listed(self.name, view, format!("return_{returned}"))
    }
}

/// Contributes the declared amount every round.
pub struct Contribute {
    pub name: &'static str,
    pub amount: u64,
}

impl Strategy for Contribute {
    fn choose(&mut self, view: &View<'_>, _: &mut dyn RngCore) -> Result<String> {
        listed(self.name, view, format!("contribute_{}", self.amount))
    }
}
