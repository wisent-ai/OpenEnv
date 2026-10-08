//! Sequential and bargaining games: dictator, centipede and Stackelberg.
//! Their mechanics follow the KantBench paper's definitions:
//! https://github.com/wisent-ai/OpenEnv/blob/main/paper/sections/games/library.tex

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::settings::Declared;

use crate::game::{amount, amounts, Entry, Game, Library};

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("dictator", "sequential", &["endowment"], dictator));
    library.add(Entry::new(
        "centipede",
        "sequential",
        &["initial_pot", "growth", "stages", "large_share", "small_share"],
        centipede,
    ));
    library.add(Entry::new(
        "stackelberg",
        "sequential",
        &["intercept", "slope", "cost", "most"],
        stackelberg,
    ));
}

/// The dictator keeps what it does not give; the recipient has no choice.
fn dictator(declared: &Declared<'_>) -> Result<Game> {
    let endowment = declared.whole("endowment")?;
    let payoff = Arc::new(move |player: &str, _: &str, _: &mut dyn RngCore| {
        let given = amount(player)?;
        Ok(((endowment - given) as f64, given as f64))
    });
    Ok(Game::new(
        "Dictator Game",
        "One player (the dictator) decides how to split an endowment with a passive recipient who has no say. Tests fairness preferences and altruistic behavior when there is no strategic incentive to share.",
        "dictator",
        amounts("give", endowment),
        payoff,
    ))
}

/// Each seat names the stage it takes at (`take_<stage>`) or passes every
/// stage (`pass_all`). The pot starts at `initial_pot` and grows by `growth`
/// at every stage passed; whoever takes first gets `large_share` of it and the
/// other `small_share`, both rounded down. When both take at the same stage
/// the agent's seat takes.
fn centipede(declared: &Declared<'_>) -> Result<Game> {
    let initial = declared.whole("initial_pot")?;
    let growth = declared.whole("growth")?;
    let stages = declared.whole("stages")?;
    let large_share = declared.number("large_share")?;
    let small_share = declared.number("small_share")?;
    let mut actions = amounts("take", stages);
    // Passing every stage takes after the last one: one stage past the
    // number of take moves' last index, which is how many take moves exist.
    let passing_all = actions.len() as u64;
    actions.push("pass_all".to_owned());
    let stage = move |action: &str| -> Result<u64> {
        if action == "pass_all" {
            return Ok(passing_all);
        }
        amount(action)
    };
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let mine = stage(player)?;
        let theirs = stage(opponent)?;
        let taken_at = mine.min(theirs);
        let pot = (initial as f64) * (growth as f64).powi(taken_at as i32);
        let large = (pot * large_share).floor();
        let small = (pot * small_share).floor();
        if mine <= theirs {
            return Ok((large, small));
        }
        Ok((small, large))
    });
    Ok(Game::new(
        "Centipede Game",
        "Players alternate deciding to take or pass. Each pass doubles the pot. The taker gets the larger share while the other gets the smaller share. Backward induction predicts immediate taking, but cooperation through passing yields higher joint payoffs.",
        "centipede",
        actions,
        payoff,
    ))
}

/// A quantity duopoly: price `intercept − slope·(q_leader + q_follower)`, each
/// firm's profit `(price − cost)·q`. Quantities run from nothing to `most`.
fn stackelberg(declared: &Declared<'_>) -> Result<Game> {
    let intercept = declared.number("intercept")?;
    let slope = declared.number("slope")?;
    let cost = declared.number("cost")?;
    let most = declared.whole("most")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        let leader = amount(player)? as f64;
        let follower = amount(opponent)? as f64;
        let price = intercept - slope * (leader + follower);
        Ok(((price - cost) * leader, (price - cost) * follower))
    });
    Ok(Game::new(
        "Stackelberg Competition",
        "A quantity-setting duopoly where the leader commits to a production quantity first, and the follower observes and responds. The leader can exploit first-mover advantage. Price is determined by total market quantity.",
        "stackelberg",
        amounts("produce", most),
        payoff,
    ))
}
