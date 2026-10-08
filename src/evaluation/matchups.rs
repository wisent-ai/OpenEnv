//! Matchups: named seats (models behind Brama, or library strategies) play
//! every game against each other, every ordered pair including a seat against
//! itself, `evaluation.episodes` times each. When the settings document
//! declares an `arena` section, each seat also gets a reputation: its
//! cooperation rate and fairness, each blended over its episodes by
//! exponential smoothing (`arena.prior` before any episode, `arena.decay` the
//! weight an old value keeps), combined with `arena.weights.cooperation` and
//! `arena.weights.fairness`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;

use crate::env::{Environment, Reset, RoundResult};
use crate::error::{Error, Result};
use crate::game::{Library, NONE};
use crate::settings::Settings;

use super::{cooperated, Runner, SeatSpec};

#[derive(Clone, Debug, Serialize)]
pub struct Matchup {
    pub game: String,
    pub first: String,
    pub second: String,
    pub first_score: f64,
    pub second_score: f64,
    pub first_cooperation: f64,
    pub second_cooperation: f64,
    pub history: Vec<RoundResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Matchups {
    pub seed: u64,
    pub seats: BTreeMap<String, SeatSpec>,
    pub matchups: Vec<Matchup>,
    /// Present only when the settings document declares `arena`.
    pub reputation: Option<BTreeMap<String, f64>>,
}

// Exponential smoothing weighs a new value (1 − decay) beside the old value's
// decay; the two make one whole: https://en.wikipedia.org/wiki/Exponential_smoothing
const WHOLE: f64 = 1.0;

fn fairness(mine: f64, theirs: f64) -> f64 {
    let size = mine.abs() + theirs.abs();
    if size == NONE {
        return WHOLE;
    }
    WHOLE - (mine - theirs).abs() / size
}

pub fn run(settings: Arc<Settings>, seats: BTreeMap<String, SeatSpec>, games: &[String]) -> Result<Matchups> {
    if seats.is_empty() || games.is_empty() {
        return Err(Error::Usage("matchups need at least one seat and one game".to_owned()));
    }
    let episodes = settings.section("evaluation")?.count("episodes")?;
    let library = Arc::new(Library::standard());
    let mut environment = Environment::new(library.clone(), settings.clone())?;
    let seed = environment.seed();
    let needs_brama = seats.values().any(|spec| matches!(spec, SeatSpec::Model(_)));
    let probe = match needs_brama {
        true => seats.values().find(|spec| matches!(spec, SeatSpec::Model(_))).cloned(),
        false => seats.values().next().cloned(),
    };
    let runner = Runner::new(settings.clone(), probe.ok_or_else(|| Error::Usage("matchups need a seat".to_owned()))?, None)?;
    let mut matchups = Vec::new();
    for key in games {
        let game = library.build(key, &settings)?;
        for (first, first_spec) in &seats {
            for (second, second_spec) in &seats {
                for _ in std::iter::repeat(()).take(episodes) {
                    let opponent = runner.seat(second_spec, &game, seed)?;
                    let mut agent = runner.seat(first_spec, &game, seed)?;
                    let request = Reset { game: key.clone(), strategy: None, rounds: None, episode_id: None };
                    let mut observation = environment.reset(&request, Some(opponent))?;
                    while !observation.done {
                        observation = environment.step(&agent.act(&observation)?)?;
                    }
                    let rounds = observation.history.len() as f64;
                    let rate = |pick: fn(&RoundResult) -> &String| {
                        observation.history.iter().filter(|round| cooperated(&game, pick(round))).count() as f64 / rounds
                    };
                    matchups.push(Matchup {
                        game: key.clone(),
                        first: first.clone(),
                        second: second.clone(),
                        first_score: observation.player_score,
                        second_score: observation.opponent_score,
                        first_cooperation: rate(|round| &round.player_action),
                        second_cooperation: rate(|round| &round.opponent_action),
                        history: observation.history,
                    });
                }
            }
        }
    }
    let reputation = match settings.document().get("arena") {
        None => None,
        Some(_) => Some(reputations(&settings, &seats, &matchups)?),
    };
    Ok(Matchups { seed, seats, matchups, reputation })
}

fn reputations(settings: &Settings, seats: &BTreeMap<String, SeatSpec>, matchups: &[Matchup]) -> Result<BTreeMap<String, f64>> {
    let arena = settings.section("arena")?;
    let prior = arena.number("prior")?;
    let decay = arena.number("decay")?;
    let weights = arena.nested("weights")?;
    let (cooperation_weight, fairness_weight) = (weights.number("cooperation")?, weights.number("fairness")?);
    let blend = |old: f64, new: f64| old * decay + new * (WHOLE - decay);
    let mut signals: BTreeMap<&str, (f64, f64)> = seats.keys().map(|name| (name.as_str(), (prior, prior))).collect();
    for matchup in matchups {
        let fair = fairness(matchup.first_score, matchup.second_score);
        for (name, rate) in [(&matchup.first, matchup.first_cooperation), (&matchup.second, matchup.second_cooperation)] {
            if let Some((cooperation, fairness_signal)) = signals.get_mut(name.as_str()) {
                *cooperation = blend(*cooperation, rate);
                *fairness_signal = blend(*fairness_signal, fair);
            }
        }
    }
    Ok(signals
        .into_iter()
        .map(|(name, (cooperation, fair))| (name.to_owned(), cooperation * cooperation_weight + fair * fairness_weight))
        .collect())
}
