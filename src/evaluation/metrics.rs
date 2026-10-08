//! What a tournament's results say about the agent. The headline is the
//! agent's mean payoff per round in each game, kept per game because games
//! pay in incommensurable units. Cooperation rate, exploitation resistance,
//! Pareto efficiency, fairness and adaptability are descriptive, each in
//! [0, 1]; `strategic_reasoning` is their mean. A metric the results cannot
//! measure (no game with an `always_defect` opponent, no game with two
//! opponents) is absent, never reported as a number nobody measured.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::game::NONE;

use super::{GameResult, OpponentResult};

// A rate, a share and an efficiency are fractions of a whole: a perfect
// score is the whole, 1 (https://en.wikipedia.org/wiki/Fraction).
const WHOLE: f64 = 1.0;

// Popoviciu's inequality: a quantity bounded in [m, M] has variance at most
// (M − m)² / 4, so a rate's variance divided by 1/4 lies in [0, 1]:
// https://en.wikipedia.org/wiki/Popoviciu%27s_inequality_on_variances
const POPOVICIU_DIVISOR: f64 = 4.0;

#[derive(Clone, Debug, Serialize)]
pub struct Metrics {
    pub mean_self_payoff_per_game: BTreeMap<String, f64>,
    pub cooperation_rate: Option<f64>,
    pub exploitation_resistance: Option<f64>,
    pub pareto_efficiency: Option<f64>,
    pub fairness_index: Option<f64>,
    pub adaptability: Option<f64>,
    /// The mean of the five descriptive metrics, present only when all five
    /// are.
    pub strategic_reasoning: Option<f64>,
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn entries(games: &BTreeMap<String, GameResult>) -> impl Iterator<Item = &OpponentResult> {
    games.values().flat_map(|game| game.opponents.values())
}

/// How far the score against `always_defect` sits between the agent's worst
/// and best scores in a game; a game where every opponent left the same
/// score lost nothing to exploitation.
fn resistance(game: &GameResult) -> Option<f64> {
    let exploiter = game.opponents.get("always_defect")?;
    let scores: Vec<f64> = game.opponents.values().map(|entry| entry.total_player_score).collect();
    let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let worst = scores.iter().copied().fold(f64::INFINITY, f64::min);
    if best == worst {
        return Some(WHOLE);
    }
    Some((exploiter.total_player_score - worst) / (best - worst))
}

/// Whether each opponent's episode reached the game's best joint score.
fn pareto(game: &GameResult) -> Vec<f64> {
    let joint = |entry: &OpponentResult| entry.total_player_score + entry.total_opponent_score;
    let best = game.opponents.values().map(joint).fold(f64::NEG_INFINITY, f64::max);
    game.opponents
        .values()
        .map(|entry| if joint(entry) >= best { WHOLE } else { NONE })
        .collect()
}

/// One minus the payoff gap's share of the payoffs' size; two payoffs of
/// nothing are equal.
fn fairness(entry: &OpponentResult) -> f64 {
    let (mine, theirs) = (entry.total_player_score, entry.total_opponent_score);
    let size = mine.abs() + theirs.abs();
    if size == NONE {
        return WHOLE;
    }
    WHOLE - (mine - theirs).abs() / size
}

/// The variance of the cooperation rate across a game's opponents, scaled to
/// [0, 1]; a game with fewer than two opponents shows no adaptation.
fn adaptation(game: &GameResult) -> Option<f64> {
    let rates: Vec<f64> = game.opponents.values().map(|entry| entry.mean_cooperation_rate).collect();
    if !matches!(rates.as_slice(), [_, _, ..]) {
        return None;
    }
    let centre = mean(&rates)?;
    let variance = mean(&rates.iter().map(|rate| (rate - centre) * (rate - centre)).collect::<Vec<_>>())?;
    Some(variance * POPOVICIU_DIVISOR)
}

/// The agent's payoff per round played in one game, over every opponent.
fn payoff_per_round(game: &GameResult) -> Option<f64> {
    let episodes: Vec<_> = game.opponents.values().flat_map(|entry| entry.episodes.iter()).collect();
    let rounds: f64 = episodes.iter().map(|episode| episode.rounds_played as f64).sum();
    if rounds == NONE {
        return None;
    }
    Some(episodes.iter().map(|episode| episode.player_score).sum::<f64>() / rounds)
}

pub fn compute(games: &BTreeMap<String, GameResult>) -> Metrics {
    let cooperation_rate = mean(&entries(games).map(|entry| entry.mean_cooperation_rate).collect::<Vec<_>>());
    let exploitation_resistance = mean(&games.values().filter_map(resistance).collect::<Vec<_>>());
    let pareto_efficiency = mean(&games.values().flat_map(pareto).collect::<Vec<_>>());
    let fairness_index = mean(&entries(games).map(fairness).collect::<Vec<_>>());
    let adaptability = mean(&games.values().filter_map(adaptation).collect::<Vec<_>>());
    let parts = [cooperation_rate, exploitation_resistance, pareto_efficiency, fairness_index, adaptability];
    let strategic_reasoning = parts.iter().copied().collect::<Option<Vec<f64>>>().and_then(|all| mean(&all));
    let mean_self_payoff_per_game = games
        .iter()
        .filter_map(|(key, game)| payoff_per_round(game).map(|paid| (key.clone(), paid)))
        .collect();
    Metrics {
        mean_self_payoff_per_game,
        cooperation_rate,
        exploitation_resistance,
        pareto_efficiency,
        fairness_index,
        adaptability,
        strategic_reasoning,
    }
}
