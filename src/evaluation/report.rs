//! A tournament as a Markdown report for a person: a summary, each game's
//! results per opponent, each opponent across games, and the metrics. Values
//! are written as the tournament recorded them; a metric it could not
//! measure says so.

use std::collections::BTreeMap;

use super::{OpponentResult, SeatSpec, Tournament};

fn shown(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value}"),
        None => "not measurable from these results".to_owned(),
    }
}

fn seat(spec: &SeatSpec) -> String {
    match spec {
        SeatSpec::Model(route) => format!("model {route}"),
        SeatSpec::Strategy(name) => format!("strategy {name}"),
    }
}

pub fn markdown(tournament: &Tournament) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut opponents: Vec<&String> = tournament.games.values().flat_map(|game| game.opponents.keys()).collect();
    opponents.sort();
    opponents.dedup();
    lines.push("# KantBench Evaluation Report\n\n## Summary\n\n| Attribute | Value |\n|---|---|".to_owned());
    lines.push(format!("| Agent | {} |", seat(&tournament.agent)));
    lines.push(format!("| Games | {} |", tournament.games.len()));
    lines.push(format!("| Opponents | {} |", opponents.len()));
    lines.push(format!("| Episodes per pairing | {} |", tournament.episodes_per_pairing));
    lines.push(format!("| Total episodes | {} |", tournament.total_episodes));
    lines.push(format!("| Seed | {} |", tournament.seed));
    lines.push(format!("| Strategic reasoning | {} |", shown(tournament.metrics.strategic_reasoning)));

    lines.push("\n## Per-game results".to_owned());
    for (key, game) in &tournament.games {
        lines.push(format!(
            "\n### {key} ({})\n\n| Opponent | Agent score | Opponent score | Cooperation rate |\n|---|---|---|---|",
            game.name
        ));
        for (name, entry) in &game.opponents {
            lines.push(format!(
                "| {name} | {} | {} | {} |",
                entry.total_player_score, entry.total_opponent_score, entry.mean_cooperation_rate
            ));
        }
    }

    lines.push("\n## Opponents across games\n\n| Opponent | Agent score | Opponent score | Mean cooperation rate | Games |\n|---|---|---|---|---|".to_owned());
    let mut by_opponent: BTreeMap<&String, Vec<&OpponentResult>> = BTreeMap::new();
    for game in tournament.games.values() {
        for (name, entry) in &game.opponents {
            by_opponent.entry(name).or_default().push(entry);
        }
    }
    for (name, entries) in by_opponent {
        let mine: f64 = entries.iter().map(|entry| entry.total_player_score).sum();
        let theirs: f64 = entries.iter().map(|entry| entry.total_opponent_score).sum();
        let rate = entries.iter().map(|entry| entry.mean_cooperation_rate).sum::<f64>() / entries.len() as f64;
        lines.push(format!("| {name} | {mine} | {theirs} | {rate} | {} |", entries.len()));
    }

    let metrics = &tournament.metrics;
    lines.push("\n## Metrics\n\n| Metric | Value |\n|---|---|".to_owned());
    for (name, value) in [
        ("cooperation_rate", metrics.cooperation_rate),
        ("exploitation_resistance", metrics.exploitation_resistance),
        ("pareto_efficiency", metrics.pareto_efficiency),
        ("fairness_index", metrics.fairness_index),
        ("adaptability", metrics.adaptability),
        ("strategic_reasoning", metrics.strategic_reasoning),
    ] {
        lines.push(format!("| {name} | {} |", shown(value)));
    }
    lines.push("\n### Mean payoff per round\n\n| Game | Agent payoff per round |\n|---|---|".to_owned());
    for (key, paid) in &metrics.mean_self_payoff_per_game {
        lines.push(format!("| {key} | {paid} |"));
    }
    lines.join("\n")
}
