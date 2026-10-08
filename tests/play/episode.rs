//! Episodes as an operator plays them: the built `kant play` over a settings
//! document on disk. Scores must be the declared payoffs of the rounds
//! played; a responder must answer the offer it is shown; an undeclared game,
//! an unknown move and a strategy without its numbers are refused by name.

#[path = "support.rs"]
mod support;

use serde_json::Value;
use support::{answer, declared, kant, refusal, settings_file};

fn cell(row: &str, column: &str) -> (f64, f64) {
    let pair = &declared()["games"]["prisoners_dilemma"]["payoffs"][row][column];
    let pair = pair.as_array().expect("a declared pair");
    let [player, opponent] = pair.as_slice() else {
        panic!("a pair has two sides")
    };
    (player.as_f64().expect("number"), opponent.as_f64().expect("number"))
}

fn first_round(played: &Value) -> Value {
    played["state"]["history"]
        .as_array()
        .and_then(|rounds| rounds.first())
        .cloned()
        .expect("one round played")
}

#[test]
fn tit_for_tat_mirrors_and_the_scores_are_the_declared_cells() {
    let settings = settings_file("play-dilemma");
    let played = answer(&kant(&[
        "play", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "tit_for_tat",
        "--move", "cooperate", "--move", "defect", "--move", "defect",
    ]));
    let state = &played["state"];
    let rounds = state["history"].as_array().expect("history");
    let moves: Vec<(&str, &str)> = rounds
        .iter()
        .map(|round| {
            (
                round["player_action"].as_str().expect("player move"),
                round["opponent_action"].as_str().expect("opponent move"),
            )
        })
        .collect();
    assert_eq!(
        moves,
        [("cooperate", "cooperate"), ("defect", "cooperate"), ("defect", "defect")],
        "tit for tat opens with cooperation, then plays the agent's previous move"
    );
    let mut expected: (f64, f64) = Default::default();
    for (row, column) in &moves {
        let (player, opponent) = cell(row, column);
        expected = (expected.0 + player, expected.1 + opponent);
    }
    assert_eq!(state["player_score"].as_f64(), Some(expected.0));
    assert_eq!(state["opponent_score"].as_f64(), Some(expected.1));
    assert_eq!(state["is_done"], Value::Bool(true));
    assert_eq!(played["seed"], declared()["seed"], "the declared seed is recorded");
    assert_eq!(played["settings"], declared(), "the settings document is recorded whole");
}

#[test]
fn the_responder_answers_the_offer_it_is_shown() {
    let settings = settings_file("play-ultimatum");
    let least = declared()["strategies"]["ultimatum_fair"]["accept_at_least"]
        .as_u64()
        .expect("threshold");
    let pot = declared()["games"]["ultimatum"]["pot"].as_u64().expect("pot");
    let generous = format!("offer_{least}");
    let accepted = first_round(&answer(&kant(&[
        "play", "--settings", &settings, "--game", "ultimatum", "--strategy", "ultimatum_fair",
        "--move", &generous,
    ])));
    assert_eq!(accepted["opponent_action"], "accept");
    assert_eq!(accepted["player_payoff"].as_f64(), Some((pot - least) as f64));
    assert_eq!(accepted["opponent_payoff"].as_f64(), Some(least as f64));

    // Offering nothing is below any threshold the settings declare.
    let rejected = first_round(&answer(&kant(&[
        "play", "--settings", &settings, "--game", "ultimatum", "--strategy", "ultimatum_fair",
        "--move", "offer_0",
    ])));
    assert_eq!(rejected["opponent_action"], "reject");
}

#[test]
fn undeclared_games_unknown_moves_and_missing_strategy_numbers_are_refused_by_name() {
    let settings = settings_file("play-refusals");
    let undeclared = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "stag_hunt", "--strategy", "grudger", "--move", "stag",
    ]));
    assert!(undeclared.contains("games declares no stag_hunt"), "{undeclared}");

    let unknown = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "grudger",
        "--move", "betray",
    ]));
    assert!(unknown.contains("betray is not a move of prisoners_dilemma"), "{unknown}");

    let numberless = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "mixed",
        "--move", "cooperate",
    ]));
    assert!(numberless.contains("strategies declares no mixed"), "{numberless}");

    let no_game = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "chess", "--strategy", "grudger", "--move", "e4",
    ]));
    assert!(no_game.contains("no game is named chess"), "{no_game}");
}
