//! Group and coalition episodes as `kant group` and `kant coalition` play
//! them: each seat is paid from the declared numbers, a coalition member
//! that breaks its agreement under penalty enforcement is fined the declared
//! share, and a strategy list that fits no seat count is refused.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const SETTINGS: &str = "{
  \"seed\": 5,
  \"games\": {
    \"nplayer_public_goods\": { \"players\": 3, \"rounds\": 1, \"endowment\": 10, \"multiplier\": 1.5 },
    \"coalition_cartel\": {
      \"players\": 4, \"rounds\": 1, \"penalty\": 0.5, \"holds_at\": 3,
      \"colluding_held\": 6, \"colluding_broken\": 1, \"competing_held\": 8, \"competing_broken\": 3
    }
  }
}";

const SCRIPT: &str = "[
  { \"negotiate\": { \"proposals\": [ { \"proposer\": 0, \"members\": [0, 1, 2, 3], \"agreed_action\": \"collude\" } ] } },
  { \"move\": \"compete\" }
]";

fn scratch(name: &str, file: &str, text: &str) -> String {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join(file);
    std::fs::write(&path, text).expect("scratch file");
    path.to_str().expect("a UTF-8 path").to_owned()
}

fn kant(args: &[&str]) -> Output {
    Command::new(BINARY).args(args).output().expect("kant runs")
}

fn answer(output: &Output) -> Value {
    assert!(output.status.success(), "kant refused: {}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).expect("one JSON document")
}

fn declared() -> Value {
    serde_json::from_str(SETTINGS).expect("settings parse")
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("a number")
}

#[test]
fn a_free_rider_keeps_its_endowment_and_shares_the_others_pool() {
    let settings = scratch("group-public-goods", "settings.json", SETTINGS);
    let played = answer(&kant(&[
        "group", "--settings", &settings, "--game", "nplayer_public_goods", "--strategy", "always_cooperate",
        "--move", "contribute_0",
    ]));
    let game = &declared()["games"]["nplayer_public_goods"];
    let round = played["state"]["history"].as_array().and_then(|rounds| rounds.first()).cloned().expect("a round");
    let moves = round["actions"].as_array().expect("moves").clone();
    assert_eq!(moves.len() as f64, number(&game["players"]), "every declared seat moved");
    let pool: f64 = moves
        .iter()
        .map(|played| {
            played
                .as_str()
                .and_then(|text| text.rsplit_once('_'))
                .and_then(|(_, amount)| amount.parse::<f64>().ok())
                .expect("an amount")
        })
        .sum();
    let share = pool * number(&game["multiplier"]) / number(&game["players"]);
    let agent = round["payoffs"].as_array().and_then(|paid| paid.first()).cloned().expect("the agent's payoff");
    assert_eq!(agent.as_f64(), Some(number(&game["endowment"]) + share));
}

#[test]
fn breaking_a_cartel_agreement_under_penalty_enforcement_costs_the_declared_share() {
    let settings = scratch("coalition-cartel", "settings.json", SETTINGS);
    let script = scratch("coalition-cartel", "script.json", SCRIPT);
    let played = answer(&kant(&[
        "coalition", "--settings", &settings, "--game", "coalition_cartel", "--strategy", "coalition_loyal",
        "--governance", "governance_passive", "--script", &script,
    ]));
    let last = played["observations"].as_array().and_then(|seen| seen.last()).cloned().expect("observations");
    let round = last["coalition_history"].as_array().and_then(|rounds| rounds.first()).cloned().expect("a coalition round");
    let defectors: Vec<u64> = round["defectors"].as_array().expect("defectors").iter().filter_map(Value::as_u64).collect();
    assert_eq!(defectors.len(), defectors.iter().filter(|seat| **seat == crate_agent()).count(), "only the agent defected");
    assert!(!defectors.is_empty(), "the agent's broken agreement is recorded");
    let game = &declared()["games"]["coalition_cartel"];
    let competing = number(&game["competing_held"]);
    let fine = round["penalties"].as_array().and_then(|fines| fines.first()).and_then(Value::as_f64);
    assert_eq!(fine, Some(competing * number(&game["penalty"])));
    assert_eq!(last["base"]["reward"].as_f64(), Some(competing - competing * number(&game["penalty"])));
}

/// The agent's seat, as the README states it: seat zero.
fn crate_agent() -> u64 {
    kantbench::group::AGENT_SEAT as u64
}

#[test]
fn a_strategy_list_that_fits_no_seat_count_is_refused() {
    let settings = scratch("group-refusal", "settings.json", SETTINGS);
    let output = kant(&[
        "group", "--settings", &settings, "--game", "nplayer_public_goods", "--strategy", "random", "--strategy",
        "random", "--strategy", "random", "--move", "contribute_0",
    ]);
    assert!(!output.status.success());
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(said.contains("name one strategy for every other seat"), "{said}");
}
