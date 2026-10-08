//! Tournaments as `kant tournament` runs them: every pairing is played the
//! declared number of times, the totals are the episodes' sums, a metric the
//! results cannot measure is absent, and a model seat without Brama is
//! refused by name.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const SETTINGS: &str = "{
  \"seed\": 21,
  \"evaluation\": { \"episodes\": 2 },
  \"agent\": { \"history_rounds\": 5 },
  \"games\": {
    \"prisoners_dilemma\": {
      \"rounds\": 3,
      \"payoffs\": {
        \"cooperate\": { \"cooperate\": [3, 3], \"defect\": [0, 5] },
        \"defect\": { \"cooperate\": [5, 0], \"defect\": [1, 1] }
      }
    }
  }
}";

fn settings_file(name: &str) -> String {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("settings.json");
    std::fs::write(&path, SETTINGS).expect("settings file");
    path.to_str().expect("a UTF-8 path").to_owned()
}

fn kant(args: &[&str]) -> Output {
    Command::new(BINARY)
        .args(args)
        .env_remove("BRAMA_URL")
        .env_remove("BRAMA_API_KEY")
        .output()
        .expect("kant runs")
}

fn declared() -> Value {
    serde_json::from_str(SETTINGS).expect("settings parse")
}

#[test]
fn every_pairing_is_played_the_declared_number_of_times_and_scored() {
    let settings = settings_file("tournament-strategies");
    let output = kant(&[
        "tournament", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "always_defect",
        "--strategy", "always_cooperate", "--agent-strategy", "tit_for_tat",
    ]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result: Value = serde_json::from_slice(&output.stdout).expect("one JSON document");
    let episodes = declared()["evaluation"]["episodes"].as_u64().expect("episodes");
    let opponents = result["games"]["prisoners_dilemma"]["opponents"].as_object().expect("opponents").clone();
    assert_eq!(result["total_episodes"].as_u64(), Some(episodes * opponents.len() as u64));
    for (name, entry) in &opponents {
        let played = entry["episodes"].as_array().expect("episodes");
        assert_eq!(played.len() as u64, episodes, "{name}");
        let summed: f64 = played.iter().filter_map(|episode| episode["player_score"].as_f64()).sum();
        assert_eq!(entry["total_player_score"].as_f64(), Some(summed), "{name}");
    }
    let rate = |name: &str| opponents[name]["mean_cooperation_rate"].as_f64().expect("a rate");
    assert!(
        rate("always_cooperate") > rate("always_defect"),
        "tit for tat cooperates more with a cooperator than with a defector"
    );
    let metrics = &result["metrics"];
    assert!(metrics["exploitation_resistance"].is_number(), "{metrics}");
    assert!(metrics["strategic_reasoning"].is_number(), "{metrics}");
    assert_eq!(result["settings"], declared());
}

#[test]
fn a_metric_the_results_cannot_measure_is_absent() {
    let settings = settings_file("tournament-absent");
    let output = kant(&[
        "tournament", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "grudger",
        "--agent-strategy", "always_cooperate",
    ]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result: Value = serde_json::from_slice(&output.stdout).expect("one JSON document");
    let metrics = &result["metrics"];
    assert!(metrics["exploitation_resistance"].is_null(), "no always_defect opponent: {metrics}");
    assert!(metrics["adaptability"].is_null(), "one opponent shows no adaptation: {metrics}");
    assert!(metrics["strategic_reasoning"].is_null(), "{metrics}");
}

#[test]
fn a_model_seat_without_brama_and_an_unnamed_seat_are_refused() {
    let settings = settings_file("tournament-refusals");
    let model = kant(&[
        "tournament", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "grudger",
        "--agent-route", "any-route",
    ]);
    assert!(!model.status.success());
    let said = String::from_utf8_lossy(&model.stderr);
    assert!(said.contains("BRAMA_URL is not set"), "{said}");

    let unnamed = kant(&["tournament", "--settings", &settings, "--game", "prisoners_dilemma", "--strategy", "grudger"]);
    assert!(!unnamed.status.success());
    let said = String::from_utf8_lossy(&unnamed.stderr);
    assert!(said.contains("exactly one of --agent-route"), "{said}");
}
