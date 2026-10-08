//! Matchups as `kant matchups` plays them: every ordered pair of seats,
//! itself included, plays every game the declared number of times; an arena
//! reputation is reported only when the settings declare one; a malformed
//! seat is refused by its spelling.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const GAMES: &str = "\"games\": {
    \"prisoners_dilemma\": {
      \"rounds\": 2,
      \"payoffs\": {
        \"cooperate\": { \"cooperate\": [3, 3], \"defect\": [0, 5] },
        \"defect\": { \"cooperate\": [5, 0], \"defect\": [1, 1] }
      }
    }
  }";

const ARENA: &str = "\"arena\": { \"prior\": 0.5, \"decay\": 0.5, \"weights\": { \"cooperation\": 0.5, \"fairness\": 0.5 } },";

fn settings_file(name: &str, arena: &str) -> (String, Value) {
    let text = format!("{{ \"seed\": 2, \"evaluation\": {{ \"episodes\": 2 }}, {arena} {GAMES} }}");
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("settings.json");
    std::fs::write(&path, &text).expect("settings file");
    (path.to_str().expect("a UTF-8 path").to_owned(), serde_json::from_str(&text).expect("settings parse"))
}

fn kant(args: &[&str]) -> Output {
    Command::new(BINARY).args(args).output().expect("kant runs")
}

const SEATS: &[&str] = &["kind=strategy:always_cooperate", "mean=strategy:always_defect"];

#[test]
fn every_ordered_pair_plays_and_the_declared_arena_ranks_the_cooperator_higher() {
    let (settings, declared) = settings_file("matchups-arena", ARENA);
    let mut args = vec!["matchups", "--settings", settings.as_str(), "--game", "prisoners_dilemma"];
    for seat in SEATS {
        args.extend(["--seat", seat]);
    }
    let output = kant(&args);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result: Value = serde_json::from_slice(&output.stdout).expect("one JSON document");
    let episodes = declared["evaluation"]["episodes"].as_u64().expect("episodes") as usize;
    let played = result["matchups"].as_array().expect("matchups");
    assert_eq!(played.len(), SEATS.len() * SEATS.len() * episodes, "every ordered pair, each seat against itself too");
    let reputation = &result["reputation"];
    assert!(reputation["kind"].as_f64() > reputation["mean"].as_f64(), "{reputation}");
}

#[test]
fn no_arena_means_no_reputation_and_a_malformed_seat_is_refused() {
    let (settings, _) = settings_file("matchups-plain", "");
    let output = kant(&["matchups", "--settings", &settings, "--game", "prisoners_dilemma", "--seat", "kind=strategy:grudger"]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result: Value = serde_json::from_slice(&output.stdout).expect("one JSON document");
    assert!(result["reputation"].is_null(), "{}", result["reputation"]);

    let refused = kant(&["matchups", "--settings", &settings, "--game", "prisoners_dilemma", "--seat", "kind=grudger"]);
    assert!(!refused.status.success());
    let said = String::from_utf8_lossy(&refused.stderr);
    assert!(said.contains("is not NAME=route:R or NAME=strategy:S"), "{said}");
}
