//! Composed and declared games as `kant play` builds them: a variant named in
//! the key reads its numbers from `variants.<name>`, a custom game is built
//! from its declaration alone, and a key that is both a library game and a
//! custom game is refused.

#[path = "support.rs"]
mod support;

use serde_json::Value;
use support::{answer, kant, refusal};

const SETTINGS: &str = "{
  \"seed\": 3,
  \"games\": {
    \"prisoners_dilemma\": {
      \"rounds\": 1,
      \"payoffs\": {
        \"cooperate\": { \"cooperate\": [3, 3], \"defect\": [0, 5] },
        \"defect\": { \"cooperate\": [5, 0], \"defect\": [1, 1] }
      }
    }
  },
  \"variants\": { \"exit\": { \"payoff\": 2 } },
  \"custom_games\": {
    \"meeting\": {
      \"name\": \"Meeting\",
      \"description\": \"Two people pick a cafe.\",
      \"actions\": [\"north\", \"south\"],
      \"rounds\": 1,
      \"symmetric\": {
        \"north\": { \"north\": 4, \"south\": 0 },
        \"south\": { \"north\": 1, \"south\": 2 }
      }
    },
    \"stag_hunt\": {
      \"name\": \"Clash\",
      \"description\": \"Named like a library game.\",
      \"actions\": [\"stag\"],
      \"rounds\": 1,
      \"symmetric\": { \"stag\": { \"stag\": 1 } }
    }
  }
}";

fn declared() -> Value {
    serde_json::from_str(SETTINGS).expect("settings parse")
}

fn settings_file(name: &str) -> String {
    let directory = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("settings.json");
    std::fs::write(&path, SETTINGS).expect("settings file");
    path.to_str().expect("a UTF-8 path").to_owned()
}

fn only_round(played: &Value) -> Value {
    played["state"]["history"]
        .as_array()
        .and_then(|rounds| rounds.first())
        .cloned()
        .expect("one round")
}

#[test]
fn an_exit_named_in_the_key_pays_its_declared_payoff() {
    let settings = settings_file("composed-exit");
    let round = only_round(&answer(&kant(&[
        "play", "--settings", &settings, "--game", "exit_prisoners_dilemma", "--strategy", "always_defect",
        "--move", "exit",
    ])));
    let exit = declared()["variants"]["exit"]["payoff"].as_f64();
    assert_eq!(round["player_payoff"].as_f64(), exit);
    assert_eq!(round["opponent_payoff"].as_f64(), exit);
}

#[test]
fn a_symmetric_custom_game_pays_each_seat_its_own_row() {
    let settings = settings_file("composed-custom");
    let round = only_round(&answer(&kant(&[
        "play", "--settings", &settings, "--game", "meeting", "--strategy", "always_defect", "--move", "north",
    ])));
    let table = &declared()["custom_games"]["meeting"]["symmetric"];
    assert_eq!(round["opponent_action"], "south", "always_defect plays the second move");
    assert_eq!(round["player_payoff"].as_f64(), table["north"]["south"].as_f64());
    assert_eq!(round["opponent_payoff"].as_f64(), table["south"]["north"].as_f64());
}

#[test]
fn a_custom_game_named_like_a_library_game_and_an_undeclared_variant_are_refused() {
    let settings = settings_file("composed-refusals");
    let clash = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "stag_hunt", "--strategy", "grudger", "--move", "stag",
    ]));
    assert!(clash.contains("custom_games declares it too"), "{clash}");
    let noisy = refusal(&kant(&[
        "play", "--settings", &settings, "--game", "noisy_payoffs_prisoners_dilemma", "--strategy", "grudger",
        "--move", "cooperate",
    ]));
    assert!(noisy.contains("variants declares no noisy_payoffs"), "{noisy}");
}
