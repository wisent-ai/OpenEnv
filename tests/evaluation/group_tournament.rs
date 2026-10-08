//! Group and coalition tournaments as `kant group-tournament` runs them: a
//! group game's agent cooperation is measured against the game's first move,
//! a coalition game reports formation and defection, and a coalition game
//! without the other seats' governance strategy is refused.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const SETTINGS: &str = "{
  \"seed\": 6,
  \"evaluation\": { \"episodes\": 2 },
  \"games\": {
    \"nplayer_volunteer_dilemma\": { \"players\": 3, \"rounds\": 2, \"benefit\": 4, \"cost\": 1, \"nobody\": 0 },
    \"coalition_cartel\": {
      \"players\": 3, \"rounds\": 2, \"penalty\": 0.5, \"holds_at\": 2,
      \"colluding_held\": 6, \"colluding_broken\": 1, \"competing_held\": 8, \"competing_broken\": 3
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
    Command::new(BINARY).args(args).output().expect("kant runs")
}

fn answer(output: &Output) -> Value {
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).expect("one JSON document")
}

#[test]
fn an_always_cooperating_agent_cooperates_every_round_of_a_group_game() {
    let settings = settings_file("group-tournament");
    let result = answer(&kant(&[
        "group-tournament", "--settings", &settings, "--game", "nplayer_volunteer_dilemma", "--strategy", "always_defect",
        "--agent-strategy", "always_cooperate",
    ]));
    let declared: Value = serde_json::from_str(SETTINGS).expect("settings parse");
    let episodes = result["games"]["nplayer_volunteer_dilemma"]["always_defect"].as_array().expect("episodes").clone();
    assert_eq!(episodes.len() as u64, declared["evaluation"]["episodes"].as_u64().expect("episodes"));
    for episode in &episodes {
        let rate = episode["cooperation_rate"].as_f64().expect("a rate");
        let all_rounds = episode["rounds_played"].as_f64().expect("rounds") / episode["rounds_played"].as_f64().expect("rounds");
        assert_eq!(rate, all_rounds, "{episode}");
        let game = &declared["games"]["nplayer_volunteer_dilemma"];
        let per_round = game["benefit"].as_f64().expect("benefit") - game["cost"].as_f64().expect("cost");
        assert_eq!(episode["player_score"].as_f64(), Some(per_round * episode["rounds_played"].as_f64().expect("rounds")));
    }
}

#[test]
fn a_coalition_game_reports_formation_and_needs_a_governance_strategy() {
    let settings = settings_file("coalition-tournament");
    let result = answer(&kant(&[
        "group-tournament", "--settings", &settings, "--game", "coalition_cartel", "--strategy", "coalition_loyal",
        "--agent-strategy", "coalition_loyal", "--governance", "governance_passive",
    ]));
    let episode = result["games"]["coalition_cartel"]["coalition_loyal"]
        .as_array()
        .and_then(|episodes| episodes.first())
        .cloned()
        .expect("an episode");
    assert!(episode["coalition_rate"].is_number(), "{episode}");
    assert!(episode["defection_rate"].is_number(), "{episode}");

    let refused = kant(&[
        "group-tournament", "--settings", &settings, "--game", "coalition_cartel", "--strategy", "coalition_loyal",
        "--agent-strategy", "coalition_loyal",
    ]);
    assert!(!refused.status.success());
    let said = String::from_utf8_lossy(&refused.stderr);
    assert!(said.contains("name the other seats' governance strategy"), "{said}");
}
