//! What the play tests share: the built `kant`, a settings document written
//! under cargo's scratch area, and readers for its answers and refusals.

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

/// The settings these tests declare: two games and one responder strategy.
pub const SETTINGS: &str = "{
  \"seed\": 11,
  \"games\": {
    \"prisoners_dilemma\": {
      \"rounds\": 3,
      \"payoffs\": {
        \"cooperate\": { \"cooperate\": [3, 3], \"defect\": [0, 5] },
        \"defect\": { \"cooperate\": [5, 0], \"defect\": [1, 1] }
      }
    },
    \"ultimatum\": { \"rounds\": 1, \"pot\": 10 }
  },
  \"strategies\": {
    \"ultimatum_fair\": { \"offer\": 5, \"accept_at_least\": 4 }
  }
}";

pub fn declared() -> Value {
    serde_json::from_str(SETTINGS).expect("settings parse")
}

/// `SETTINGS` written to a directory of the calling test's own.
pub fn settings_file(name: &str) -> String {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let path = directory.join("settings.json");
    std::fs::write(&path, SETTINGS).expect("settings file");
    path.to_str().expect("a UTF-8 path").to_owned()
}

pub fn kant(args: &[&str]) -> Output {
    Command::new(BINARY).args(args).output().expect("kant runs")
}

pub fn answer(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "kant refused: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("one JSON document")
}

pub fn refusal(output: &Output) -> String {
    assert!(!output.status.success(), "kant answered where it must refuse");
    String::from_utf8_lossy(&output.stderr).into_owned()
}
