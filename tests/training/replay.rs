//! Actual CLI dataset, persisted states, environment settlements and scorer replay.
use std::{env, error::Error, fs, path::PathBuf, process::Command, sync::Arc};

use kantbench::{
    env::{Environment, GameAction, Reset},
    game::Library,
    settings::Settings,
    training::{read_states, reward::Scorer},
};
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[test]
#[ignore = "requires declared noisy single-action game and candidate revision"]
fn persisted_payoff_streams_match_real_settlements_across_resets() -> Result<()> {
    let revision = env::var("KANT_TEST_SOURCE_REVISION")?;
    let supplied = PathBuf::from(env::var("KANT_TEST_REPLAY_SETTINGS")?);
    let key = env::var("KANT_TEST_REPLAY_GAME")?;
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !head.status.success() || String::from_utf8(head.stdout)?.trim() != revision {
        return Err("candidate source revision differs".into());
    }
    let clean = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    if !clean.status.success() || !clean.stdout.is_empty() {
        return Err("candidate checkout must be clean".into());
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/real-tests/payoff-replay")
        .join(uuid::Uuid::new_v4().to_string());
    let inputs = root.join("inputs");
    fs::create_dir_all(&inputs)?;
    let settings_path = inputs.join("settings.json");
    fs::copy(supplied, &settings_path)?;
    fs::copy(env!("CARGO_BIN_EXE_kant"), root.join("kant"))?;
    let mut report = json!({"revision": revision, "game": key, "settlements": []});
    let result = (|| -> Result<()> {
        let settings = Arc::new(Settings::read(&settings_path)?);
        report["settings"] = json!(settings.document());
        if settings.seed()?.is_none() {
            return Err("replay qualification requires a declared seed".into());
        }
        let library = Arc::new(Library::standard());
        let game = library.build(&key, &settings)?;
        if !game.has_variant("noisy_payoffs") {
            return Err("qualification game must include noisy_payoffs".into());
        }
        if settings
            .entry("variants", "noisy_payoffs")?
            .number("scale")?
            <= kantbench::game::NONE
        {
            return Err(
                "qualification requires positive payoff noise, not a deterministic variant".into(),
            );
        }
        if game.has_variant("free_chat") {
            return Err("this settlement journey requires action-only rounds".into());
        }
        let (played, other_agent_moves) = game.actions.split_first().ok_or("game has no move")?;
        let (_, other_opponent_moves) = game
            .opponent_moves()
            .split_first()
            .ok_or("opponent has no move")?;
        if !other_agent_moves.is_empty() || !other_opponent_moves.is_empty() {
            return Err("one move per seat is required: uniform expected payoff then equals the actual settlement".into());
        }
        let output = inputs.join("dataset");
        let args = [
            "dataset",
            "--settings",
            settings_path.to_str().ok_or("settings path is not UTF8")?,
            "--game",
            &key,
            "--strategy",
            "random",
            "--agent-strategy",
            "random",
            "--output",
            output.to_str().ok_or("dataset path is not UTF8")?,
        ];
        let written = Command::new(env!("CARGO_BIN_EXE_kant"))
            .args(args)
            .output()?;
        report["dataset_command"] = json!({"args": args, "exit": written.status.code(), "stdout": String::from_utf8(written.stdout)?, "stderr": String::from_utf8(written.stderr)?});
        if !written.status.success() {
            return Err(format!("dataset refused: {}", report["dataset_command"]).into());
        }
        let states_path = output.join("states.json");
        let states = read_states(&states_path)?;
        report["persisted_states"] = serde_json::from_slice(&fs::read(&states_path)?)?;
        let mut starts = states.iter().filter(|state| state.history.is_empty());
        if starts.next().is_none()
            || starts.next().is_none()
            || !states.iter().any(|state| !state.history.is_empty())
        {
            return Err("fixture must exercise multiple episodes and a later round".into());
        }
        let mut environment = Environment::new(library, settings.clone())?;
        for state in &states {
            if state.history.is_empty() {
                environment.reset(
                    &Reset {
                        game: key.clone(),
                        strategy: Some("random".to_owned()),
                        rounds: None,
                        episode_id: None,
                    },
                    None,
                )?;
            }
            let before = environment.state().ok_or("environment has no episode")?;
            if before.history.len() != state.history.len()
                || environment.payoff_seed()? != state.payoff_seed
            {
                return Err("persisted state does not identify the live payoff stream".into());
            }
            for (actual, recorded) in before.history.iter().zip(&state.history) {
                if actual.player_action != recorded.player_action
                    || actual.opponent_action != recorded.opponent_action
                    || actual.payoff_seed != recorded.payoff_seed
                {
                    return Err(
                        "persisted history differs from actual settlement provenance".into(),
                    );
                }
            }
            let scorer = Scorer::new(settings.clone(), vec![state.clone()])?;
            let expected = scorer.score(&state.prompt, played)?;
            let observed = environment.step(&GameAction::new(played))?;
            report["settlements"]
                .as_array_mut()
                .ok_or("settlement report is not an array")?
                .push(json!({"state": state, "scored": expected, "observed": observed}));
            if expected.reward != observed.reward {
                return Err(format!(
                    "payoff replay differs for seed {}: scored {}, settled {}",
                    state.payoff_seed, expected.reward, observed.reward
                )
                .into());
            }
        }
        let first = states.first().ok_or("dataset has no first state")?;
        let different = states
            .iter()
            .find(|state| state.payoff_seed != first.payoff_seed)
            .ok_or("fixture did not produce distinct payoff streams")?;
        let mut conflict = first.clone();
        conflict.payoff_seed = different.payoff_seed;
        match Scorer::new(settings, vec![first.clone(), conflict]) {
            Ok(_) => return Err("conflicting payoff streams were admitted under one prompt".into()),
            Err(error) => {
                let cause = error.to_string();
                report["conflicting_stream_refusal"] = json!(cause);
                if !cause.contains("payoff seeds") {
                    return Err(format!("refusal lost payoff provenance: {cause}").into());
                }
            }
        }
        let mut missing = json!({"states": [first]});
        missing["states"]
            .as_array_mut()
            .ok_or("states array")?
            .first_mut()
            .ok_or("first state")?
            .as_object_mut()
            .ok_or("state object")?
            .remove("payoff_seed")
            .ok_or("required payoff seed absent from generated state")?;
        let incomplete = inputs.join("incomplete.json");
        fs::write(&incomplete, serde_json::to_vec(&missing)?)?;
        match read_states(&incomplete) {
            Ok(_) => return Err("seed-only state was accepted without payoff provenance".into()),
            Err(error) => {
                let cause = error.to_string();
                report["missing_stream_refusal"] = json!(cause);
                if !cause.contains("payoff_seed")
                    || !cause.contains(incomplete.to_str().ok_or("non-UTF8 path")?)
                {
                    return Err(
                        format!("missing provenance refusal lost path/cause: {cause}").into(),
                    );
                }
            }
        }
        Ok(())
    })();
    report["error"] = json!(result.as_ref().err().map(ToString::to_string));
    let cleanup = fs::remove_dir_all(inputs);
    report["cleanup_error"] = json!(cleanup.as_ref().err().map(ToString::to_string));
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    cleanup?;
    result
}
