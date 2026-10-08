//! `kant play`: one episode against a library strategy, the agent's moves
//! given in order with `--move`. The answer records the seed, the settings
//! document, every observation and the final state, so the episode can be
//! replayed and its numbers traced.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::env::{Environment, GameAction, Reset};
use crate::error::{Error, Result};
use crate::game::Library;

use crate::cli::Words;

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let moves = words.all("move");
    if moves.is_empty() {
        return Err(Error::Usage("give the agent's moves in order with --move M".to_owned()));
    }
    let request = Reset {
        game: words.required("game")?.to_owned(),
        strategy: Some(words.required("strategy")?.to_owned()),
        rounds: words.count("rounds")?,
        episode_id: words.one("episode")?.map(str::to_owned),
    };
    let mut environment = Environment::new(Arc::new(Library::standard()), settings.clone())?;
    let mut observations = vec![serde_json::to_value(environment.reset(&request, None)?)
        .map_err(|source| Error::Json { origin: "observation".to_owned(), source })?];
    for action in moves {
        let observation = environment.step(&GameAction::new(action))?;
        observations.push(
            serde_json::to_value(observation)
                .map_err(|source| Error::Json { origin: "observation".to_owned(), source })?,
        );
    }
    Ok(json!({
        "seed": environment.seed(),
        "settings": settings.document(),
        "observations": observations,
        "state": environment.state(),
    }))
}
