//! `kant dataset --settings FILE --game G... --strategy S... --agent-strategy A
//! --output DIRECTORY`: the training inputs Ster reads (`prompts.json` for
//! `ster tune grpo`, `pairs.json` for `ster tune dpo`) and the `states.json`
//! `kant serve --states` scores answers against.

use std::path::PathBuf;

use serde_json::Value;

use crate::cli::Words;
use crate::error::Result;
use crate::training;

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let games: Vec<String> = words.all("game").into_iter().map(str::to_owned).collect();
    let strategies: Vec<String> = words.all("strategy").into_iter().map(str::to_owned).collect();
    let agent = words.required("agent-strategy")?;
    let directory = PathBuf::from(words.required("output")?);
    let found = training::states(&settings, &games, &strategies, agent)?;
    training::write_dataset(&settings, &found, &directory)
}
