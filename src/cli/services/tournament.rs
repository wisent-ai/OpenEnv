//! `kant tournament`: the agent against every named strategy in every named
//! game, scored. The agent's seat is a model behind Brama
//! (`--agent-route R`) or a library strategy (`--agent-strategy S`), never
//! both and never neither.

use serde_json::{json, Value};

use crate::cli::Words;
use crate::error::{Error, Result};
use crate::evaluation::{Runner, SeatSpec};

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let agent = match (words.one("agent-route")?, words.one("agent-strategy")?) {
        (Some(route), None) => SeatSpec::Model(route.to_owned()),
        (None, Some(name)) => SeatSpec::Strategy(name.to_owned()),
        _ => {
            return Err(Error::Usage(
                "name the agent's seat with exactly one of --agent-route R (a model behind Brama) or --agent-strategy S".to_owned(),
            ))
        }
    };
    let games: Vec<String> = words.all("game").into_iter().map(str::to_owned).collect();
    let strategies: Vec<String> = words.all("strategy").into_iter().map(str::to_owned).collect();
    let opponent = words.one("opponent-route")?.map(str::to_owned);
    let tournament = Runner::new(settings.clone(), agent, opponent)?.run(&games, &strategies)?;
    let mut answer = serde_json::to_value(&tournament).map_err(|source| Error::Json { origin: "tournament".to_owned(), source })?;
    answer["settings"] = json!(settings.document());
    if let Some(path) = words.one("report")? {
        let path = std::path::PathBuf::from(path);
        std::fs::write(&path, crate::evaluation::report::markdown(&tournament))
            .map_err(|source| Error::Io { path: path.clone(), source })?;
        answer["report"] = json!(path);
    }
    Ok(answer)
}

/// `kant group-tournament --settings FILE --game G... --strategy S...
/// --agent-strategy A [--governance S]`: group and coalition games, the
/// agent's seat held by a group or coalition strategy.
pub fn group(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let games: Vec<String> = words.all("game").into_iter().map(str::to_owned).collect();
    let strategies: Vec<String> = words.all("strategy").into_iter().map(str::to_owned).collect();
    let agent = words.required("agent-strategy")?;
    let played = crate::evaluation::group::run(settings.clone(), agent, &games, &strategies, words.one("governance")?)?;
    let mut answer = serde_json::to_value(&played).map_err(|source| Error::Json { origin: "group tournament".to_owned(), source })?;
    answer["settings"] = json!(settings.document());
    Ok(answer)
}
