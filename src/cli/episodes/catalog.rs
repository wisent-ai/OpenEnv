//! `kant games` and `kant strategies`: what the library holds and what each
//! entry reads from a settings document. With `--settings FILE` each entry
//! also says whether that document lets it be built, and if not, why.

use serde_json::{json, Value};

use crate::error::Result;
use crate::game::Library;
use crate::settings::Settings;
use crate::strategy;

use crate::cli::Words;

fn optional_settings(words: &Words) -> Result<Option<Settings>> {
    match words.one("settings")? {
        None => Ok(None),
        Some(path) => Ok(Some(Settings::read(std::path::Path::new(path))?)),
    }
}

pub fn games(words: &Words) -> Result<Value> {
    let library = Library::standard();
    let settings = optional_settings(words)?;
    let rows: Vec<Value> = library
        .entries()
        .map(|entry| {
            let mut reads: Vec<&str> = vec!["rounds"];
            reads.extend(entry.parameters);
            let mut row = json!({
                "key": entry.key,
                "family": entry.family,
                "reads": reads,
            });
            if let Some(settings) = &settings {
                row["built"] = match library.build(entry.key, settings) {
                    Ok(game) => game.summary(),
                    Err(refusal) => json!({ "refused": refusal.to_string() }),
                };
            }
            row
        })
        .collect();
    Ok(json!({ "games": rows }))
}

pub fn strategies(words: &Words) -> Result<Value> {
    let settings = optional_settings(words)?;
    let rows: Vec<Value> = strategy::NAMES
        .iter()
        .map(|name| {
            let mut row = json!({
                "name": name,
                "reads": strategy::parameters(name),
            });
            if let Some(settings) = &settings {
                row["ready"] = match strategy::named(name, settings) {
                    Ok(_) => Value::Bool(true),
                    Err(refusal) => json!({ "refused": refusal.to_string() }),
                };
            }
            row
        })
        .collect();
    Ok(json!({ "strategies": rows }))
}
