//! `kant matchups --settings FILE --game G... --seat NAME=route:R
//! --seat NAME=strategy:S ...`: named seats play every game against each
//! other, with each seat's arena reputation when the settings declare one.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::cli::Words;
use crate::error::{Error, Result};
use crate::evaluation::{matchups, SeatSpec};

fn seat(spelled: &str) -> Result<(String, SeatSpec)> {
    let refuse = || Error::Usage(format!("--seat {spelled} is not NAME=route:R or NAME=strategy:S"));
    let (name, holder) = spelled.split_once('=').ok_or_else(refuse)?;
    let spec = match holder.split_once(':') {
        Some(("route", route)) if !route.is_empty() => SeatSpec::Model(route.to_owned()),
        Some(("strategy", strategy)) if !strategy.is_empty() => SeatSpec::Strategy(strategy.to_owned()),
        _ => return Err(refuse()),
    };
    if name.is_empty() {
        return Err(refuse());
    }
    Ok((name.to_owned(), spec))
}

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let mut seats = BTreeMap::new();
    for spelled in words.all("seat") {
        let (name, spec) = seat(spelled)?;
        if seats.insert(name.clone(), spec).is_some() {
            return Err(Error::Usage(format!("--seat {name} is named twice")));
        }
    }
    let games: Vec<String> = words.all("game").into_iter().map(str::to_owned).collect();
    let played = matchups::run(settings.clone(), seats, &games)?;
    let mut answer = serde_json::to_value(&played).map_err(|source| Error::Json { origin: "matchups".to_owned(), source })?;
    answer["settings"] = json!(settings.document());
    Ok(answer)
}
