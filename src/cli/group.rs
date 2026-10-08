//! `kant group` and `kant coalition`: one episode of a group game, the agent
//! holding seat zero. A group episode takes the agent's moves in order; a
//! coalition episode takes a script of steps, each either a negotiation
//! (`{"negotiate": {proposals, responses, governance_proposals,
//! governance_votes}}`) or a move (`{"move": "collude"}`).

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::coalition::{CoalitionAction, CoalitionEnvironment, CoalitionReset};
use crate::env::GameAction;
use crate::error::{Error, Result};
use crate::group::environment::{GroupEnvironment, Seat};
use crate::group::{GroupLibrary, AGENT_SEATS};

use super::Words;

fn document<T: serde::Serialize>(value: &T, what: &str) -> Result<Value> {
    serde_json::to_value(value).map_err(|source| Error::Json { origin: what.to_owned(), source })
}

pub fn group(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let library = Arc::new(GroupLibrary::standard());
    let key = words.required("game")?;
    let players = library.build(key, &settings)?.players;
    let names: Vec<String> = words.all("strategy").into_iter().map(str::to_owned).collect();
    let seats = Seat::strategies(&names, players.saturating_sub(AGENT_SEATS))?;
    let moves = words.all("move");
    if moves.is_empty() {
        return Err(Error::Usage("give the agent's moves in order with --move M".to_owned()));
    }
    let mut environment = GroupEnvironment::new(library, settings.clone())?;
    let mut observations = vec![document(&environment.reset(key, seats, words.count("rounds")?, words.one("episode")?.map(str::to_owned))?, "observation")?];
    for played in moves {
        observations.push(document(&environment.step(&GameAction::new(played))?, "observation")?);
    }
    Ok(json!({
        "seed": environment.seed(),
        "settings": settings.document(),
        "observations": observations,
        "state": environment.state(),
    }))
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    Negotiate(CoalitionAction),
    Move(String),
}

pub fn coalition(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let script_path = std::path::PathBuf::from(words.required("script")?);
    let text = std::fs::read_to_string(&script_path).map_err(|source| Error::Io { path: script_path.clone(), source })?;
    let steps: Vec<Step> = serde_json::from_str(&text).map_err(|source| Error::Json {
        origin: script_path.display().to_string(),
        source,
    })?;
    let request = CoalitionReset {
        game: words.required("game")?.to_owned(),
        strategies: words.all("strategy").into_iter().map(str::to_owned).collect(),
        governance: words.all("governance").into_iter().map(str::to_owned).collect(),
        rounds: words.count("rounds")?,
        episode_id: words.one("episode")?.map(str::to_owned),
    };
    let mut environment = CoalitionEnvironment::new(Arc::new(GroupLibrary::standard()), settings.clone())?;
    let mut observations = vec![document(&environment.reset(&request)?, "observation")?];
    for step in steps {
        let observation = match step {
            Step::Negotiate(action) => environment.negotiate(&action)?,
            Step::Move(played) => environment.act(&GameAction::new(&played))?,
        };
        observations.push(document(&observation, "observation")?);
    }
    Ok(json!({
        "seed": environment.seed(),
        "settings": settings.document(),
        "observations": observations,
    }))
}
