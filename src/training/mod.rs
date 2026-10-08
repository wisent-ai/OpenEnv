//! Training data and rewards for Ster, which owns the gradient
//! (`ster tune grpo`, `ster tune dpo`). KantBench writes what Ster reads and
//! pays what Ster asks it to score:
//!
//! - `prompts.json`, `{"prompts": [...]}`, the game states an agent meets,
//!   each rendered as the prompt a model seat reads;
//! - `states.json`, each prompt with the game, the rounds played before it
//!   and the seed, so a reward can be computed for an answer to it;
//! - `pairs.json`, a Ster pair set: for each state the move with the highest
//!   expected payoff (positive) against the one with the lowest (negative),
//!   each written after the prompt, when they differ by at least the declared
//!   `training.pair_margin`.
//!
//! The reward of a move is its expected self-payoff against an opponent who
//! plays every one of its moves equally often, in the state the prompt shows:
//! the agent's own payoff and nothing else, so cooperation measured later is
//! not a shaping term read back out.

pub mod reward;

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent::{prompt, StrategyAgent};
use crate::env::{Agent, Environment, Reset};
use crate::error::{Error, Result};
use crate::game::Library;
use crate::settings::Settings;

/// One state an agent meets: its prompt, the game, the rounds before it as
/// `[agent move, opponent move]`, and the seed the episode ran on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub prompt: String,
    pub game: String,
    pub history: Vec<(String, String)>,
    pub seed: u64,
}

/// Play `training.episodes` episodes of each game against each strategy with
/// `agent` holding the agent's seat, and record every state before a move.
pub fn states(settings: &Arc<Settings>, games: &[String], strategies: &[String], agent: &str) -> Result<Vec<State>> {
    if games.is_empty() || strategies.is_empty() {
        return Err(Error::Usage("training data needs at least one game and one opponent strategy".to_owned()));
    }
    let episodes = settings.section("training")?.count("episodes")?;
    let rounds_shown = settings.section("agent")?.whole("history_rounds")? as usize;
    let library = Arc::new(Library::standard());
    let mut environment = Environment::new(library.clone(), settings.clone())?;
    let seed = environment.seed();
    let mut found = Vec::new();
    for key in games {
        let game = library.build(key, settings)?;
        for strategy in strategies {
            for _ in std::iter::repeat(()).take(episodes) {
                let mut seat = StrategyAgent::new(agent, game.clone(), settings, seed)?;
                let request = Reset { game: key.clone(), strategy: Some(strategy.clone()), rounds: None, episode_id: None };
                let mut observation = environment.reset(&request, None)?;
                while !observation.done {
                    found.push(State {
                        prompt: prompt::build(&observation, rounds_shown),
                        game: key.clone(),
                        history: observation
                            .history
                            .iter()
                            .map(|round| (round.player_action.clone(), round.opponent_action.clone()))
                            .collect(),
                        seed,
                    });
                    observation = environment.step(&seat.act(&observation)?)?;
                }
            }
        }
    }
    Ok(found)
}

fn write(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(|source| Error::Json { origin: path.display().to_string(), source })?;
    std::fs::write(path, text).map_err(|source| Error::Io { path: path.to_path_buf(), source })
}

/// Write `prompts.json`, `states.json` and `pairs.json` into `directory`;
/// the answer counts what each holds.
pub fn write_dataset(settings: &Arc<Settings>, found: &[State], directory: &Path) -> Result<Value> {
    std::fs::create_dir_all(directory).map_err(|source| Error::Io { path: directory.to_path_buf(), source })?;
    let margin = settings.section("training")?.number("pair_margin")?;
    let library = Library::standard();
    let mut pairs = Vec::new();
    for state in found {
        let ranked = reward::ranked(&library, settings, state)?;
        if let (Some((best_move, best)), Some((worst_move, worst))) = (ranked.first(), ranked.last()) {
            if best - worst >= margin {
                pairs.push(json!({
                    "positive": format!("{}\n{best_move}", state.prompt),
                    "negative": format!("{}\n{worst_move}", state.prompt),
                }));
            }
        }
    }
    let prompts: Vec<&str> = found.iter().map(|state| state.prompt.as_str()).collect();
    write(&directory.join("prompts.json"), &json!({ "prompts": prompts }))?;
    write(&directory.join("states.json"), &json!({ "states": found }))?;
    write(&directory.join("pairs.json"), &json!({ "trait_name": "kantbench_payoff", "pairs": pairs }))?;
    Ok(json!({
        "directory": directory,
        "prompts": prompts.len(),
        "states": found.len(),
        "pairs": pairs.len(),
        "settings": settings.document(),
    }))
}

/// The states a `states.json` holds.
pub fn read_states(path: &Path) -> Result<Vec<State>> {
    #[derive(Deserialize)]
    struct File {
        states: Vec<State>,
    }
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
    let file: File = serde_json::from_str(&text).map_err(|source| Error::Json { origin: path.display().to_string(), source })?;
    Ok(file.states)
}
