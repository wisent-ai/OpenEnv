//! The two-seat environment: reset to a game and an opponent, then step one
//! agent move at a time. The opponent's move is played inside `step`, by a
//! library strategy or by another agent. A free-chat game splits each round
//! into a message step and an action step (`free_chat`).

mod episode;
mod free_chat;
pub mod models;

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;

pub use episode::Opponent;
pub use models::{round_after, GameAction, GameObservation, GameState, RoundResult};

use crate::error::{Error, Result};
use crate::game::Library;
use crate::settings::Settings;
use crate::strategy;

use episode::Episode;

/// Another model holding the opponent's seat: it sees the episode from its
/// own side and answers with an action.
pub trait Agent: Send {
    fn act(&mut self, observation: &GameObservation) -> Result<GameAction>;
}

/// What to play next: a game from the library, and the opponent strategy
/// when no agent holds the opponent's seat.
#[derive(Clone, Debug, Default)]
pub struct Reset {
    pub game: String,
    pub strategy: Option<String>,
    /// Rounds for this episode instead of the game's declared `rounds`.
    pub rounds: Option<usize>,
    pub episode_id: Option<String>,
}

pub struct Environment {
    library: Arc<Library>,
    settings: Arc<Settings>,
    seed: u64,
    rng: StdRng,
    episode: Option<Episode>,
}

impl Environment {
    /// An environment over `library` whose numbers come from `settings`. Its
    /// random stream starts from the declared seed, or from one drawn from
    /// the operating system, which `seed()` reports so a run can record it.
    pub fn new(library: Arc<Library>, settings: Arc<Settings>) -> Result<Self> {
        let seed = match settings.seed()? {
            Some(seed) => seed,
            None => rand::random(),
        };
        Ok(Self {
            library,
            settings,
            seed,
            rng: StdRng::seed_from_u64(seed),
            episode: None,
        })
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Start an episode. `agent` holds the opponent's seat when given;
    /// otherwise `request.strategy` names the library strategy that does.
    pub fn reset(&mut self, request: &Reset, agent: Option<Box<dyn Agent>>) -> Result<GameObservation> {
        let game = self.library.build(&request.game, &self.settings)?;
        let opponent = match (agent, &request.strategy) {
            (Some(agent), _) => Opponent::Agent(agent),
            (None, Some(name)) => Opponent::Strategy {
                name: name.clone(),
                strategy: strategy::named(name, &self.settings)?,
            },
            (None, None) => {
                return Err(Error::Usage(
                    "an episode needs an opponent: name a strategy or give an agent for the opponent's seat".to_owned(),
                ))
            }
        };
        let rounds = match request.rounds {
            Some(rounds) => rounds,
            None => game.rounds,
        };
        let episode_id = match &request.episode_id {
            Some(id) => id.clone(),
            None => uuid::Uuid::new_v4().to_string(),
        };
        let episode = Episode::start(game, opponent, &request.game, episode_id, rounds);
        let observation = episode.observation(Default::default(), None);
        self.episode = Some(episode);
        Ok(observation)
    }

    /// Play the agent's action; the opponent answers inside this step.
    pub fn step(&mut self, action: &GameAction) -> Result<GameObservation> {
        let episode = self.episode.as_mut().ok_or(Error::NotStarted)?;
        if episode.state.is_done {
            return Err(Error::Finished {
                episode: episode.state.episode_id.clone(),
                rounds: episode.state.history.len(),
            });
        }
        if episode.game.has_variant("free_chat") {
            return free_chat::step(episode, action, &mut self.rng);
        }
        episode.check_move(&action.action)?;
        let (opponent_action, opponent_message) = episode.opponent_move(&action.action, None, &mut self.rng)?;
        let result = episode.settle(&action.action, &opponent_action, action.message(), opponent_message, &mut self.rng)?;
        Ok(episode.after_round(result))
    }

    /// The running episode's state; none before the first reset.
    pub fn state(&self) -> Option<&GameState> {
        self.episode.as_ref().map(|episode| &episode.state)
    }
}
