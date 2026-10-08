//! `kant serve --settings FILE --listen ADDRESS [--states FILE]`: the
//! environment server. The address is the caller's; how many WebSocket
//! sessions may be open at once is the settings document's
//! `server.sessions`. With `--states` (the `states.json` `kant dataset`
//! wrote) it also answers `/reward`, Ster's outside scorer for that dataset.

use std::path::Path;

use serde_json::Value;

use crate::cli::Words;
use crate::error::{Error, Result};
use crate::training::{self, reward::Scorer};

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let listen = words.required("listen")?;
    let sessions = settings.section("server")?.count("sessions")?;
    let scorer = match words.one("states")? {
        Some(path) => Some(Scorer::new(settings.clone(), training::read_states(Path::new(path))?)?),
        None => None,
    };
    crate::server::serve(listen, settings, sessions, scorer)?;
    Err(Error::Usage("the server stopped".to_owned()))
}
