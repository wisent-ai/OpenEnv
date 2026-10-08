//! `kant serve --settings FILE --listen ADDRESS`: the environment server. The
//! address is the caller's; how many WebSocket sessions may be open at once
//! is the settings document's `server.sessions`. Nothing is assumed for
//! either.

use serde_json::Value;

use crate::error::{Error, Result};

use super::Words;

pub fn run(words: &Words) -> Result<Value> {
    let settings = words.settings()?;
    let listen = words.required("listen")?;
    let sessions = settings.section("server")?.count("sessions")?;
    crate::server::serve(listen, settings, sessions)?;
    Err(Error::Usage("the server stopped".to_owned()))
}
