//! The settings document a run declares: every number that decides how a game
//! pays, how an opponent plays, how many rounds an episode lasts or how many
//! episodes an evaluation runs. KantBench carries none of these numbers itself;
//! a run names its document (`--settings FILE`) and its results record the
//! document whole, so a reported score always travels with the numbers that
//! produced it.
//!
//! The document is one JSON object whose sections are objects keyed by what
//! they configure: `games`, `strategies`, `variants`, and one section per area
//! (`evaluation`, `arena`, `training`). A value a game, strategy or variant
//! needs and the document does not declare is refused by name
//! (`games.ultimatum declares no pot`), never replaced.

mod declared;

use std::path::Path;

use serde_json::{Map, Value};

pub use declared::Declared;

use crate::error::{Error, Result};

/// One parsed settings document and where it came from.
#[derive(Clone, Debug)]
pub struct Settings {
    origin: String,
    document: Map<String, Value>,
}

impl Settings {
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text, &path.display().to_string())
    }

    pub fn parse(text: &str, origin: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text).map_err(|source| Error::Json {
            origin: origin.to_owned(),
            source,
        })?;
        let Value::Object(document) = value else {
            return Err(Error::NotAnObject {
                origin: origin.to_owned(),
            });
        };
        Ok(Self {
            origin: origin.to_owned(),
            document,
        })
    }

    /// The path or label the document was read from.
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// The document as declared, for a result to record whole.
    pub fn document(&self) -> &Map<String, Value> {
        &self.document
    }

    /// The seed the document declares, if it declares one. A run without one
    /// draws its seed from the operating system and records the drawn value.
    pub fn seed(&self) -> Result<Option<u64>> {
        match self.document.get("seed") {
            None => Ok(None),
            Some(value) => value.as_u64().map(Some).ok_or_else(|| Error::Malformed {
                scope: self.origin.clone(),
                name: "seed".to_owned(),
                expected: "a whole number".to_owned(),
                found: value.to_string(),
            }),
        }
    }

    /// A top-level section that is itself the settings of one area.
    pub fn section(&self, name: &str) -> Result<Declared<'_>> {
        declared::object(&self.document, &self.origin, name)
            .map(|values| Declared::over(name, values))
    }

    /// One entry of a section: `games.prisoners_dilemma`,
    /// `strategies.generous_tit_for_tat`, `variants.exit`.
    pub fn entry(&self, section: &str, key: &str) -> Result<Declared<'_>> {
        let values = declared::object(&self.document, &self.origin, section)?;
        declared::object(values, section, key)
            .map(|values| Declared::over(&format!("{section}.{key}"), values))
    }

    /// Whether a section declares an entry, without refusing when it does not.
    pub fn declares(&self, section: &str, key: &str) -> bool {
        self.document
            .get(section)
            .and_then(Value::as_object)
            .is_some_and(|values| values.contains_key(key))
    }
}
