//! The `kant` command line. Each command module reads its words and answers
//! one JSON document on standard output; a refusal is one line on standard
//! error naming what is missing, with a failing exit status.

pub mod catalog;
pub mod play;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::settings::Settings;

/// The words after the command: positionals, `--name value` pairs and
/// switches. An option followed by nothing or by another option is a switch.
pub struct Words {
    pub positionals: Vec<String>,
    values: BTreeMap<String, Vec<String>>,
    switches: BTreeSet<String>,
}

impl Words {
    pub fn parse(args: &[String]) -> Self {
        let mut parsed = Self {
            positionals: Vec::new(),
            values: BTreeMap::new(),
            switches: BTreeSet::new(),
        };
        let mut rest = args.iter().peekable();
        while let Some(word) = rest.next() {
            let Some(name) = word.strip_prefix("--") else {
                parsed.positionals.push(word.clone());
                continue;
            };
            match rest.next_if(|next| !next.starts_with("--")) {
                Some(value) => parsed.values.entry(name.to_owned()).or_default().push(value.clone()),
                None => {
                    parsed.switches.insert(name.to_owned());
                }
            }
        }
        parsed
    }

    pub fn one(&self, name: &str) -> Result<Option<&str>> {
        if self.switches.contains(name) {
            return Err(Error::Usage(format!("--{name} needs a value")));
        }
        match self.values.get(name).map(Vec::as_slice) {
            None => Ok(None),
            Some([value]) => Ok(Some(value.as_str())),
            Some(_) => Err(Error::Usage(format!("--{name} is given more than once"))),
        }
    }

    pub fn required(&self, name: &str) -> Result<&str> {
        self.one(name)?
            .ok_or_else(|| Error::Usage(format!("--{name} is required")))
    }

    pub fn all(&self, name: &str) -> Vec<&str> {
        match self.values.get(name) {
            Some(values) => values.iter().map(String::as_str).collect(),
            None => Vec::new(),
        }
    }

    pub fn switch(&self, name: &str) -> bool {
        self.switches.contains(name)
    }

    /// A whole-number option above zero, refused by name when malformed.
    pub fn count(&self, name: &str) -> Result<Option<usize>> {
        match self.one(name)? {
            None => Ok(None),
            Some(text) => text
                .parse::<std::num::NonZeroUsize>()
                .map(|count| Some(count.get()))
                .map_err(|_| Error::Usage(format!("--{name} must be a whole number above zero, not {text}"))),
        }
    }

    /// The settings document `--settings FILE` names.
    pub fn settings(&self) -> Result<Arc<Settings>> {
        let path = PathBuf::from(self.required("settings")?);
        Ok(Arc::new(Settings::read(&path)?))
    }
}
