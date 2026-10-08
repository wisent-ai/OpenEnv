//! One declared object read by name, each read refusing a missing or
//! malformed value under the scope it was declared in.

use std::num::NonZeroUsize;

use rand::distributions::Bernoulli;
use serde_json::{Map, Value};

use crate::error::{Error, Result};

pub(super) fn object<'a>(
    values: &'a Map<String, Value>,
    scope: &str,
    name: &str,
) -> Result<&'a Map<String, Value>> {
    match values.get(name) {
        None => Err(Error::Undeclared {
            scope: scope.to_owned(),
            name: name.to_owned(),
        }),
        Some(Value::Object(inner)) => Ok(inner),
        Some(other) => Err(Error::Malformed {
            scope: scope.to_owned(),
            name: name.to_owned(),
            expected: "an object".to_owned(),
            found: other.to_string(),
        }),
    }
}

/// The declared values of one game, strategy, variant or area.
#[derive(Clone, Debug)]
pub struct Declared<'a> {
    scope: String,
    values: &'a Map<String, Value>,
}

impl<'a> Declared<'a> {
    /// A declaration over an object, under the scope its refusals name. A
    /// request body read the same way names its own scope.
    pub fn over(scope: &str, values: &'a Map<String, Value>) -> Self {
        Self {
            scope: scope.to_owned(),
            values,
        }
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    pub fn values(&self) -> &'a Map<String, Value> {
        self.values
    }

    pub fn has(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    fn value(&self, name: &str) -> Result<&'a Value> {
        self.values.get(name).ok_or_else(|| Error::Undeclared {
            scope: self.scope.clone(),
            name: name.to_owned(),
        })
    }

    fn malformed(&self, name: &str, expected: &str, found: &Value) -> Error {
        Error::Malformed {
            scope: self.scope.clone(),
            name: name.to_owned(),
            expected: expected.to_owned(),
            found: found.to_string(),
        }
    }

    /// Any finite number.
    pub fn number(&self, name: &str) -> Result<f64> {
        let value = self.value(name)?;
        value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| self.malformed(name, "a finite number", value))
    }

    /// A whole number of zero or more: an amount, a pot, an endowment.
    pub fn whole(&self, name: &str) -> Result<u64> {
        let value = self.value(name)?;
        value
            .as_u64()
            .ok_or_else(|| self.malformed(name, "a whole number", value))
    }

    /// A whole number that may be negative: a penalty, a signed payoff.
    pub fn integer(&self, name: &str) -> Result<i64> {
        let value = self.value(name)?;
        value
            .as_i64()
            .ok_or_else(|| self.malformed(name, "a whole number", value))
    }

    /// A count that must be at least one: rounds, players, episodes.
    pub fn count(&self, name: &str) -> Result<usize> {
        let value = self.value(name)?;
        value
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .and_then(NonZeroUsize::new)
            .map(NonZeroUsize::get)
            .ok_or_else(|| self.malformed(name, "a whole number above zero", value))
    }

    /// A probability from zero to one, as a draw that can be taken.
    pub fn probability(&self, name: &str) -> Result<Bernoulli> {
        let value = self.value(name)?;
        value
            .as_f64()
            .and_then(|chance| Bernoulli::new(chance).ok())
            .ok_or_else(|| self.malformed(name, "a probability from zero to one", value))
    }

    pub fn text(&self, name: &str) -> Result<&'a str> {
        let value = self.value(name)?;
        value
            .as_str()
            .ok_or_else(|| self.malformed(name, "text", value))
    }

    pub fn flag(&self, name: &str) -> Result<bool> {
        let value = self.value(name)?;
        value
            .as_bool()
            .ok_or_else(|| self.malformed(name, "true or false", value))
    }

    /// A list of finite numbers.
    pub fn numbers(&self, name: &str) -> Result<Vec<f64>> {
        let value = self.value(name)?;
        let list = value
            .as_array()
            .ok_or_else(|| self.malformed(name, "a list of numbers", value))?;
        list.iter()
            .map(|item| {
                item.as_f64()
                    .filter(|number| number.is_finite())
                    .ok_or_else(|| self.malformed(name, "a list of finite numbers", value))
            })
            .collect()
    }

    /// A list of text values, such as a declared game's moves.
    pub fn texts(&self, name: &str) -> Result<Vec<String>> {
        let value = self.value(name)?;
        let list = value
            .as_array()
            .ok_or_else(|| self.malformed(name, "a list of text", value))?;
        list.iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| self.malformed(name, "a list of text", value))
            })
            .collect()
    }

    /// A nested object, read with the same refusals under a longer scope.
    pub fn nested(&self, name: &str) -> Result<Declared<'a>> {
        let value = self.value(name)?;
        let values = value
            .as_object()
            .ok_or_else(|| self.malformed(name, "an object", value))?;
        Ok(Declared {
            scope: format!("{}.{name}", self.scope),
            values,
        })
    }
}
