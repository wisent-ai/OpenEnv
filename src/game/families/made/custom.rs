//! Games a run declares whole, under `custom_games.<key>`: their name, moves
//! and cells, with no entry in the library. Two shapes are read:
//!
//! - `payoffs`: every cell `[agent, opponent]`, as a library matrix game reads;
//! - `symmetric`: one number per cell, the row seat's payoff, with the column
//!   seat's payoff for `(a, b)` taken from the cell `(b, a)`.
//!
//! The moves must be distinct; a missing cell is refused by its row and
//! column.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};
use crate::game::{matrix_payoff, Game, Matrix};
use crate::settings::Declared;

pub(crate) fn build(key: &str, declared: &Declared<'_>) -> Result<Game> {
    let actions = declared.texts("actions")?;
    let distinct: BTreeSet<&String> = actions.iter().collect();
    if actions.is_empty() || distinct.len() != actions.len() {
        return Err(Error::Malformed {
            scope: declared.scope().to_owned(),
            name: "actions".to_owned(),
            expected: "one or more distinct moves".to_owned(),
            found: actions.join(", "),
        });
    }
    let name = declared.text("name")?;
    let description = declared.text("description")?;
    let matrix = match (declared.has("payoffs"), declared.has("symmetric")) {
        (true, false) => Matrix::declared(declared, &actions, &actions)?,
        (false, true) => symmetric(declared, &actions)?,
        _ => {
            return Err(Error::Malformed {
                scope: declared.scope().to_owned(),
                name: "payoffs".to_owned(),
                expected: "exactly one of payoffs or symmetric".to_owned(),
                found: format!("payoffs: {}, symmetric: {}", declared.has("payoffs"), declared.has("symmetric")),
            })
        }
    };
    let mut game = Game::new(name, description, "matrix", actions, matrix_payoff(key, matrix));
    game.base = key.to_owned();
    Ok(game)
}

fn symmetric(declared: &Declared<'_>, actions: &[String]) -> Result<Matrix> {
    let table = declared.nested("symmetric")?;
    let mut own = BTreeMap::new();
    for row in actions {
        let line = table.nested(row)?;
        for column in actions {
            own.insert((row.clone(), column.clone()), line.number(column)?);
        }
    }
    let mut cells = BTreeMap::new();
    for ((row, column), mine) in &own {
        let theirs = own[&(column.clone(), row.clone())];
        cells.insert((row.clone(), column.clone()), (*mine, theirs));
    }
    Ok(Matrix::from_cells(cells))
}
