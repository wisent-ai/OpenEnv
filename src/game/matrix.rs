//! A normal-form game's payoffs, declared cell by cell:
//!
//! ```json
//! "payoffs": {
//!   "cooperate": { "cooperate": [3, 3], "defect": [0, 5] },
//!   "defect":    { "cooperate": [5, 0], "defect": [1, 1] }
//! }
//! ```
//!
//! Each row is the agent's move, each column the opponent's, and each cell the
//! pair `[agent, opponent]`. Every cell of the game's moves must be declared;
//! a missing one is refused by its row and column when the game is built, so
//! no pair of moves pays nothing by omission.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::settings::Declared;

use super::{Entry, Game, Payoff};

/// Moves from their names.
pub fn moves(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// A normal-form game whose every cell the declaration states.
pub fn declared_matrix(
    declared: &Declared<'_>,
    name: &str,
    description: &str,
    actions: Vec<String>,
) -> Result<Game> {
    let matrix = Matrix::declared(declared, &actions, &actions)?;
    Ok(Game::new(name, description, "matrix", actions, matrix_payoff(name, matrix)))
}

/// A library entry for a normal-form game: it reads `payoffs` and nothing
/// else. The moves come first so the cooperative one opens the list.
pub fn matrix_entry(
    key: &'static str,
    family: &'static str,
    actions: &'static [&'static str],
    name: &'static str,
    description: &'static str,
) -> Entry {
    Entry::new(key, family, &["payoffs"], move |declared| {
        declared_matrix(declared, name, description, moves(actions))
    })
}

/// A library entry for a normal-form game whose seats move differently: the
/// agent picks a row from `rows`, the opponent a column from `columns` (a
/// sender signals, a receiver responds). It reads `payoffs` cell by cell.
pub fn matrix_between(
    key: &'static str,
    family: &'static str,
    rows: &'static [&'static str],
    columns: &'static [&'static str],
    name: &'static str,
    description: &'static str,
) -> Entry {
    Entry::new(key, family, &["payoffs"], move |declared| {
        let (rows, columns) = (moves(rows), moves(columns));
        let matrix = Matrix::declared(declared, &rows, &columns)?;
        let mut game = Game::new(name, description, "matrix", rows, matrix_payoff(name, matrix));
        game.opponent_actions = super::OpponentMoves::Own(columns);
        Ok(game)
    })
}

/// The declared payoff of every pair of moves.
#[derive(Clone, Debug, Default)]
pub struct Matrix {
    cells: BTreeMap<(String, String), (f64, f64)>,
}

impl Matrix {
    /// Read `payoffs` from a declaration for every row in `rows` against
    /// every column in `columns`.
    pub fn declared(declared: &Declared<'_>, rows: &[String], columns: &[String]) -> Result<Self> {
        let payoffs = declared.nested("payoffs")?;
        let mut cells = BTreeMap::new();
        for row in rows {
            let line = payoffs.nested(row)?;
            for column in columns {
                let pair = line.numbers(column)?;
                let [player, opponent] = pair.as_slice() else {
                    return Err(Error::Malformed {
                        scope: line.scope().to_owned(),
                        name: column.clone(),
                        expected: "a pair [agent, opponent]".to_owned(),
                        found: format!("{pair:?}"),
                    });
                };
                cells.insert((row.clone(), column.clone()), (*player, *opponent));
            }
        }
        Ok(Self { cells })
    }

    /// A matrix built from cells a caller computed, such as a generated game.
    pub fn from_cells(cells: BTreeMap<(String, String), (f64, f64)>) -> Self {
        Self { cells }
    }

    pub fn cell(&self, game: &str, player: &str, opponent: &str) -> Result<(f64, f64)> {
        self.cells
            .get(&(player.to_owned(), opponent.to_owned()))
            .copied()
            .ok_or_else(|| Error::NoPayoff {
                game: game.to_owned(),
                player: player.to_owned(),
                opponent: opponent.to_owned(),
            })
    }

    pub fn cells(&self) -> &BTreeMap<(String, String), (f64, f64)> {
        &self.cells
    }
}

/// A payoff that reads a matrix; a pair outside it is refused by name.
pub fn matrix_payoff(game: &str, matrix: Matrix) -> Payoff {
    let game = game.to_owned();
    Arc::new(move |player, opponent, _| matrix.cell(&game, player, opponent))
}
