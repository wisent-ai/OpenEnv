//! Generated normal-form games: a square game whose cells are drawn from the
//! declared range by a declared seed, so the same declaration always yields
//! the same game. Moves are named `a`, `b`, … `z`, then `aa`, `ab`, ….

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::game::{matrix_payoff, Entry, Game, Library, Matrix};
use crate::settings::Declared;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new(
        "random_symmetric_3x3",
        "generated",
        &["moves", "lowest", "highest", "draw_seed"],
        |declared| generated(declared, Shape::Symmetric),
    ));
    library.add(Entry::new(
        "random_asymmetric_3x3",
        "generated",
        &["moves", "lowest", "highest", "draw_seed"],
        |declared| generated(declared, Shape::Independent),
    ));
    library.add(Entry::new(
        "random_zero_sum_3x3",
        "generated",
        &["moves", "lowest", "highest", "draw_seed"],
        |declared| generated(declared, Shape::ZeroSum),
    ));
    library.add(Entry::new(
        "random_coordination_3x3",
        "generated",
        &["moves", "lowest", "highest", "draw_seed", "bonus"],
        |declared| generated(declared, Shape::Coordination(declared.number("bonus")?)),
    ));
}

#[derive(Clone, Copy)]
enum Shape {
    /// The first seat's payoff for `(a, b)` is the second seat's for `(b, a)`.
    Symmetric,
    /// Every cell is drawn on its own.
    Independent,
    /// One draw per cell; the opponent gets its negation.
    ZeroSum,
    /// One draw per cell paid to both seats, plus the declared bonus when the
    /// seats choose alike.
    Coordination(f64),
}

/// The first `count` move names: single letters, then letter pairs.
pub(crate) fn labels(count: usize) -> Vec<String> {
    let alphabet: Vec<char> = ('a'..='z').collect();
    let singles = alphabet.iter().map(|letter| letter.to_string());
    let pairs = alphabet
        .iter()
        .flat_map(|first| alphabet.iter().map(move |second| format!("{first}{second}")));
    singles.chain(pairs).take(count).collect()
}

fn generated(declared: &Declared<'_>, shape: Shape) -> Result<Game> {
    let count = declared.count("moves")?;
    let lowest = declared.integer("lowest")?;
    let highest = declared.integer("highest")?;
    if lowest > highest {
        return Err(Error::Malformed {
            scope: declared.scope().to_owned(),
            name: "highest".to_owned(),
            expected: format!("at least lowest ({lowest})"),
            found: highest.to_string(),
        });
    }
    let seed = declared.whole("draw_seed")?;
    let mut rng = StdRng::seed_from_u64(seed);
    let actions = labels(count);
    let mut cells = BTreeMap::new();
    for row in &actions {
        for column in &actions {
            let key = (row.clone(), column.clone());
            if cells.contains_key(&key) {
                continue;
            }
            let first = rng.gen_range(lowest..=highest) as f64;
            let cell = match shape {
                Shape::Symmetric => {
                    let second = rng.gen_range(lowest..=highest) as f64;
                    cells.insert((column.clone(), row.clone()), (second, first));
                    (first, second)
                }
                Shape::Independent => (first, rng.gen_range(lowest..=highest) as f64),
                Shape::ZeroSum => (first, -first),
                Shape::Coordination(bonus) if row == column => (first + bonus, first + bonus),
                Shape::Coordination(_) => (first, first),
            };
            cells.insert(key, cell);
        }
    }
    let (title, kind) = match shape {
        Shape::Symmetric => ("Symmetric", "symmetric matrix game with payoffs drawn"),
        Shape::Independent => ("Asymmetric", "asymmetric matrix game with independent payoffs drawn"),
        Shape::ZeroSum => ("Zero-Sum", "zero-sum game, every outcome summing to zero, with the row payoff drawn"),
        Shape::Coordination(_) => ("Coordination", "coordination game, matching moves earning a bonus, with payoffs drawn"),
    };
    let name = format!("Random {title} {count}x{count} (seed={seed})");
    let description = format!(
        "A randomly generated {count}x{count} {kind} in [{lowest}, {highest}]. Tests generalization to novel strategic structures."
    );
    let payoff = matrix_payoff(&name, Matrix::from_cells(cells));
    Ok(Game::new(&name, &description, "matrix", actions, payoff))
}
