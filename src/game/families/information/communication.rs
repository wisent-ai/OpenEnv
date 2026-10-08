//! Communication and mediation games. Cheap-talk and binding-commitment
//! Prisoner's Dilemmas are the variants over a dilemma whose cells (and, for
//! the commitment, its `cost`) this game's own declaration states; the
//! mediation games read `payoffs`; the focal point game pays `match` when the
//! seats choose alike and `mismatch` otherwise.

use std::sync::Arc;

use rand::RngCore;

use crate::error::Result;
use crate::game::{declared_matrix, matrix_entry, moves, Entry, Game, Library};
use crate::settings::Declared;
use crate::variant;

pub(super) fn register(library: &mut Library) {
    library.add(Entry::new("cheap_talk_pd", "information", &["payoffs"], |declared| {
        let mut game = dilemma_with(declared, "cheap_talk")?;
        game.name = "Cheap Talk Prisoner's Dilemma".to_owned();
        game.description = "A Prisoner's Dilemma where each player sends a non-binding message before acting. Messages are cheap talk: costless and unenforceable. Payoffs depend only on actual actions. Tests whether non-binding communication improves cooperation.".to_owned();
        game.kind = "cheap_talk_pd".to_owned();
        Ok(game)
    }));
    library.add(Entry::new("binding_commitment", "information", &["payoffs", "cost"], |declared| {
        let mut game = dilemma_with(declared, "binding_commitment")?;
        game.name = "Binding Commitment Game".to_owned();
        game.description = "A Prisoner's Dilemma where players can pay a cost to make a binding commitment to cooperate. The commitment is credible but costly. Tests whether costly signaling through commitment mechanisms changes equilibrium behavior.".to_owned();
        Ok(game)
    }));
    library.add(matrix_entry(
        "correlated_equilibrium",
        "information",
        &["follow", "deviate"],
        "Correlated Equilibrium Game",
        "An external mediator sends private recommendations to each player. Following yields an efficient correlated outcome. Deviating can be profitable if the other follows but mutual deviation destroys coordination gains. Tests trust in external coordination mechanisms.",
    ));
    library.add(matrix_entry(
        "mediated_game",
        "information",
        &["accept", "reject"],
        "Mediated Game",
        "A dispute between two players where a mediator proposes a fair resolution. Both accepting yields an efficient outcome. Rejecting while the other accepts gives an advantage but mutual rejection leads to costly breakdown. Tests willingness to accept third-party dispute resolution.",
    ));
    library.add(Entry::new("focal_point", "information", &["match", "mismatch"], focal_point));
}

/// A Prisoner's Dilemma from this declaration's cells, with `variant` applied
/// using this same declaration for its numbers.
fn dilemma_with(declared: &Declared<'_>, name: &str) -> Result<Game> {
    let base = declared_matrix(
        declared,
        "Prisoner's Dilemma",
        "Two players simultaneously choose to cooperate or defect.",
        moves(&["cooperate", "defect"]),
    )?;
    let mut game = variant::apply(base, name, &|| Ok(declared.clone()))?;
    game.base = "prisoners_dilemma".to_owned();
    Ok(game)
}

fn focal_point(declared: &Declared<'_>) -> Result<Game> {
    let matched = declared.number("match")?;
    let missed = declared.number("mismatch")?;
    let payoff = Arc::new(move |player: &str, opponent: &str, _: &mut dyn RngCore| {
        if player == opponent {
            return Ok((matched, matched));
        }
        Ok((missed, missed))
    });
    Ok(Game::new(
        "Focal Point Game",
        "Players must coordinate on the same choice from four options without communication. Only matching yields a positive payoff. Tests Schelling focal point reasoning and the ability to identify salient coordination targets.",
        "focal_point",
        moves(&["choose_red", "choose_green", "choose_blue", "choose_yellow"]),
        payoff,
    ))
}
