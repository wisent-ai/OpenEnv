//! What the explorer shows besides play: a game as the settings document
//! builds it (its moves and payoff cells), and a tournament run on request,
//! the same `kant tournament` runs.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::error::Error;
use crate::evaluation::{Runner, SeatSpec};

use super::{refused, Shared};

/// A two-seat game as built: its summary, and every cell its moves pay, as
/// rows of the agent's moves against the opponent's.
pub(super) async fn game(State(shared): State<Shared>, Path(key): Path<String>) -> Response {
    let built = match shared.pairs.build(&key, &shared.settings) {
        Ok(built) => built,
        Err(error) => return refused(StatusCode::BAD_REQUEST, &error),
    };
    let mut rng = rand::thread_rng();
    let mut cells = Vec::new();
    for mine in &built.actions {
        let mut row = Vec::new();
        for theirs in built.opponent_moves() {
            // A fresh build per cell, so a game whose payoffs move with the
            // episode shows its opening cells rather than a sequence's end.
            let fresh = match shared.pairs.build(&key, &shared.settings) {
                Ok(fresh) => fresh,
                Err(error) => return refused(StatusCode::BAD_REQUEST, &error),
            };
            row.push(match fresh.pay(mine, theirs, &mut rng) {
                Ok((paid, answered)) => json!([paid, answered]),
                Err(error) => json!({ "refused": error.to_string() }),
            });
        }
        cells.push(row);
    }
    Json(json!({ "game": built.summary(), "rows": built.actions, "columns": built.opponent_moves(), "cells": cells })).into_response()
}

#[derive(Deserialize)]
pub(super) struct TournamentRequest {
    games: Vec<String>,
    strategies: Vec<String>,
    #[serde(default)]
    agent_strategy: Option<String>,
    #[serde(default)]
    agent_route: Option<String>,
    #[serde(default)]
    opponent_route: Option<String>,
}

/// A tournament on request; it runs off the request threads, since a model
/// seat waits on Brama.
pub(super) async fn tournament(State(shared): State<Shared>, Json(request): Json<TournamentRequest>) -> Response {
    let agent = match (request.agent_route, request.agent_strategy) {
        (Some(route), None) => SeatSpec::Model(route),
        (None, Some(name)) => SeatSpec::Strategy(name),
        _ => {
            let error = Error::Usage("name the agent's seat with exactly one of agent_route or agent_strategy".to_owned());
            return refused(StatusCode::BAD_REQUEST, &error);
        }
    };
    let settings = shared.settings.clone();
    let ran = tokio::task::spawn_blocking(move || {
        Runner::new(settings, agent, request.opponent_route)?.run(&request.games, &request.strategies)
    })
    .await;
    match ran {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(error)) => refused(StatusCode::BAD_REQUEST, &error),
        Err(join) => refused(StatusCode::INTERNAL_SERVER_ERROR, &Error::Usage(format!("the tournament stopped: {join}"))),
    }
}

