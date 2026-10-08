//! `kant serve`: the KantBench environment over OpenEnv's HTTP and WebSocket
//! protocol. A WebSocket on `/ws` is one persistent session (`reset`, `step`,
//! `state`, `close` messages); `/reset`, `/step` and `/state` answer over a
//! session that lives for one request, as OpenEnv's simulation mode does;
//! `/web` is the interactive explorer, with `/game/<key>` (a game's payoff
//! cells) and `/tournament` (a tournament on request) behind it; `/health`,
//! `/metadata` and `/schema` describe the server. The number of open sessions
//! is the settings document's.

mod session;
mod socket;
pub mod space;
mod views;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::http::header;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::error::{Error, Result};
use crate::game::Library;
use crate::group::GroupLibrary;
use crate::training::reward::Scorer;
use crate::settings::Settings;

pub use session::{ResetRequest, Session, StepResult};
pub use space::{KantBenchAction, KantBenchObservation};

const EXPLORER: &str = include_str!("explorer/page.html");
const EXPLORER_SCRIPT: &str = include_str!("explorer/script.js");
const EXPLORER_STYLE: &str = include_str!("explorer/style.css");

#[derive(Clone)]
pub(crate) struct Shared {
    pairs: Arc<Library>,
    groups: Arc<GroupLibrary>,
    settings: Arc<Settings>,
    /// One permit per session that may be open at once.
    places: Arc<Semaphore>,
    /// Rewards for answers to a training dataset's prompts.
    scorer: Option<Arc<Scorer>>,
}

impl Shared {
    fn session(&self) -> Session {
        Session::new(self.pairs.clone(), self.groups.clone(), self.settings.clone())
    }
}

/// A refusal as an HTTP answer: what was refused, in FastAPI's `detail`.
pub(crate) fn refused(status: StatusCode, error: &Error) -> Response {
    (status, Json(json!({ "detail": error.to_string() }))).into_response()
}

fn router(shared: Shared) -> Router {
    Router::new()
        .route("/", get(|| async { Redirect::temporary("/web") }))
        .route("/web", get(|| async { Html(EXPLORER) }))
        .route("/explorer.js", get(|| async { ([(header::CONTENT_TYPE, "text/javascript")], EXPLORER_SCRIPT) }))
        .route("/explorer.css", get(|| async { ([(header::CONTENT_TYPE, "text/css")], EXPLORER_STYLE) }))
        .route("/health", get(|| async { Json(json!({ "status": "healthy" })) }))
        .route("/metadata", get(metadata))
        .route("/schema", get(schema))
        .route("/games", get(games))
        .route("/reset", post(reset))
        .route("/step", post(step))
        .route("/state", get(state))
        .route("/ws", get(socket::upgrade))
        .route("/reward", post(reward))
        .route("/game/:key", get(views::game))
        .route("/tournament", post(views::tournament))
        .with_state(shared)
}

async fn metadata() -> Json<Value> {
    Json(json!({
        "name": "KantBench",
        "description": "Game-theory environments for language-model agents: two-seat and group games, opponent strategies and composable variants.",
        "version": env!("CARGO_PKG_VERSION"),
        "documentation_url": env!("CARGO_PKG_REPOSITORY"),
    }))
}

async fn schema() -> Json<Value> {
    Json(json!({
        "action": schemars::schema_for!(KantBenchAction),
        "observation": schemars::schema_for!(KantBenchObservation),
        "reset": schemars::schema_for!(ResetRequest),
    }))
}

/// Every game key a reset may name, two-seat and group, for the explorer.
async fn games(State(shared): State<Shared>) -> Json<Value> {
    let pairs: Vec<&str> = shared.pairs.entries().map(|entry| entry.key).collect();
    let groups: Vec<&str> = shared.groups.entries().map(|entry| entry.key).collect();
    Json(json!({
        "pair": pairs,
        "group": groups,
        "strategies": crate::strategy::NAMES,
        "group_strategies": crate::group::strategies::NAMES,
        "variants": crate::variant::NAMES,
    }))
}

async fn reset(State(shared): State<Shared>, Json(request): Json<ResetRequest>) -> Response {
    match shared.session().reset(&request) {
        Ok(result) => Json(result).into_response(),
        Err(error) => refused(StatusCode::BAD_REQUEST, &error),
    }
}

#[derive(serde::Deserialize)]
struct StepRequest {
    action: KantBenchAction,
}

/// A step over a session that lives for this request has no episode to step:
/// OpenEnv's simulation mode answers the same. Persistent play is `/ws`.
async fn step(State(shared): State<Shared>, Json(request): Json<StepRequest>) -> Response {
    match shared.session().step(&request.action) {
        Ok(result) => Json(result).into_response(),
        Err(error) => refused(StatusCode::CONFLICT, &error),
    }
}

async fn state(State(shared): State<Shared>) -> Response {
    match shared.session().state() {
        Ok(state) => Json(state).into_response(),
        Err(error) => refused(StatusCode::CONFLICT, &error),
    }
}

#[derive(serde::Deserialize)]
struct RewardRequest {
    prompt: String,
    text: String,
}

/// Ster's outside scorer: `{"prompt", "text"}` in, `{"reward", "move"}` out.
/// A server started without `--states` has no dataset to score against.
async fn reward(State(shared): State<Shared>, Json(request): Json<RewardRequest>) -> Response {
    let Some(scorer) = &shared.scorer else {
        let error = Error::Usage("this server was started without --states; start it with the states.json kant dataset wrote".to_owned());
        return refused(StatusCode::CONFLICT, &error);
    };
    match scorer.score(&request.prompt, &request.text) {
        Ok(scored) => Json(scored).into_response(),
        Err(error) => refused(StatusCode::UNPROCESSABLE_ENTITY, &error),
    }
}

/// Serve until the process is stopped, with at most `most` WebSocket
/// sessions open at once and, when `scorer` is given, `/reward` scoring
/// answers to its dataset. The bound address, which the operating system
/// chose when `listen` names port zero, is announced on standard error
/// before the first request is taken.
pub fn serve(listen: &str, settings: Arc<Settings>, most: usize, scorer: Option<Scorer>) -> Result<()> {
    let address: SocketAddr = listen
        .parse()
        .map_err(|_| Error::Usage(format!("--listen {listen} is not a socket address (host:port)")))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| Error::Io { path: "tokio runtime".into(), source })?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .map_err(|source| Error::Io { path: listen.into(), source })?;
        let bound = listener.local_addr().map_err(|source| Error::Io { path: listen.into(), source })?;
        eprintln!("kant: serving KantBench on http://{bound}");
        let shared = Shared {
            pairs: Arc::new(Library::standard()),
            groups: Arc::new(GroupLibrary::standard()),
            settings,
            places: Arc::new(Semaphore::new(most)),
            scorer: scorer.map(Arc::new),
        };
        axum::serve(listener, router(shared))
            .await
            .map_err(|source| Error::Io { path: bound.to_string().into(), source })
    })
}
