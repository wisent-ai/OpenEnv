//! One persistent session over a WebSocket, in OpenEnv's message shapes:
//! `{"type": "reset", "data": {...}}`, `{"type": "step", "data": {"move": ...}}`,
//! `{"type": "state"}` and `{"type": "close"}` in; `{"type": "observation",
//! "data": <step result>}`, `{"type": "state", "data": {...}}` and
//! `{"type": "error", "data": {"code", "message"}}` out. A session past the
//! server's declared number of open sessions is refused with
//! `CAPACITY_REACHED` and closed.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use serde_json::{json, Value};

use super::session::{ResetRequest, Session};
use super::space::KantBenchAction;
use super::Shared;

pub(super) async fn upgrade(State(shared): State<Shared>, socket: WebSocketUpgrade) -> Response {
    socket.on_upgrade(move |socket| run(socket, shared))
}

fn error(code: &str, message: impl std::fmt::Display) -> Value {
    json!({ "type": "error", "data": { "code": code, "message": message.to_string() } })
}

async fn send(socket: &mut WebSocket, value: &Value) -> bool {
    socket.send(Message::Text(value.to_string())).await.is_ok()
}

async fn run(mut socket: WebSocket, shared: Shared) {
    let Ok(_place) = shared.places.clone().try_acquire_owned() else {
        send(&mut socket, &error("CAPACITY_REACHED", "every session place of this server is taken; close another session first")).await;
        let _ = socket.close().await;
        return;
    };
    let mut session = shared.session();
    while let Some(Ok(message)) = socket.recv().await {
        let text = match message {
            Message::Text(text) => text,
            Message::Close(_) => break,
            _ => continue,
        };
        match answer(&mut session, &text) {
            Some(reply) => {
                if !send(&mut socket, &reply).await {
                    break;
                }
            }
            None => break,
        }
    }
    let _ = socket.close().await;
}

/// The reply to one message, or nothing when the client closes the session.
fn answer(session: &mut Session, text: &str) -> Option<Value> {
    let message: Value = match serde_json::from_str(text) {
        Ok(message) => message,
        Err(refusal) => return Some(error("INVALID_JSON", format!("the message is not JSON: {refusal}"))),
    };
    let data = message.get("data").cloned();
    let observed = |result: crate::error::Result<super::session::StepResult>| match result {
        Ok(result) => json!({ "type": "observation", "data": result }),
        Err(refusal) => error("EXECUTION_ERROR", refusal),
    };
    Some(match message.get("type").and_then(Value::as_str) {
        Some("reset") => match data.map(serde_json::from_value::<ResetRequest>) {
            Some(Ok(request)) => observed(session.reset(&request)),
            Some(Err(refusal)) => error("VALIDATION_ERROR", format!("reset data: {refusal}")),
            None => error("VALIDATION_ERROR", "reset needs data naming the game and strategy"),
        },
        Some("step") => match data.map(serde_json::from_value::<KantBenchAction>) {
            Some(Ok(action)) => observed(session.step(&action)),
            Some(Err(refusal)) => error("VALIDATION_ERROR", format!("step data: {refusal}")),
            None => error("VALIDATION_ERROR", "step needs data naming the move"),
        },
        Some("state") => match session.state() {
            Ok(state) => json!({ "type": "state", "data": state }),
            Err(refusal) => error("EXECUTION_ERROR", refusal),
        },
        Some("close") => return None,
        Some(other) => error("UNKNOWN_TYPE", format!("unknown message type {other}; send reset, step, state or close")),
        None => error("VALIDATION_ERROR", "the message has no string field type"),
    })
}
