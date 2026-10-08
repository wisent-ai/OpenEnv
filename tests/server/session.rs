//! The environment server as a client meets it: the built `kant serve` over a
//! settings document, a WebSocket session that resets and steps a game, the
//! explorer page, and the refusal of a session past the declared
//! `server.sessions`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};
use tungstenite::{connect, Message};

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

/// One session place, so a second open session is refused.
const SETTINGS: &str = "{
  \"seed\": 9,
  \"server\": { \"sessions\": 1 },
  \"games\": {
    \"prisoners_dilemma\": {
      \"rounds\": 2,
      \"payoffs\": {
        \"cooperate\": { \"cooperate\": [3, 3], \"defect\": [0, 5] },
        \"defect\": { \"cooperate\": [5, 0], \"defect\": [1, 1] }
      }
    }
  }
}";

struct Served(Child);

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The loopback address with the port left to the operating system (port
/// zero: https://doc.rust-lang.org/std/net/struct.TcpListener.html#method.bind).
fn any_loopback_port() -> String {
    SocketAddr::from((Ipv4Addr::LOCALHOST, Default::default())).to_string()
}

fn serve(name: &str) -> (Served, String) {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let settings = directory.join("settings.json");
    std::fs::write(&settings, SETTINGS).expect("settings file");
    let mut child = Command::new(BINARY)
        .args(["serve", "--settings"])
        .arg(&settings)
        .args(["--listen", &any_loopback_port()])
        .stderr(Stdio::piped())
        .spawn()
        .expect("kant serve starts");
    let mut announced = String::new();
    BufReader::new(child.stderr.take().expect("stderr"))
        .read_line(&mut announced)
        .expect("the bound address is announced");
    let address = announced.rsplit("http://").next().expect("address").trim().to_owned();
    (Served(child), address)
}

type Socket = tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<TcpStream>>;

fn reply(socket: &mut Socket) -> Value {
    loop {
        if let Message::Text(text) = socket.read().expect("a reply") {
            return serde_json::from_str(&text).expect("a JSON reply");
        }
    }
}

fn exchange(socket: &mut Socket, message: Value) -> Value {
    socket.send(Message::Text(message.to_string())).expect("message sent");
    reply(socket)
}

fn declared_cell(row: &str, column: &str) -> Value {
    let settings: Value = serde_json::from_str(SETTINGS).expect("settings parse");
    settings["games"]["prisoners_dilemma"]["payoffs"][row][column].clone()
}

#[test]
fn a_websocket_session_plays_a_game_and_a_second_session_is_refused() {
    let (_served, address) = serve("server-session");
    let (mut socket, _) = connect(format!("ws://{address}/ws")).expect("the session opens");
    let reset = exchange(&mut socket, json!({ "type": "reset", "data": { "game": "prisoners_dilemma", "strategy": "tit_for_tat" } }));
    assert_eq!(reset["type"], "observation", "{reset}");
    assert_eq!(reset["data"]["observation"]["available_moves"], json!(["cooperate", "defect"]));
    assert!(reset["data"]["observation"].get("your_payoff").is_none(), "no round has paid anything yet");

    let first = exchange(&mut socket, json!({ "type": "step", "data": { "move": "defect" } }));
    let paid = declared_cell("defect", "cooperate");
    assert_eq!(first["data"]["observation"]["opponent_move"], "cooperate");
    assert_eq!(first["data"]["reward"].as_f64(), paid.as_array().and_then(|pair| pair.first()).and_then(Value::as_f64));
    assert_eq!(first["data"]["done"], Value::Bool(false));

    let (mut second, _) = connect(format!("ws://{address}/ws")).expect("the second socket opens");
    let refused = reply(&mut second);
    assert_eq!(refused["data"]["code"], "CAPACITY_REACHED", "{refused}");

    let invalid = exchange(&mut socket, json!({ "type": "step", "data": { "move": "betray" } }));
    assert_eq!(invalid["data"]["code"], "EXECUTION_ERROR");
    assert!(invalid["data"]["message"].as_str().is_some_and(|said| said.contains("betray is not a move")), "{invalid}");

    let last = exchange(&mut socket, json!({ "type": "step", "data": { "move": "cooperate" } }));
    assert_eq!(last["data"]["done"], Value::Bool(true));
    let undeclared = exchange(&mut socket, json!({ "type": "reset", "data": { "game": "stag_hunt", "strategy": "grudger" } }));
    assert!(
        undeclared["data"]["message"].as_str().is_some_and(|said| said.contains("games declares no stag_hunt")),
        "{undeclared}"
    );
}

#[test]
fn the_explorer_page_and_its_script_are_served() {
    let (_served, address) = serve("server-explorer");
    for (path, expected) in [("/web", "KantBench explorer"), ("/explorer.js", "new WebSocket"), ("/games", "prisoners_dilemma")] {
        let mut stream = TcpStream::connect(&address).expect("the server answers");
        write!(stream, "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n").expect("request");
        let mut answer = String::new();
        stream.read_to_string(&mut answer).expect("response");
        assert!(answer.contains(expected), "{path}: {answer}");
    }
}

#[test]
fn a_settings_document_without_a_session_count_is_refused_before_listening() {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("server-refusal");
    std::fs::create_dir_all(&directory).expect("scratch directory");
    let settings = directory.join("settings.json");
    std::fs::write(&settings, "{ \"games\": {} }").expect("settings file");
    let output = Command::new(BINARY)
        .args(["serve", "--settings"])
        .arg(&settings)
        .args(["--listen", &any_loopback_port()])
        .output()
        .expect("kant runs");
    assert!(!output.status.success());
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(said.contains("declares no server"), "{said}");
}
