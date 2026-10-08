//! The explorer's views as the page calls them: `/game/<key>` shows the
//! payoffs the settings document declares, `/tournament` runs a tournament,
//! and a tournament without a named agent seat is refused.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const SETTINGS: &str = "{
  \"seed\": 9,
  \"server\": { \"sessions\": 1 },
  \"evaluation\": { \"episodes\": 1 },
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

fn http(address: &str, method: &str, path: &str, body: &str) -> Value {
    let mut stream = TcpStream::connect(address).expect("the server answers");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("request");
    let mut answer = String::new();
    stream.read_to_string(&mut answer).expect("response");
    let (_, text) = answer.split_once("\r\n\r\n").expect("a response body");
    serde_json::from_str(text).expect("a JSON body")
}

fn numbers(value: &Value) -> Option<Vec<f64>> {
    value.as_array().map(|pair| pair.iter().filter_map(Value::as_f64).collect())
}

#[test]
fn the_explorer_shows_declared_payoffs_and_runs_a_tournament() {
    let (_served, address) = serve("server-views");
    let settings: Value = serde_json::from_str(SETTINGS).expect("settings parse");
    let built = http(&address, "GET", "/game/prisoners_dilemma", "");
    let rows: Vec<&str> = built["rows"].as_array().expect("rows").iter().filter_map(Value::as_str).collect();
    let columns: Vec<&str> = built["columns"].as_array().expect("columns").iter().filter_map(Value::as_str).collect();
    for (row_index, row) in rows.iter().enumerate() {
        for (column_index, column) in columns.iter().enumerate() {
            let declared = &settings["games"]["prisoners_dilemma"]["payoffs"][row][column];
            assert_eq!(numbers(&built["cells"][row_index][column_index]), numbers(declared), "{row} against {column}");
        }
    }

    let request = json!({ "games": ["prisoners_dilemma"], "strategies": ["always_defect", "tit_for_tat"], "agent_strategy": "grudger" });
    let result = http(&address, "POST", "/tournament", &request.to_string());
    assert!(result["metrics"]["strategic_reasoning"].is_number(), "{result}");

    let unnamed = json!({ "games": ["prisoners_dilemma"], "strategies": ["grudger"] });
    let refused = http(&address, "POST", "/tournament", &unnamed.to_string());
    assert!(
        refused["detail"].as_str().is_some_and(|said| said.contains("exactly one of agent_route or agent_strategy")),
        "{refused}"
    );
}
