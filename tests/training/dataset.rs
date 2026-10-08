//! Training data and rewards as Ster meets them: `kant dataset` writes one
//! prompt per state an agent met, a pair set whose positive side is the move
//! with the higher expected payoff, and the states; `kant serve --states`
//! pays an answer its expected payoff against a uniform opponent, pays an
//! unparsable answer the declared amount, and refuses a prompt it never
//! wrote.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_kant");

const SETTINGS: &str = "{
  \"seed\": 4,
  \"server\": { \"sessions\": 1 },
  \"agent\": { \"history_rounds\": 3 },
  \"training\": { \"episodes\": 1, \"pair_margin\": 0.5, \"unparsed_reward\": -1 },
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

fn scratch(name: &str) -> PathBuf {
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

fn read(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("written file")).expect("JSON")
}

fn declared() -> Value {
    serde_json::from_str(SETTINGS).expect("settings parse")
}

/// A move's mean payoff against every opponent move, from the declared cells.
fn expected(played: &str) -> f64 {
    let row = declared()["games"]["prisoners_dilemma"]["payoffs"][played].clone();
    let cells = row.as_object().expect("a row").values().map(|pair| pair.as_array().and_then(|pair| pair.first()).and_then(Value::as_f64).expect("a payoff")).collect::<Vec<_>>();
    cells.iter().sum::<f64>() / cells.len() as f64
}

fn post(address: &str, body: &Value) -> (String, Value) {
    let mut stream = TcpStream::connect(address).expect("the server answers");
    let text = body.to_string();
    write!(
        stream,
        "POST /reward HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )
    .expect("request");
    let mut answer = String::new();
    stream.read_to_string(&mut answer).expect("response");
    let (head, body) = answer.split_once("\r\n\r\n").expect("a response body");
    let status = head.lines().next().expect("a status line").to_owned();
    (status, serde_json::from_str(body).expect("a JSON body"))
}

#[test]
fn the_dataset_feeds_ster_and_the_server_pays_answers_to_it() {
    let directory = scratch("training-dataset");
    let settings = directory.join("settings.json");
    std::fs::write(&settings, SETTINGS).expect("settings file");
    let output_directory = directory.join("data");
    let written = Command::new(BINARY)
        .args(["dataset", "--settings"])
        .arg(&settings)
        .args(["--game", "prisoners_dilemma", "--strategy", "tit_for_tat", "--agent-strategy", "always_cooperate", "--output"])
        .arg(&output_directory)
        .output()
        .expect("kant runs");
    assert!(written.status.success(), "{}", String::from_utf8_lossy(&written.stderr));

    let rounds = declared()["games"]["prisoners_dilemma"]["rounds"].as_u64().expect("rounds") as usize;
    let prompts = read(output_directory.join("prompts.json"));
    let states = read(output_directory.join("states.json"));
    let pairs = read(output_directory.join("pairs.json"));
    let prompt_list = prompts["prompts"].as_array().expect("prompts").clone();
    assert_eq!(prompt_list.len(), rounds, "one prompt per round the agent played");
    assert_eq!(states["states"].as_array().map(Vec::len), Some(rounds));
    let first_pair = pairs["pairs"].as_array().and_then(|pairs| pairs.first()).cloned().expect("a pair");
    assert!(first_pair["positive"].as_str().is_some_and(|side| side.ends_with("\ndefect")), "{first_pair}");
    assert!(first_pair["negative"].as_str().is_some_and(|side| side.ends_with("\ncooperate")), "{first_pair}");

    let mut child = Command::new(BINARY)
        .args(["serve", "--settings"])
        .arg(&settings)
        .args(["--listen", &SocketAddr::from((Ipv4Addr::LOCALHOST, Default::default())).to_string(), "--states"])
        .arg(output_directory.join("states.json"))
        .stderr(Stdio::piped())
        .spawn()
        .expect("kant serve starts");
    let mut announced = String::new();
    BufReader::new(child.stderr.take().expect("stderr")).read_line(&mut announced).expect("announcement");
    let _served = Served(child);
    let address = announced.rsplit("http://").next().expect("address").trim().to_owned();
    let prompt = prompt_list.first().cloned().expect("a prompt");

    let (_, paid) = post(&address, &json!({ "prompt": prompt, "text": "defect" }));
    assert_eq!(paid["reward"].as_f64(), Some(expected("defect")), "{paid}");
    assert_eq!(paid["move"], "defect");
    let (_, unparsed) = post(&address, &json!({ "prompt": prompt, "text": "I would rather not say" }));
    assert_eq!(unparsed["reward"].as_f64(), declared()["training"]["unparsed_reward"].as_f64(), "{unparsed}");

    let (status, unknown) = post(&address, &json!({ "prompt": "a prompt nobody wrote", "text": "defect" }));
    assert!(status.contains("422"), "{status}");
    assert!(unknown["detail"].as_str().is_some_and(|said| said.contains("not one of this dataset's states")), "{unknown}");
}
