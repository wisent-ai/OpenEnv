//! `kant`: KantBench's command line. Every answer is one JSON document on
//! standard output; every refusal is one line on standard error that names
//! what is missing, and the exit status fails.

use std::process::ExitCode;

use kantbench::cli::{self, Words};
use kantbench::Error;

const USAGE: &str = "usage (command first, then options):
  kant games [--settings FILE]
                    (every game, what it reads from a settings document, and with
                     --settings whether that document builds it)
  kant strategies [--settings FILE]
                    (every opponent strategy and what it reads)
  kant play --settings FILE --game G --strategy S --move M [--move M]...
            [--rounds N] [--episode ID]
                    (one episode against a strategy, the agent's moves in order)";

fn dispatch(args: &[String]) -> Result<serde_json::Value, Error> {
    let words = Words::parse(args);
    let Some(command) = words.positionals.first() else {
        return Err(Error::Usage(USAGE.to_owned()));
    };
    match command.as_str() {
        "games" => cli::catalog::games(&words),
        "strategies" => cli::catalog::strategies(&words),
        "play" => cli::play::run(&words),
        other => Err(Error::Usage(format!("unknown command {other}\n{USAGE}"))),
    }
}

fn main() -> ExitCode {
    let mut program = std::env::args();
    // The first word is the program's own name.
    program.next();
    let args: Vec<String> = program.collect();
    match dispatch(&args) {
        Ok(answer) => {
            println!("{answer:#}");
            ExitCode::SUCCESS
        }
        Err(refusal) => {
            eprintln!("kant: {refusal}");
            ExitCode::FAILURE
        }
    }
}
