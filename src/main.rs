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
                    (one episode against a strategy, the agent's moves in order)
  kant group --settings FILE --game G --strategy S [--strategy S]... --move M [--move M]...
             [--rounds N] [--episode ID]
                    (one group episode; one strategy for every other seat, or one each)
  kant coalition --settings FILE --game G --strategy S [--strategy S]...
                 --governance S [--governance S]... --script FILE [--rounds N] [--episode ID]
                    (one coalition episode; the script is a JSON list of
                     {\"negotiate\": {...}} and {\"move\": M} steps)
  kant serve --settings FILE --listen ADDRESS [--states FILE]
                    (the OpenEnv server: /ws sessions, /reset, /web explorer; at most
                     server.sessions sessions open at once; with --states, /reward scores
                     answers to a dataset's prompts; announces its bound address on
                     standard error)
  kant tournament --settings FILE --game G [--game G]... --strategy S [--strategy S]...
                  (--agent-route R | --agent-strategy S) [--opponent-route R]
                    (the agent against every strategy in every game, evaluation.episodes
                     times each, with the metrics; a model seat goes through Brama)
  kant dataset --settings FILE --game G [--game G]... --strategy S [--strategy S]...
               --agent-strategy A --output DIRECTORY
                    (prompts.json for ster tune grpo, pairs.json for ster tune dpo, and the
                     states.json kant serve --states scores against)
  kant matchups --settings FILE --game G [--game G]... --seat NAME=route:R|NAME=strategy:S...
                    (every seat against every seat, itself included, in every game,
                     evaluation.episodes times; arena reputation when arena is declared)";

fn dispatch(args: &[String]) -> Result<serde_json::Value, Error> {
    let words = Words::parse(args);
    let Some(command) = words.positionals.first() else {
        return Err(Error::Usage(USAGE.to_owned()));
    };
    match command.as_str() {
        "games" => cli::episodes::catalog::games(&words),
        "strategies" => cli::episodes::catalog::strategies(&words),
        "play" => cli::episodes::play::run(&words),
        "group" => cli::episodes::group::group(&words),
        "coalition" => cli::episodes::group::coalition(&words),
        "serve" => cli::services::serve::run(&words),
        "tournament" => cli::services::tournament::run(&words),
        "dataset" => cli::services::dataset::run(&words),
        "matchups" => cli::services::matchups::run(&words),
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
