//! Variants that tag every move: `<prefix>_<tag>_<move>`. Cheap talk tags a
//! move with the move a seat says it will play; gossip with a rating of the
//! opponent; the meta-games with a rule. The tag never changes the base
//! move, so each composed move maps back to its tag and base move, looked up
//! rather than split, since base moves carry underscores of their own.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use rand::RngCore;

use crate::error::{Error, Result};
use crate::game::{Game, OpponentMoves};

use super::rules::Rule;

/// Composed moves and the map from each back to `(tag, base move)`.
pub(super) struct Tagged {
    pub moves: Vec<String>,
    pub parts: BTreeMap<String, (String, String)>,
}

pub(super) fn tag(prefix: &str, tags: &[String], base: &[String]) -> Tagged {
    let mut moves = Vec::new();
    let mut parts = BTreeMap::new();
    for label in tags {
        for played in base {
            let composed = format!("{prefix}_{label}_{played}");
            parts.insert(composed.clone(), (label.clone(), played.clone()));
            moves.push(composed);
        }
    }
    Tagged { moves, parts }
}

fn split(game: &str, parts: &BTreeMap<String, (String, String)>, action: &str) -> Result<(String, String)> {
    parts.get(action).cloned().ok_or_else(|| Error::InvalidAction {
        game: game.to_owned(),
        action: action.to_owned(),
        allowed: parts.keys().cloned().collect::<Vec<_>>().join(", "),
    })
}

/// How a round of tagged moves pays, given both tags and the base payoffs.
pub(super) enum Settle {
    /// Tags are talk: the base payoffs stand.
    Talk,
    /// Matching rules bind for this round.
    Proposal(BTreeMap<String, Rule>),
    /// The first matching rule other than `none` binds for the rest of the
    /// episode.
    Constitution(BTreeMap<String, Rule>),
}

/// `base` with every move tagged by `prefix` and `tags` on both seats.
pub(super) fn compose(base: Game, variant: &str, prefix: &str, tags: &[String], settle: Settle) -> Result<Game> {
    let agent = tag(prefix, tags, &base.actions);
    let opponent = match &base.opponent_actions {
        OpponentMoves::Shared => None,
        OpponentMoves::Own(moves) => Some(tag(prefix, tags, moves)),
    };
    let cooperative = base.actions.first().cloned().ok_or_else(|| Error::Unsupported {
        game: base.key.clone(),
        reason: format!("{variant} needs a game with moves"),
    })?;
    let name = base.key.clone();
    let agent_parts = agent.parts.clone();
    let opponent_parts = match &opponent {
        Some(tagged) => tagged.parts.clone(),
        None => agent.parts.clone(),
    };
    let inner = base.payoff.clone();
    let adopted: Mutex<Option<Rule>> = Mutex::new(None);
    let payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        let (said, mine) = split(&name, &agent_parts, player)?;
        let (heard, theirs) = split(&name, &opponent_parts, opponent)?;
        let paid = inner(&mine, &theirs, rng)?;
        let moves = (mine.as_str(), theirs.as_str());
        Ok(match &settle {
            Settle::Talk => paid,
            Settle::Proposal(rules) if said == heard => match rules.get(&said) {
                Some(rule) => rule.apply(paid, moves, &cooperative),
                None => paid,
            },
            Settle::Proposal(_) => paid,
            Settle::Constitution(rules) => {
                let mut adopted = adopted.lock().map_err(|_| Error::Unsupported {
                    game: name.clone(),
                    reason: "its adopted rule was left broken by a panic in an earlier round".to_owned(),
                })?;
                if adopted.is_none() && said == heard && said != "none" {
                    *adopted = rules.get(&said).copied();
                }
                match *adopted {
                    Some(rule) => rule.apply(paid, moves, &cooperative),
                    None => paid,
                }
            }
        })
    });
    let mut game = base;
    game.actions = agent.moves;
    if let Some(tagged) = opponent {
        game.opponent_actions = OpponentMoves::Own(tagged.moves);
    }
    game.payoff = payoff;
    game.variants.push(variant.to_owned());
    Ok(game)
}

/// The proposer tags its move with a rule; the responder answers
/// `raccept_<move>` or `rreject_<move>`. An accepted rule binds this round.
pub(super) fn proposer_responder(base: Game, rules: BTreeMap<String, Rule>) -> Result<Game> {
    let tags: Vec<String> = rules.keys().cloned().collect();
    let agent = tag("rprop", &tags, &base.actions);
    let mut answers = BTreeMap::new();
    let mut opponent_moves = Vec::new();
    for played in base.opponent_moves() {
        let accept = format!("raccept_{played}");
        let reject = format!("rreject_{played}");
        answers.insert(accept.clone(), (true, played.clone()));
        answers.insert(reject.clone(), (false, played.clone()));
        opponent_moves.push(accept);
        opponent_moves.push(reject);
    }
    let cooperative = base.actions.first().cloned().ok_or_else(|| Error::Unsupported {
        game: base.key.clone(),
        reason: "proposer_responder needs a game with moves".to_owned(),
    })?;
    let name = base.key.clone();
    let parts = agent.parts.clone();
    let inner = base.payoff.clone();
    let payoff = Arc::new(move |player: &str, opponent: &str, rng: &mut dyn RngCore| {
        let (proposed, mine) = split(&name, &parts, player)?;
        let (accepted, theirs) = answers.get(opponent).cloned().ok_or_else(|| Error::InvalidAction {
            game: name.clone(),
            action: opponent.to_owned(),
            allowed: answers.keys().cloned().collect::<Vec<_>>().join(", "),
        })?;
        let paid = inner(&mine, &theirs, rng)?;
        Ok(match (accepted, rules.get(&proposed)) {
            (true, Some(rule)) => rule.apply(paid, (&mine, &theirs), &cooperative),
            _ => paid,
        })
    });
    let mut game = base;
    game.actions = agent.moves;
    game.opponent_actions = OpponentMoves::Own(opponent_moves);
    game.payoff = payoff;
    game.responds = true;
    game.variants.push("proposer_responder".to_owned());
    Ok(game)
}
