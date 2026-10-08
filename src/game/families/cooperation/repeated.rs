//! Repeated, behavioral and Prisoner's Dilemma variant games that are normal
//! forms: each reads its cells from `payoffs`. The optional dilemma adds an
//! exit to a dilemma, reading the exit's numbers from its `exit` object.

use crate::game::{declared_matrix, matrix_entry, moves, Entry, Library};
use crate::variant;

type Row = (&'static str, &'static [&'static str], &'static str, &'static str);

pub(super) fn register(library: &mut Library) {
    let table: Vec<Row> = vec![
        ("bank_run", &["wait", "withdraw"], "Bank Run (Diamond-Dybvig)", "Depositors simultaneously decide whether to withdraw early. If both wait, the bank survives and both earn a premium. If both withdraw, the bank fails. Models coordination failure in financial systems."),
        ("global_stag_hunt", &["stag", "hare"], "Global Stag Hunt", "A higher-stakes Stag Hunt modeling coordination under uncertainty. Both hunting stag yields a large payoff but hunting stag alone yields nothing. Models bank runs, currency attacks, and regime change dynamics."),
        ("hawk_dove_bourgeois", &["dove", "hawk", "bourgeois"], "Hawk-Dove-Bourgeois", "Extended Hawk-Dove with a Bourgeois strategy that plays Hawk when incumbent and Dove when intruder. The Bourgeois strategy is an evolutionarily stable strategy. Tests reasoning about ownership conventions."),
        ("finitely_repeated_pd", &["cooperate", "defect"], "Finitely Repeated Prisoner's Dilemma", "A Prisoner's Dilemma played for a known finite number of rounds. Backward induction predicts mutual defection in every round, yet cooperation often emerges experimentally. Tests backward induction versus cooperation heuristics."),
        ("markov_game", &["cooperate", "defect"], "Markov Decision Game", "A repeated game where the payoff structure shifts based on recent history. Players must adapt strategies to changing incentives. Tests dynamic programming and Markov-perfect equilibrium reasoning over multiple rounds."),
        ("asymmetric_pd", &["cooperate", "defect"], "Asymmetric Prisoner's Dilemma", "A Prisoner's Dilemma where players have unequal payoff structures. The first player has an alibi advantage with a higher punishment payoff. Tests strategic reasoning under asymmetric incentive conditions."),
        ("donation_game", &["donate", "keep"], "Donation Game", "A simplified cooperation model: each player independently decides whether to donate. Donating costs the donor but gives a larger benefit to the recipient. The dominant strategy is to keep, but mutual donation is Pareto superior."),
        ("friend_or_foe", &["friend", "foe"], "Friend or Foe", "A game show variant of the Prisoner's Dilemma. If both choose friend, winnings are shared. If one steals (foe), they take all. If both choose foe, neither gets anything. Unlike standard PD, mutual defection yields zero, creating a weak equilibrium."),
        ("peace_war", &["disarm", "arm"], "Peace-War Game", "An international relations framing of the Prisoner's Dilemma. Players choose to arm or disarm. Mutual disarmament yields the best joint outcome but unilateral arming dominates. Models the security dilemma and arms race escalation dynamics."),
        ("discounted_pd", &["cooperate", "defect"], "Discounted Prisoner's Dilemma", "A high-stakes Prisoner's Dilemma with many rounds, modeling an effectively infinite repeated interaction. The shadow of the future makes cooperation sustainable under folk theorem conditions. Tests long-horizon strategic reasoning with higher temptation and reward differentials."),
        ("stochastic_pd", &["cooperate", "defect"], "Stochastic Prisoner's Dilemma", "A Prisoner's Dilemma variant where action execution is noisy. With some probability each player's intended action is flipped. Expected payoffs differ from the standard PD, reflecting the tremble probabilities. Tests robustness of strategies to noise."),
        ("risk_dominance", &["risky", "safe"], "Risk Dominance Game", "A coordination game with two pure Nash equilibria: one payoff-dominant (risky-risky yields higher mutual payoff) and one risk-dominant (safe-safe is more robust to uncertainty). Tests whether agents optimize for payoff or safety under strategic uncertainty about the opponent's behavior."),
        ("evolutionary_pd", &["always_coop", "always_defect", "tit_for_tat"], "Evolutionary Prisoner's Dilemma", "A multi-strategy Prisoner's Dilemma representing long-run evolutionary dynamics. Players choose from always cooperate and always defect and tit-for-tat. Payoffs represent expected long-run fitness across many interactions between strategies."),
    ];
    for (key, actions, name, description) in table {
        library.add(matrix_entry(key, "cooperation", actions, name, description));
    }
    library.add(Entry::new("optional_pd", "cooperation", &["payoffs", "exit"], |declared| {
        let base = declared_matrix(
            declared,
            "Optional Prisoner's Dilemma",
            "A Prisoner's Dilemma with a third action: exit. Exiting gives a safe intermediate payoff regardless of the opponent's choice. Tests whether outside options change cooperation dynamics and models situations where players can walk away from interactions.",
            moves(&["cooperate", "defect"]),
        )?;
        let mut game = variant::apply(base, "exit", &|| declared.nested("exit"))?;
        game.base = "prisoners_dilemma".to_owned();
        Ok(game)
    }));
}
