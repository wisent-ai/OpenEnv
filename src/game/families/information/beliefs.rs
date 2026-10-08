//! Incomplete-information and network games played as normal forms: global
//! game, jury voting, information cascade, adverse selection, security, link
//! formation, trust with punishment and dueling. Each reads `payoffs`.

use crate::game::{matrix_entry, Library};

pub(super) fn register(library: &mut Library) {
    for entry in [
        matrix_entry(
            "global_game",
            "information",
            &["attack", "wait"],
            "Global Game",
            "A coordination game modeling regime change or bank runs under incomplete information. Players receive private signals about fundamentals and choose to attack or wait. Successful coordination on attack yields high payoffs but unilateral attack is costly. Tests strategic behavior under private information.",
        ),
        matrix_entry(
            "jury_voting",
            "information",
            &["guilty", "acquit"],
            "Jury Voting Game",
            "Two jurors simultaneously vote guilty or acquit under a unanimity rule. Conviction requires both voting guilty. Each juror has a private signal about the defendant. Strategic voting may differ from sincere voting. Tests information aggregation under voting.",
        ),
        matrix_entry(
            "information_cascade",
            "information",
            &["follow_signal", "follow_crowd"],
            "Information Cascade Game",
            "Players choose whether to follow their own private signal or follow the crowd. Independent signal-following leads to better information aggregation while crowd-following creates herding. Asymmetric payoffs reflect the benefit of diverse information. Tests independence of judgment under social influence.",
        ),
        matrix_entry(
            "adverse_selection_insurance",
            "information",
            &["reveal_type", "hide_type"],
            "Adverse Selection Insurance Game",
            "An insurance market game with asymmetric information. Each player can reveal their private risk type for efficient pricing or hide it to exploit information asymmetry. Mutual revelation enables fair pricing. Hiding while the other reveals creates adverse selection profit. Tests screening and pooling dynamics.",
        ),
        matrix_entry(
            "security_game",
            "information",
            &["target_a", "target_b"],
            "Security Game",
            "An attacker-defender game where the defender allocates protection to one of two targets and the attacker simultaneously chooses which target to attack. Matching the attacker's target means a successful defense. Misallocation lets the attacker succeed. Tests strategic resource allocation under adversarial uncertainty.",
        ),
        matrix_entry(
            "link_formation",
            "information",
            &["connect", "isolate"],
            "Link Formation Game",
            "A network formation game where two players simultaneously decide whether to form a connection. A link forms only when both agree. Mutual connection yields network benefits. Unilateral connection attempt is costly. Mutual isolation yields nothing. Tests bilateral consent in network formation.",
        ),
        matrix_entry(
            "trust_with_punishment",
            "information",
            &["cooperate", "defect", "punish"],
            "Trust with Punishment Game",
            "An extended trust game where players can cooperate or defect as in the standard Prisoner's Dilemma plus a costly punishment action. Punishing reduces the opponent's payoff but also costs the punisher. Tests whether altruistic punishment enforces cooperation even at personal cost.",
        ),
        matrix_entry(
            "dueling_game",
            "information",
            &["fire_early", "fire_late"],
            "Dueling Game",
            "A timing game where two players simultaneously choose when to fire: early for a safe but moderate payoff or late for higher accuracy. Firing early against a late opponent is advantageous. Mutual late firing yields better outcomes than mutual early. Tests patience versus preemption under uncertainty.",
        ),
    ] {
        library.add(entry);
    }
}
