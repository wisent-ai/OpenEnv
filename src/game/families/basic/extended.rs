//! Further normal-form games: zero-sum, coordination and the two-seat forms of
//! the social dilemmas whose group forms live in `nplayer`. Each reads its
//! cells from `payoffs`.

use crate::game::{matrix_entry, Library};

pub(super) fn register(library: &mut Library) {
    for entry in [
        matrix_entry(
            "matching_pennies",
            "extended",
            &["heads", "tails"],
            "Matching Pennies",
            "A pure zero-sum game. The matcher wins if both choose the same side; the mismatcher wins if they differ. The only Nash equilibrium is a mixed strategy of equal randomization.",
        ),
        matrix_entry(
            "rock_paper_scissors",
            "extended",
            &["rock", "paper", "scissors"],
            "Rock-Paper-Scissors",
            "A three-action zero-sum game: rock beats scissors, scissors beats paper, paper beats rock. The unique Nash equilibrium is uniform randomization over all three actions.",
        ),
        matrix_entry(
            "battle_of_the_sexes",
            "extended",
            &["opera", "football"],
            "Battle of the Sexes",
            "Two players want to coordinate but have different preferences. The first player prefers opera, the second prefers football. Both prefer any coordination over miscoordination. Two pure Nash equilibria exist at (opera, opera) and (football, football).",
        ),
        matrix_entry(
            "pure_coordination",
            "extended",
            &["left", "right"],
            "Pure Coordination",
            "Two players receive a positive payoff only when they choose the same action. Both (left, left) and (right, right) are Nash equilibria. Tests whether agents can converge on a focal point without communication.",
        ),
        matrix_entry(
            "deadlock",
            "extended",
            &["cooperate", "defect"],
            "Deadlock",
            "Similar to the Prisoner's Dilemma but with different payoff ordering: DC > DD > CC > CD. Both players prefer mutual defection over mutual cooperation. The unique Nash equilibrium is (defect, defect) and it is also Pareto optimal.",
        ),
        matrix_entry(
            "harmony",
            "extended",
            &["cooperate", "defect"],
            "Harmony",
            "The opposite of a social dilemma: cooperation is the dominant strategy for both players. Payoff ordering CC > DC > CD > DD means rational self-interest naturally leads to the socially optimal outcome of mutual cooperation.",
        ),
        matrix_entry(
            "volunteer_dilemma",
            "extended",
            &["volunteer", "abstain"],
            "Volunteer's Dilemma",
            "At least one player must volunteer (at personal cost) for everyone to receive a benefit. If nobody volunteers, all get nothing. Models bystander effects and public good provision.",
        ),
        matrix_entry(
            "el_farol",
            "extended",
            &["attend", "stay_home"],
            "El Farol Bar Problem",
            "Each player decides whether to attend a bar. If attendance is below capacity, going is better than staying home. If the bar is crowded, staying home is better. Models minority games and congestion dynamics.",
        ),
    ] {
        library.add(entry);
    }
}
