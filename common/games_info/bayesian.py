"""Bayesian and incomplete information games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.batch4.bayesian_constants import (
    GG_ATTACK_ATTACK, GG_ATTACK_WAIT, GG_WAIT_ATTACK, GG_WAIT_WAIT,
    JV_CONVICT_CONVICT, JV_ACQUIT_ACQUIT, JV_SPLIT_VOTE,
    IC_SIGNAL_SIGNAL, IC_SIGNAL_CROWD, IC_CROWD_SIGNAL, IC_CROWD_CROWD,
    ASI_REVEAL_REVEAL, ASI_REVEAL_HIDE, ASI_HIDE_REVEAL, ASI_HIDE_HIDE,
)


# -- Global Game (regime change / bank run under private signals) --
_GG: dict[tuple[str, str], tuple[float, float]] = {
    ("attack", "attack"):   (float(GG_ATTACK_ATTACK), float(GG_ATTACK_ATTACK)),
    ("attack", "wait"):     (float(GG_ATTACK_WAIT), float(GG_WAIT_ATTACK)),
    ("wait", "attack"):     (float(GG_WAIT_ATTACK), float(GG_ATTACK_WAIT)),
    ("wait", "wait"):       (float(GG_WAIT_WAIT), float(GG_WAIT_WAIT)),
}


# -- Jury Voting (unanimity rule for conviction) --
_JV: dict[tuple[str, str], tuple[float, float]] = {
    ("guilty", "guilty"):   (float(JV_CONVICT_CONVICT), float(JV_CONVICT_CONVICT)),
    ("guilty", "acquit"):   (float(JV_SPLIT_VOTE), float(JV_SPLIT_VOTE)),
    ("acquit", "guilty"):   (float(JV_SPLIT_VOTE), float(JV_SPLIT_VOTE)),
    ("acquit", "acquit"):   (float(JV_ACQUIT_ACQUIT), float(JV_ACQUIT_ACQUIT)),
}


# -- Information Cascade (follow own signal vs follow crowd) --
_IC: dict[tuple[str, str], tuple[float, float]] = {
    ("follow_signal", "follow_signal"): (
        float(IC_SIGNAL_SIGNAL), float(IC_SIGNAL_SIGNAL),
    ),
    ("follow_signal", "follow_crowd"): (
        float(IC_SIGNAL_CROWD), float(IC_CROWD_SIGNAL),
    ),
    ("follow_crowd", "follow_signal"): (
        float(IC_CROWD_SIGNAL), float(IC_SIGNAL_CROWD),
    ),
    ("follow_crowd", "follow_crowd"): (
        float(IC_CROWD_CROWD), float(IC_CROWD_CROWD),
    ),
}


# -- Adverse Selection (reveal or hide private type) --
_ASI: dict[tuple[str, str], tuple[float, float]] = {
    ("reveal_type", "reveal_type"): (
        float(ASI_REVEAL_REVEAL), float(ASI_REVEAL_REVEAL),
    ),
    ("reveal_type", "hide_type"): (
        float(ASI_REVEAL_HIDE), float(ASI_HIDE_REVEAL),
    ),
    ("hide_type", "reveal_type"): (
        float(ASI_HIDE_REVEAL), float(ASI_REVEAL_HIDE),
    ),
    ("hide_type", "hide_type"): (
        float(ASI_HIDE_HIDE), float(ASI_HIDE_HIDE),
    ),
}


# -- Register --
BAYESIAN_GAMES: dict[str, GameConfig] = {
    "global_game": GameConfig(
        name="Global Game",
        description=(
            "A coordination game modeling regime change or bank runs under "
            "incomplete information. Players receive private signals about "
            "fundamentals and choose to attack or wait. Successful coordination "
            "on attack yields high payoffs but unilateral attack is costly. "
            "Tests strategic behavior under private information."
        ),
        actions=["attack", "wait"],
        game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_GG),
    ),
    "jury_voting": GameConfig(
        name="Jury Voting Game",
        description=(
            "Two jurors simultaneously vote guilty or acquit under a unanimity "
            "rule. Conviction requires both voting guilty. Each juror has a "
            "private signal about the defendant. Strategic voting may differ "
            "from sincere voting. Tests information aggregation under voting."
        ),
        actions=["guilty", "acquit"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_JV),
    ),
    "information_cascade": GameConfig(
        name="Information Cascade Game",
        description=(
            "Players choose whether to follow their own private signal or "
            "follow the crowd. Independent signal-following leads to better "
            "information aggregation while crowd-following creates herding. "
            "Asymmetric payoffs reflect the benefit of diverse information. "
            "Tests independence of judgment under social influence."
        ),
        actions=["follow_signal", "follow_crowd"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_IC),
    ),
    "adverse_selection_insurance": GameConfig(
        name="Adverse Selection Insurance Game",
        description=(
            "An insurance market game with asymmetric information. Each player "
            "can reveal their private risk type for efficient pricing or hide "
            "it to exploit information asymmetry. Mutual revelation enables "
            "fair pricing. Hiding while the other reveals creates adverse "
            "selection profit. Tests screening and pooling dynamics."
        ),
        actions=["reveal_type", "hide_type"],
        game_type="matrix",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_ASI),
    ),
}

GAMES.update(BAYESIAN_GAMES)
