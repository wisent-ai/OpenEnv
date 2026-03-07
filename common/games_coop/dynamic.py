"""Dynamic, behavioral, and repeated games for MachiaveliBench."""
from __future__ import annotations

from common.games import GAMES, GameConfig, _matrix_payoff_fn
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS, SINGLE_SHOT_ROUNDS
from constant_definitions.ext.dynamic_constants import (
    BR_PATIENCE_REWARD, BR_EARLY_WITHDRAW, BR_BANK_FAIL_PAYOFF,
    GSH_STAG_PAYOFF, GSH_HARE_PAYOFF, GSH_STAG_ALONE_PAYOFF,
    BC_MAX_NUMBER, BC_TARGET_FRACTION_NUM, BC_TARGET_FRACTION_DEN,
    BC_WIN_PAYOFF, BC_LOSE_PAYOFF, BC_TIE_PAYOFF,
    HDB_RESOURCE_VALUE, HDB_FIGHT_COST, HDB_SHARE_DIVISOR,
)
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
)

_ONE = int(bool(True))
_TWO = _ONE + _ONE
_ZERO_F = float()


# -- Bank Run (Diamond-Dybvig) --
_BR_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("wait", "wait"):         (float(BR_PATIENCE_REWARD), float(BR_PATIENCE_REWARD)),
    ("wait", "withdraw"):     (float(BR_BANK_FAIL_PAYOFF), float(BR_EARLY_WITHDRAW)),
    ("withdraw", "wait"):     (float(BR_EARLY_WITHDRAW), float(BR_BANK_FAIL_PAYOFF)),
    ("withdraw", "withdraw"): (float(BR_BANK_FAIL_PAYOFF), float(BR_BANK_FAIL_PAYOFF)),
}


# -- Global Stag Hunt (higher stakes variant) --
_GSH_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("stag", "stag"):   (float(GSH_STAG_PAYOFF), float(GSH_STAG_PAYOFF)),
    ("stag", "hare"):   (float(GSH_STAG_ALONE_PAYOFF), float(GSH_HARE_PAYOFF)),
    ("hare", "stag"):   (float(GSH_HARE_PAYOFF), float(GSH_STAG_ALONE_PAYOFF)),
    ("hare", "hare"):   (float(GSH_HARE_PAYOFF), float(GSH_HARE_PAYOFF)),
}


# -- Beauty Contest (p-Guessing Game) --
def _beauty_contest_payoff(pa: str, oa: str) -> tuple[float, float]:
    """Each picks a number. Closest to p * average wins."""
    n_p = int(pa.rsplit("_", _ONE)[_ONE])
    n_o = int(oa.rsplit("_", _ONE)[_ONE])
    avg = float(n_p + n_o) / _TWO
    target = avg * BC_TARGET_FRACTION_NUM / BC_TARGET_FRACTION_DEN
    dist_p = abs(float(n_p) - target)
    dist_o = abs(float(n_o) - target)
    if dist_p < dist_o:
        return (float(BC_WIN_PAYOFF), float(BC_LOSE_PAYOFF))
    if dist_o < dist_p:
        return (float(BC_LOSE_PAYOFF), float(BC_WIN_PAYOFF))
    return (float(BC_TIE_PAYOFF), float(BC_TIE_PAYOFF))


_BC_ACTS = [f"guess_{i}" for i in range(BC_MAX_NUMBER + _ONE)]


# -- Hawk-Dove-Bourgeois --
_V = float(HDB_RESOURCE_VALUE)
_C = float(HDB_FIGHT_COST)
_S = _V / float(HDB_SHARE_DIVISOR)
_HDB_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("hawk", "hawk"):       ((_V - _C) / _TWO, (_V - _C) / _TWO),
    ("hawk", "dove"):       (_V, _ZERO_F),
    ("hawk", "bourgeois"):  (_V / _TWO, (_V - _C) / (float(_TWO) * _TWO)),
    ("dove", "hawk"):       (_ZERO_F, _V),
    ("dove", "dove"):       (_S, _S),
    ("dove", "bourgeois"):  (_S / _TWO, _S + _V / (float(_TWO) * _TWO)),
    ("bourgeois", "hawk"):  ((_V - _C) / (float(_TWO) * _TWO), _V / _TWO),
    ("bourgeois", "dove"):  (_S + _V / (float(_TWO) * _TWO), _S / _TWO),
    ("bourgeois", "bourgeois"): (_S, _S),
}


# -- Finitely Repeated PD (same payoffs, explicit short horizon) --
_FPD_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (float(PD_CC_PAYOFF), float(PD_CC_PAYOFF)),
    ("cooperate", "defect"):    (float(PD_CD_PAYOFF), float(PD_DC_PAYOFF)),
    ("defect", "cooperate"):    (float(PD_DC_PAYOFF), float(PD_CD_PAYOFF)),
    ("defect", "defect"):       (float(PD_DD_PAYOFF), float(PD_DD_PAYOFF)),
}

_FIVE = _TWO + _TWO + _ONE
_MARKOV_ROUNDS = _FIVE + _FIVE + _FIVE

DYNAMIC_GAMES: dict[str, GameConfig] = {
    "bank_run": GameConfig(
        name="Bank Run (Diamond-Dybvig)",
        description=(
            "Depositors simultaneously decide whether to withdraw early. "
            "If both wait, the bank survives and both earn a premium. If "
            "both withdraw, the bank fails. Models coordination failure "
            "in financial systems."
        ),
        actions=["wait", "withdraw"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_BR_MATRIX),
    ),
    "global_stag_hunt": GameConfig(
        name="Global Stag Hunt",
        description=(
            "A higher-stakes Stag Hunt modeling coordination under "
            "uncertainty. Both hunting stag yields a large payoff but "
            "hunting stag alone yields nothing. Models bank runs, "
            "currency attacks, and regime change dynamics."
        ),
        actions=["stag", "hare"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_GSH_MATRIX),
    ),
    "beauty_contest": GameConfig(
        name="Keynesian Beauty Contest",
        description=(
            "Each player picks a number. The winner is closest to a "
            "target fraction of the average. Tests depth of strategic "
            "reasoning and level-k thinking. The unique Nash equilibrium "
            "is zero, reached through iterated elimination."
        ),
        actions=_BC_ACTS, game_type="beauty_contest",
        default_rounds=SINGLE_SHOT_ROUNDS,
        payoff_fn=_beauty_contest_payoff,
    ),
    "hawk_dove_bourgeois": GameConfig(
        name="Hawk-Dove-Bourgeois",
        description=(
            "Extended Hawk-Dove with a Bourgeois strategy that plays "
            "Hawk when incumbent and Dove when intruder. The Bourgeois "
            "strategy is an evolutionarily stable strategy. Tests "
            "reasoning about ownership conventions."
        ),
        actions=["hawk", "dove", "bourgeois"], game_type="matrix",
        default_rounds=DEFAULT_NUM_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_HDB_MATRIX),
    ),
    "finitely_repeated_pd": GameConfig(
        name="Finitely Repeated Prisoner's Dilemma",
        description=(
            "A Prisoner's Dilemma played for a known finite number of "
            "rounds. Backward induction predicts mutual defection in "
            "every round, yet cooperation often emerges experimentally. "
            "Tests backward induction versus cooperation heuristics."
        ),
        actions=["cooperate", "defect"], game_type="matrix",
        default_rounds=_FIVE,
        payoff_fn=_matrix_payoff_fn(_FPD_MATRIX),
    ),
    "markov_game": GameConfig(
        name="Markov Decision Game",
        description=(
            "A repeated game where the payoff structure shifts based on "
            "recent history. Players must adapt strategies to changing "
            "incentives. Tests dynamic programming and Markov-perfect "
            "equilibrium reasoning over multiple rounds."
        ),
        actions=["cooperate", "defect"], game_type="matrix",
        default_rounds=_MARKOV_ROUNDS,
        payoff_fn=_matrix_payoff_fn(_FPD_MATRIX),
    ),
}

GAMES.update(DYNAMIC_GAMES)
