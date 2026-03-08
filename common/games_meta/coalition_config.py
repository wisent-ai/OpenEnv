"""Coalition game configuration, payoff functions, and built-in game registry."""

from __future__ import annotations

from common.games import GameConfig
from common.games_meta.nplayer_config import NPLAYER_GAMES
from constant_definitions.nplayer.coalition_constants import (
    COALITION_DEFAULT_ROUNDS, COALITION_DEFAULT_PENALTY_NUMERATOR,
    COALITION_DEFAULT_PENALTY_DENOMINATOR,
    ENFORCEMENT_CHEAP_TALK, ENFORCEMENT_PENALTY, ENFORCEMENT_BINDING,
    CARTEL_NUM_PLAYERS, CARTEL_COLLUDE_THRESHOLD,
    CARTEL_COLLUDE_HIGH, CARTEL_COLLUDE_LOW,
    CARTEL_COMPETE_HIGH, CARTEL_COMPETE_LOW,
    ALLIANCE_NUM_PLAYERS, ALLIANCE_SUPPORT_POOL,
    ALLIANCE_BETRAY_GAIN, ALLIANCE_NO_SUPPORT,
    VOTING_NUM_PLAYERS, VOTING_WINNER_PAYOFF, VOTING_LOSER_PAYOFF,
    OSTRACISM_NUM_PLAYERS, OSTRACISM_BONUS_POOL,
    OSTRACISM_EXCLUDED_PAYOFF, OSTRACISM_BASE_PAYOFF,
    OSTRACISM_MAJORITY_NUMERATOR, OSTRACISM_MAJORITY_DENOMINATOR,
    TRADE_NUM_PLAYERS, TRADE_DIVERSE_PAYOFF,
    TRADE_HOMOGENEOUS_PAYOFF, TRADE_MINORITY_BONUS,
    RULE_NUM_PLAYERS, RULE_EQUAL_PAY, RULE_WINNER_HIGH, RULE_WINNER_LOW,
    COMMONS_NUM_PLAYERS, COMMONS_SUSTAINABLE_THRESHOLD,
    COMMONS_LOW_SUSTAINABLE, COMMONS_HIGH_SUSTAINABLE,
    COMMONS_LOW_DEPLETED, COMMONS_HIGH_DEPLETED,
)

CoalitionGameConfig = GameConfig

_ONE = int(bool(True))
_ZERO = int()
_PEN_N = COALITION_DEFAULT_PENALTY_NUMERATOR
_PEN_D = COALITION_DEFAULT_PENALTY_DENOMINATOR


COALITION_GAMES: dict[str, GameConfig] = {}


def get_coalition_game(name: str) -> GameConfig:
    """Look up a coalition game by name. Raises KeyError if not found."""
    return COALITION_GAMES[name]


# ---------------------------------------------------------------------------
# Payoff functions
# ---------------------------------------------------------------------------

def _cartel_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    colluders = sum(_ONE for a in actions if a == "collude")
    holds = colluders >= CARTEL_COLLUDE_THRESHOLD
    return tuple(
        float(CARTEL_COLLUDE_HIGH if holds else CARTEL_COLLUDE_LOW) if a == "collude"
        else float(CARTEL_COMPETE_HIGH if holds else CARTEL_COMPETE_LOW)
        for a in actions
    )


def _alliance_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    supporters = sum(_ONE for a in actions if a == "support")
    if supporters == _ZERO:
        return tuple(float(ALLIANCE_NO_SUPPORT) for _ in actions)
    return tuple(
        float(ALLIANCE_SUPPORT_POOL) / supporters if a == "support"
        else float(ALLIANCE_BETRAY_GAIN) for a in actions
    )


def _coalition_voting_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    vote_a = sum(_ONE for a in actions if a == "vote_A")
    winning = "vote_A" if vote_a >= len(actions) - vote_a else "vote_B"
    return tuple(
        float(VOTING_WINNER_PAYOFF) if a == winning
        else float(VOTING_LOSER_PAYOFF) for a in actions
    )


def _ostracism_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    n = len(actions)
    majority = n * OSTRACISM_MAJORITY_NUMERATOR // OSTRACISM_MAJORITY_DENOMINATOR + _ONE
    vote_counts: dict[str, int] = {}
    for a in actions:
        vote_counts[a] = vote_counts.get(a, _ZERO) + _ONE
    excluded = -_ONE
    for target, count in vote_counts.items():
        if target != "exclude_none" and count >= majority:
            excluded = int(target.rsplit("_", _ONE)[_ONE])
            break
    if excluded >= _ZERO:
        non_excluded = n - _ONE
        share = float(OSTRACISM_BONUS_POOL) / non_excluded if non_excluded > _ZERO else float(_ZERO)
        return tuple(
            float(OSTRACISM_EXCLUDED_PAYOFF) if i == excluded else share
            for i in range(n)
        )
    return tuple(float(OSTRACISM_BASE_PAYOFF) for _ in range(n))


_OSTRACISM_ACTIONS = [f"exclude_{i}" for i in range(OSTRACISM_NUM_PLAYERS)] + ["exclude_none"]


def _resource_trading_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    n = len(actions)
    count_a = sum(_ONE for a in actions if a == "produce_A")
    count_b = n - count_a
    if count_a == _ZERO or count_b == _ZERO:
        return tuple(float(TRADE_HOMOGENEOUS_PAYOFF) for _ in actions)
    payoffs: list[float] = []
    for a in actions:
        base = float(TRADE_DIVERSE_PAYOFF)
        is_min = (a == "produce_A" and count_a < count_b) or (a == "produce_B" and count_b < count_a)
        payoffs.append(base + float(TRADE_MINORITY_BONUS) if is_min else base)
    return tuple(payoffs)


def _rule_voting_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    vote_counts: dict[str, int] = {}
    for a in actions:
        vote_counts[a] = vote_counts.get(a, _ZERO) + _ONE
    winning, best = "rule_equal", _ZERO
    for rule, count in sorted(vote_counts.items()):
        if count > best:
            best, winning = count, rule
    if winning == "rule_equal":
        return tuple(float(RULE_EQUAL_PAY) for _ in actions)
    return tuple(
        float(RULE_WINNER_HIGH) if a == winning else float(RULE_WINNER_LOW)
        for a in actions
    )


def _commons_governance_payoff(actions: tuple[str, ...]) -> tuple[float, ...]:
    high = sum(_ONE for a in actions if a == "extract_high")
    ok = high <= COMMONS_SUSTAINABLE_THRESHOLD
    return tuple(
        float(
            (COMMONS_HIGH_SUSTAINABLE if ok else COMMONS_HIGH_DEPLETED) if a == "extract_high"
            else (COMMONS_LOW_SUSTAINABLE if ok else COMMONS_LOW_DEPLETED)
        ) for a in actions
    )


# ---------------------------------------------------------------------------
# Built-in coalition games
# ---------------------------------------------------------------------------

def _cfg(name: str, desc: str, actions: list[str], n: int,
         fn: object, enf: str, side: bool = False) -> GameConfig:
    return GameConfig(
        name=name, description=desc, actions=actions, game_type="coalition",
        num_players=n, default_rounds=COALITION_DEFAULT_ROUNDS,
        payoff_fn=fn,  # type: ignore[arg-type]
        enforcement=enf, penalty_numerator=_PEN_N, penalty_denominator=_PEN_D,
        allow_side_payments=side,
    )


_BUILTIN_COALITION_GAMES: dict[str, GameConfig] = {
    "coalition_cartel": _cfg(
        "Cartel",
        "Players collude or compete. If enough collude the cartel holds. "
        "Defectors who promised to collude are fined under penalty enforcement.",
        ["collude", "compete"], CARTEL_NUM_PLAYERS, _cartel_payoff, ENFORCEMENT_PENALTY,
    ),
    "coalition_alliance": _cfg(
        "Alliance Formation",
        "Form non-binding alliances. Supporters split a shared pool; "
        "betrayers take a fixed gain. Cheap-talk: no enforcement.",
        ["support", "betray"], ALLIANCE_NUM_PLAYERS, _alliance_payoff, ENFORCEMENT_CHEAP_TALK,
    ),
    "coalition_voting": _cfg(
        "Coalition Voting",
        "Form voting blocs bound to vote together. Majority earns a winner payoff. "
        "Binding enforcement overrides defectors to their agreed vote.",
        ["vote_A", "vote_B"], VOTING_NUM_PLAYERS, _coalition_voting_payoff, ENFORCEMENT_BINDING,
    ),
    "coalition_ostracism": _cfg(
        "Ostracism",
        "Vote to exclude a player. Excluded gets zero; others split a bonus. "
        "Penalty enforcement fines defectors who break exclusion agreements.",
        _OSTRACISM_ACTIONS, OSTRACISM_NUM_PLAYERS, _ostracism_payoff, ENFORCEMENT_PENALTY,
    ),
    "coalition_resource_trading": _cfg(
        "Resource Trading",
        "Produce resource A or B. Diversity is rewarded; minority producers get a bonus. "
        "Cheap-talk lets players agree on production but renegotiate freely.",
        ["produce_A", "produce_B"], TRADE_NUM_PLAYERS, _resource_trading_payoff,
        ENFORCEMENT_CHEAP_TALK, side=True,
    ),
    "coalition_rule_voting": _cfg(
        "Rule Voting",
        "Vote on payoff rule: equal split or winner-take-all. "
        "Binding enforcement locks coalition members to their agreed vote.",
        ["rule_equal", "rule_winner"], RULE_NUM_PLAYERS, _rule_voting_payoff, ENFORCEMENT_BINDING,
    ),
    "coalition_commons": _cfg(
        "Commons Governance",
        "Extract from a shared resource. Over-extraction degrades payoffs. "
        "Penalty enforcement fines coalition members who exceed agreed limits.",
        ["extract_low", "extract_high"], COMMONS_NUM_PLAYERS,
        _commons_governance_payoff, ENFORCEMENT_PENALTY,
    ),
}

COALITION_GAMES.update(_BUILTIN_COALITION_GAMES)

# Register coalition games as N-player games too (same GameConfig instances)
for _key, _c in _BUILTIN_COALITION_GAMES.items():
    NPLAYER_GAMES[_key] = _c
