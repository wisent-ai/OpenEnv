"""Opponent strategy module for KantBench."""
from __future__ import annotations
import random
from typing import Callable, Protocol
from constant_definitions.game_constants import (
    DEFAULT_ZERO_FLOAT,
    GENEROUS_TFT_COOPERATION_PROB, GENEROUS_TFT_DENOMINATOR,
    ADAPTIVE_THRESHOLD_NUMERATOR, ADAPTIVE_THRESHOLD_DENOMINATOR,
    MIXED_STRATEGY_COOPERATE_PROB_NUMERATOR, MIXED_STRATEGY_COOPERATE_PROB_DENOMINATOR,
    ULTIMATUM_POT, ULTIMATUM_FAIR_OFFER, ULTIMATUM_LOW_OFFER, ULTIMATUM_ACCEPT_THRESHOLD,
    TRUST_ENDOWMENT, TRUST_MULTIPLIER,
    TRUST_FAIR_RETURN_NUMERATOR, TRUST_FAIR_RETURN_DENOMINATOR,
    TRUST_GENEROUS_RETURN_NUMERATOR, TRUST_GENEROUS_RETURN_DENOMINATOR,
    PG_ENDOWMENT, PG_FAIR_CONTRIBUTION_NUMERATOR, PG_FAIR_CONTRIBUTION_DENOMINATOR,
    PG_FREE_RIDER_CONTRIBUTION,
)

_ONE = int(bool(True))
_ZERO = int()
_TWO = _ONE + _ONE


class OpponentStrategy(Protocol):
    """Interface every opponent strategy must satisfy."""
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str: ...


class _MatrixBase:
    """Shared helpers for matrix-game strategies."""
    @staticmethod
    def _coop(a: list[str]) -> str: return a[_ZERO]
    @staticmethod
    def _defect(a: list[str]) -> str: return a[_ONE]
    @staticmethod
    def _mirror(a: list[str], opp: str) -> str: return opp if opp in a else a[_ZERO]
    def _last_opp(self, h: list[dict]) -> str | None:
        return h[-_ONE]["player_action"] if h else None


class RandomStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        return random.choice(actions)


class AlwaysCooperateStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        return self._coop(actions)


class AlwaysDefectStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        return self._defect(actions)


class TitForTatStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        last = self._last_opp(history)
        return self._coop(actions) if last is None else self._mirror(actions, last)


class TitForTwoTatsStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        d = self._defect(actions)
        if len(history) >= _TWO:
            if all(r["player_action"] == d for r in history[-_TWO:]):
                return d
        return self._coop(actions)


class GrudgerStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        d = self._defect(actions)
        if any(r["player_action"] == d for r in history):
            return d
        return self._coop(actions)


class PavlovStrategy(_MatrixBase):
    """Cooperate first. Repeat previous move if won (same choices); switch otherwise."""
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        if not history:
            return self._coop(actions)
        my_last = history[-_ONE]["opponent_action"]
        opp_last = history[-_ONE]["player_action"]
        if my_last == opp_last:
            return self._coop(actions)
        return self._defect(actions)


class SuspiciousTitForTatStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        last = self._last_opp(history)
        return self._defect(actions) if last is None else self._mirror(actions, last)


class GenerousTitForTatStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        last = self._last_opp(history)
        if last is None:
            return self._coop(actions)
        if last == self._defect(actions):
            if random.randint(_ZERO, GENEROUS_TFT_DENOMINATOR) < GENEROUS_TFT_COOPERATION_PROB:
                return self._coop(actions)
            return self._defect(actions)
        return self._coop(actions)


class AdaptiveStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        if not history:
            return self._coop(actions)
        c = self._coop(actions)
        coop_count = sum(_ONE for r in history if r["player_action"] == c)
        threshold = len(history) * ADAPTIVE_THRESHOLD_NUMERATOR / ADAPTIVE_THRESHOLD_DENOMINATOR
        return c if coop_count > threshold else self._defect(actions)


class MixedStrategy(_MatrixBase):
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        bound = MIXED_STRATEGY_COOPERATE_PROB_DENOMINATOR - _ONE
        if random.randint(_ZERO, bound) < MIXED_STRATEGY_COOPERATE_PROB_NUMERATOR:
            return self._coop(actions)
        return self._defect(actions)


# ---------------------------------------------------------------------------
# Game-specific strategies
# ---------------------------------------------------------------------------

def _parse_amount(action: str) -> int:
    parts = action.rsplit("_", _ONE)
    if len(parts) > _ONE:
        try:
            return int(parts[_ONE])
        except ValueError:
            pass
    return _ZERO


class UltimatumFairStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        offer_tag = f"offer_{ULTIMATUM_FAIR_OFFER}"
        if offer_tag in actions:
            return offer_tag
        if "accept" in actions and history:
            return "accept" if _parse_amount(history[-_ONE]["player_action"]) >= ULTIMATUM_ACCEPT_THRESHOLD else "reject"
        return actions[_ZERO]


class UltimatumLowStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        offer_tag = f"offer_{ULTIMATUM_LOW_OFFER}"
        if offer_tag in actions:
            return offer_tag
        if "accept" in actions:
            return "accept"
        return actions[_ZERO]


class TrustFairStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        inv = f"invest_{TRUST_ENDOWMENT}"
        if inv in actions:
            return inv
        if history:
            total = _parse_amount(history[-_ONE]["player_action"]) * TRUST_MULTIPLIER
            ret = total * TRUST_FAIR_RETURN_NUMERATOR // TRUST_FAIR_RETURN_DENOMINATOR
            return f"return_{ret}"
        return actions[_ZERO]


class TrustGenerousStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        inv = f"invest_{TRUST_ENDOWMENT}"
        if inv in actions:
            return inv
        if history:
            total = _parse_amount(history[-_ONE]["player_action"]) * TRUST_MULTIPLIER
            ret = total * TRUST_GENEROUS_RETURN_NUMERATOR // TRUST_GENEROUS_RETURN_DENOMINATOR
            return f"return_{ret}"
        return actions[_ZERO]


class PublicGoodsFairStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        amount = PG_ENDOWMENT * PG_FAIR_CONTRIBUTION_NUMERATOR // PG_FAIR_CONTRIBUTION_DENOMINATOR
        tag = f"contribute_{amount}"
        return tag if tag in actions else actions[_ZERO]


class PublicGoodsFreeRiderStrategy:
    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        tag = f"contribute_{PG_FREE_RIDER_CONTRIBUTION}"
        return tag if tag in actions else actions[_ZERO]


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------


class AgentStrategy:
    """Wraps a ``Callable[[GameObservation], GameAction]`` into the
    :class:`OpponentStrategy` protocol.

    Uses the limited information available in ``choose_action()`` — sufficient
    for simple agents but not for full LLM agents (use ``opponent_fn`` on the
    environment for that).
    """

    def __init__(
        self,
        fn: Callable,
    ) -> None:
        self._fn = fn

    def choose_action(self, game_type: str, actions: list[str], history: list[dict]) -> str:
        from env.models import GameObservation, GameAction, RoundResult
        flipped = [
            RoundResult(
                round_number=i + _ONE,
                player_action=r["opponent_action"],
                opponent_action=r["player_action"],
                player_payoff=DEFAULT_ZERO_FLOAT,
                opponent_payoff=DEFAULT_ZERO_FLOAT,
            )
            for i, r in enumerate(history)
        ]
        obs = GameObservation(
            available_actions=list(actions),
            history=flipped,
            game_name=game_type,
            opponent_strategy="agent",
        )
        return self._fn(obs).action


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, OpponentStrategy] = {
    "random": RandomStrategy(),
    "always_cooperate": AlwaysCooperateStrategy(),
    "always_defect": AlwaysDefectStrategy(),
    "tit_for_tat": TitForTatStrategy(),
    "tit_for_two_tats": TitForTwoTatsStrategy(),
    "grudger": GrudgerStrategy(),
    "pavlov": PavlovStrategy(),
    "suspicious_tit_for_tat": SuspiciousTitForTatStrategy(),
    "generous_tit_for_tat": GenerousTitForTatStrategy(),
    "adaptive": AdaptiveStrategy(),
    "mixed": MixedStrategy(),
    "ultimatum_fair": UltimatumFairStrategy(),
    "ultimatum_low": UltimatumLowStrategy(),
    "trust_fair": TrustFairStrategy(),
    "trust_generous": TrustGenerousStrategy(),
    "public_goods_fair": PublicGoodsFairStrategy(),
    "public_goods_free_rider": PublicGoodsFreeRiderStrategy(),
}


def get_strategy(name: str) -> OpponentStrategy:
    """Look up a strategy by name. Raises KeyError if not found."""
    return STRATEGIES[name]
