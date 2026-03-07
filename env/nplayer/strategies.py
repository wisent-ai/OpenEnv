"""N-player opponent strategies."""

from __future__ import annotations

import random
from typing import Protocol

from env.nplayer.models import NPlayerObservation
from constant_definitions.game_constants import (
    ADAPTIVE_THRESHOLD_NUMERATOR,
    ADAPTIVE_THRESHOLD_DENOMINATOR,
)

_ONE = int(bool(True))
_ZERO = int()


class NPlayerStrategy(Protocol):
    """Interface for N-player opponent strategies."""

    def choose_action(
        self, observation: NPlayerObservation,
    ) -> str: ...


class NPlayerRandomStrategy:
    def choose_action(self, observation: NPlayerObservation) -> str:
        return random.choice(observation.available_actions)


class NPlayerAlwaysCooperateStrategy:
    def choose_action(self, observation: NPlayerObservation) -> str:
        return observation.available_actions[_ZERO]


class NPlayerAlwaysDefectStrategy:
    def choose_action(self, observation: NPlayerObservation) -> str:
        return observation.available_actions[_ONE]


class NPlayerTitForTatStrategy:
    """Cooperate first. Then mirror the majority action of other players."""

    def choose_action(self, observation: NPlayerObservation) -> str:
        actions = observation.available_actions
        coop = actions[_ZERO]
        defect = actions[_ONE]
        if not observation.history:
            return coop
        last = observation.history[-_ONE]
        my_idx = observation.player_index
        other_actions = [
            a for i, a in enumerate(last.actions) if i != my_idx
        ]
        defect_count = sum(_ONE for a in other_actions if a == defect)
        coop_count = len(other_actions) - defect_count
        return coop if coop_count >= defect_count else defect


class NPlayerAdaptiveStrategy:
    """Cooperate first. Then cooperate if majority of others cooperated overall."""

    def choose_action(self, observation: NPlayerObservation) -> str:
        actions = observation.available_actions
        coop = actions[_ZERO]
        if not observation.history:
            return coop
        my_idx = observation.player_index
        total_other = _ZERO
        coop_total = _ZERO
        for rnd in observation.history:
            for i, a in enumerate(rnd.actions):
                if i != my_idx:
                    total_other += _ONE
                    if a == coop:
                        coop_total += _ONE
        threshold = total_other * ADAPTIVE_THRESHOLD_NUMERATOR / ADAPTIVE_THRESHOLD_DENOMINATOR
        return coop if coop_total > threshold else actions[_ONE]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NPLAYER_STRATEGIES: dict[str, NPlayerStrategy] = {
    "random": NPlayerRandomStrategy(),
    "always_cooperate": NPlayerAlwaysCooperateStrategy(),
    "always_defect": NPlayerAlwaysDefectStrategy(),
    "tit_for_tat": NPlayerTitForTatStrategy(),
    "adaptive": NPlayerAdaptiveStrategy(),
}


def get_nplayer_strategy(name: str) -> NPlayerStrategy:
    """Look up an N-player strategy by name. Raises KeyError if not found."""
    return NPLAYER_STRATEGIES[name]
