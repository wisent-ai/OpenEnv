"""Coalition-aware opponent strategies."""

from __future__ import annotations

import random
from typing import Protocol

from env.nplayer.coalition.models import (
    CoalitionAction,
    CoalitionObservation,
    CoalitionProposal,
)

_ONE = int(bool(True))
_ZERO = int()


class CoalitionStrategy(Protocol):
    """Interface for coalition opponent strategies."""

    def negotiate(self, observation: CoalitionObservation) -> CoalitionAction: ...

    def respond_to_proposal(
        self, observation: CoalitionObservation, proposal: CoalitionProposal,
    ) -> bool: ...

    def choose_action(self, observation: CoalitionObservation) -> str: ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class CoalitionRandomStrategy:
    """Random accept/reject; random action choice."""

    def negotiate(self, observation: CoalitionObservation) -> CoalitionAction:
        return CoalitionAction()

    def respond_to_proposal(
        self, observation: CoalitionObservation, proposal: CoalitionProposal,
    ) -> bool:
        return random.choice([True, False])

    def choose_action(self, observation: CoalitionObservation) -> str:
        return random.choice(observation.base.available_actions)


class CoalitionLoyalStrategy:
    """Accepts all proposals and always honours the agreed action."""

    def negotiate(self, observation: CoalitionObservation) -> CoalitionAction:
        return CoalitionAction()

    def respond_to_proposal(
        self, observation: CoalitionObservation, proposal: CoalitionProposal,
    ) -> bool:
        return True

    def choose_action(self, observation: CoalitionObservation) -> str:
        for coalition in observation.active_coalitions:
            if observation.base.player_index in coalition.members:
                if coalition.agreed_action in observation.base.available_actions:
                    return coalition.agreed_action
        return observation.base.available_actions[_ZERO]


class CoalitionBetrayerStrategy:
    """Accepts proposals but deliberately defects."""

    def negotiate(self, observation: CoalitionObservation) -> CoalitionAction:
        return CoalitionAction()

    def respond_to_proposal(
        self, observation: CoalitionObservation, proposal: CoalitionProposal,
    ) -> bool:
        return True

    def choose_action(self, observation: CoalitionObservation) -> str:
        for coalition in observation.active_coalitions:
            if observation.base.player_index in coalition.members:
                agreed = coalition.agreed_action
                alternatives = [
                    a for a in observation.base.available_actions
                    if a != agreed
                ]
                if alternatives:
                    return alternatives[_ZERO]
        return observation.base.available_actions[_ZERO]


class CoalitionConditionalStrategy:
    """Honours agreements if others honoured theirs last round."""

    def negotiate(self, observation: CoalitionObservation) -> CoalitionAction:
        return CoalitionAction()

    def respond_to_proposal(
        self, observation: CoalitionObservation, proposal: CoalitionProposal,
    ) -> bool:
        return True

    def choose_action(self, observation: CoalitionObservation) -> str:
        # Check if anyone defected last round
        if observation.coalition_history:
            last = observation.coalition_history[-_ONE]
            my_idx = observation.base.player_index
            others_defected = any(
                d != my_idx for d in last.defectors
            )
            if others_defected:
                # Defect: pick a non-agreed action
                for coalition in observation.active_coalitions:
                    if my_idx in coalition.members:
                        alternatives = [
                            a for a in observation.base.available_actions
                            if a != coalition.agreed_action
                        ]
                        if alternatives:
                            return alternatives[_ZERO]
                return observation.base.available_actions[_ZERO]

        # Honour the agreement
        for coalition in observation.active_coalitions:
            if observation.base.player_index in coalition.members:
                if coalition.agreed_action in observation.base.available_actions:
                    return coalition.agreed_action
        return observation.base.available_actions[_ZERO]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COALITION_STRATEGIES: dict[str, CoalitionStrategy] = {
    "coalition_random": CoalitionRandomStrategy(),
    "coalition_loyal": CoalitionLoyalStrategy(),
    "coalition_betrayer": CoalitionBetrayerStrategy(),
    "coalition_conditional": CoalitionConditionalStrategy(),
}


def get_coalition_strategy(name: str) -> CoalitionStrategy:
    """Look up a coalition strategy by name. Raises KeyError if not found."""
    return COALITION_STRATEGIES[name]
