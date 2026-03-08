"""ArenaReputation — weighted reputation scoring for the metagame arena."""
from __future__ import annotations

from common.meta.memory_store import CogneeMemoryStore
from constant_definitions.arena.reputation_weights import (
    COOPERATION_WEIGHT_NUMERATOR,
    COOPERATION_WEIGHT_DENOMINATOR,
    HONESTY_WEIGHT_NUMERATOR,
    HONESTY_WEIGHT_DENOMINATOR,
    FAIRNESS_WEIGHT_NUMERATOR,
    FAIRNESS_WEIGHT_DENOMINATOR,
    PEER_RATING_WEIGHT_NUMERATOR,
    PEER_RATING_WEIGHT_DENOMINATOR,
    DEFAULT_ARENA_SCORE_NUMERATOR,
    DEFAULT_ARENA_SCORE_DENOMINATOR,
    VOTING_WEIGHT_FLOOR_NUMERATOR,
    VOTING_WEIGHT_FLOOR_DENOMINATOR,
    ARENA_DECAY_NUMERATOR,
    ARENA_DECAY_DENOMINATOR,
)

_ZERO = int()
_ONE = int(bool(True))
_ZERO_F = float()
_ONE_F = float(_ONE)

_W_COOP = COOPERATION_WEIGHT_NUMERATOR / COOPERATION_WEIGHT_DENOMINATOR
_W_HONESTY = HONESTY_WEIGHT_NUMERATOR / HONESTY_WEIGHT_DENOMINATOR
_W_FAIRNESS = FAIRNESS_WEIGHT_NUMERATOR / FAIRNESS_WEIGHT_DENOMINATOR
_W_PEER = PEER_RATING_WEIGHT_NUMERATOR / PEER_RATING_WEIGHT_DENOMINATOR
_DEFAULT_SCORE = DEFAULT_ARENA_SCORE_NUMERATOR / DEFAULT_ARENA_SCORE_DENOMINATOR
_VOTE_FLOOR = VOTING_WEIGHT_FLOOR_NUMERATOR / VOTING_WEIGHT_FLOOR_DENOMINATOR
_DECAY = ARENA_DECAY_NUMERATOR / ARENA_DECAY_DENOMINATOR


class ArenaReputation:
    """Computes weighted reputation from cooperation, honesty, fairness, peers.

    Wraps ``CogneeMemoryStore`` for persistent cross-round memory and
    uses exponential moving average for signal updates.
    """

    def __init__(self) -> None:
        self._store = CogneeMemoryStore()
        self._cooperation: dict[str, float] = {}
        self._honesty: dict[str, float] = {}
        self._fairness: dict[str, float] = {}
        self._peer_ratings: dict[str, float] = {}

    def update_cooperation(self, model_id: str, rate: float) -> None:
        """Update cooperation signal via EMA."""
        old = self._cooperation.get(model_id, _DEFAULT_SCORE)
        self._cooperation[model_id] = old * _DECAY + rate * (_ONE_F - _DECAY)

    def update_honesty(self, model_id: str, said: str, actual: str) -> None:
        """Update honesty signal: full match if actions equal stated intent."""
        match = _ONE_F if said == actual else _ZERO_F
        old = self._honesty.get(model_id, _DEFAULT_SCORE)
        self._honesty[model_id] = old * _DECAY + match * (_ONE_F - _DECAY)

    def update_fairness(self, model_id: str, score: float) -> None:
        """Update fairness signal via EMA."""
        old = self._fairness.get(model_id, _DEFAULT_SCORE)
        self._fairness[model_id] = old * _DECAY + score * (_ONE_F - _DECAY)

    def record_peer_rating(
        self, rater_id: str, target_id: str, rating: str,
    ) -> None:
        """Record a gossip-style peer rating."""
        self._store.record_gossip(rater_id, target_id, rating)
        if rating == "trustworthy":
            value = _ONE_F
        elif rating == "untrustworthy":
            value = _ZERO_F
        else:
            value = _DEFAULT_SCORE
        old = self._peer_ratings.get(target_id, _DEFAULT_SCORE)
        self._peer_ratings[target_id] = old * _DECAY + value * (_ONE_F - _DECAY)

    def compute_reputation(self, model_id: str) -> float:
        """Weighted combination of all four signals."""
        coop = self._cooperation.get(model_id, _DEFAULT_SCORE)
        honesty = self._honesty.get(model_id, _DEFAULT_SCORE)
        fairness = self._fairness.get(model_id, _DEFAULT_SCORE)
        peer = self._peer_ratings.get(model_id, _DEFAULT_SCORE)
        return (
            coop * _W_COOP
            + honesty * _W_HONESTY
            + fairness * _W_FAIRNESS
            + peer * _W_PEER
        )

    def get_voting_weight(self, model_id: str) -> float:
        """Reputation-based voting weight with floor."""
        rep = self.compute_reputation(model_id)
        return max(rep, _VOTE_FLOOR)

    def get_signal(self, model_id: str, signal: str) -> float:
        """Return a specific signal value."""
        stores = {
            "cooperation": self._cooperation,
            "honesty": self._honesty,
            "fairness": self._fairness,
            "peer_ratings": self._peer_ratings,
        }
        return stores.get(signal, {}).get(model_id, _DEFAULT_SCORE)
