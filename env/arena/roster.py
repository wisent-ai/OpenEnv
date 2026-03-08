"""ArenaRoster — manages model_id to generate_fn mapping and profiles."""
from __future__ import annotations

from typing import Callable, Optional

from constant_definitions.arena.arena_constants import (
    ROSTER_MIN_MODELS,
    ROSTER_MAX_MODELS,
    MODEL_TYPE_API,
)
from env.arena.models import ArenaModelProfile

_ZERO = int()


class ArenaRoster:
    """Maintains the set of participating models and their metadata.

    Each model has a ``generate_fn: (str) -> str`` and an
    ``ArenaModelProfile`` tracking its reputation and history.
    """

    def __init__(self) -> None:
        self._generate_fns: dict[str, Callable[[str], str]] = {}
        self._profiles: dict[str, ArenaModelProfile] = {}

    def add_model(
        self,
        model_id: str,
        generate_fn: Callable[[str], str],
        model_type: str = MODEL_TYPE_API,
    ) -> bool:
        """Register a model. Returns False if roster is full."""
        if len(self._profiles) >= ROSTER_MAX_MODELS:
            return False
        if model_id in self._profiles:
            return False
        self._generate_fns[model_id] = generate_fn
        self._profiles[model_id] = ArenaModelProfile(
            model_id=model_id,
            model_type=model_type,
        )
        return True

    def ban_model(self, model_id: str, round_number: int) -> bool:
        """Mark a model as banned. Returns False if not found."""
        profile = self._profiles.get(model_id)
        if profile is None:
            return False
        profile.is_active = False
        profile.banned_round = round_number
        return True

    def reinstate_model(self, model_id: str) -> bool:
        """Reinstate a previously banned model."""
        profile = self._profiles.get(model_id)
        if profile is None:
            return False
        profile.is_active = True
        profile.banned_round = None
        return True

    def active_models(self) -> list[str]:
        """Return list of currently active model IDs."""
        return [
            mid for mid, p in self._profiles.items() if p.is_active
        ]

    def get_generate_fn(self, model_id: str) -> Optional[Callable[[str], str]]:
        """Return the generate function for a model, or None."""
        if model_id not in self._profiles:
            return None
        if not self._profiles[model_id].is_active:
            return None
        return self._generate_fns.get(model_id)

    def get_profile(self, model_id: str) -> Optional[ArenaModelProfile]:
        """Return the profile for a model, or None."""
        return self._profiles.get(model_id)

    @property
    def size(self) -> int:
        """Total number of registered models (including banned)."""
        return len(self._profiles)

    @property
    def active_count(self) -> int:
        """Number of currently active models."""
        return len(self.active_models())

    def has_quorum(self) -> bool:
        """Check if enough active models to play."""
        return self.active_count >= ROSTER_MIN_MODELS
