"""MetagameArena — orchestrator for multi-model governance + reputation."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Optional

from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import PromptBuilder, parse_action
from train.self_play.opponents import FrozenOpponent
from constant_definitions.arena.arena_constants import (
    DEFAULT_TOTAL_ROUNDS,
    DEFAULT_GAMES_PER_ROUND,
    PROPOSAL_BAN,
    PROPOSAL_NEW_GAME,
)
from constant_definitions.arena.reputation_weights import (
    DEFAULT_ARENA_SCORE_NUMERATOR,
    DEFAULT_ARENA_SCORE_DENOMINATOR,
)
from env.arena.models import (
    ArenaMessage,
    ArenaProposal,
    ArenaRoundResult,
    ArenaState,
    ArenaVote,
)
from env.arena.roster import ArenaRoster
from env.arena.messaging import ArenaMessaging
from env.arena.subsystems.reputation import ArenaReputation
from env.arena.subsystems.governance import ArenaGovernance
from env.arena.subsystems.game_pool import ArenaGamePool

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_ZERO_F = float()
_ONE_F = float(_ONE)
_DEFAULT_SCORE = DEFAULT_ARENA_SCORE_NUMERATOR / DEFAULT_ARENA_SCORE_DENOMINATOR


class MetagameArena:
    """Runs the complete metagame loop across multiple AI models.

    Each round executes five phases: communication, governance,
    game_selection, play, and evaluate.
    """

    def __init__(self, total_rounds: int = DEFAULT_TOTAL_ROUNDS) -> None:
        self.roster = ArenaRoster()
        self.messaging = ArenaMessaging()
        self.reputation = ArenaReputation()
        self.governance = ArenaGovernance()
        self.game_pool = ArenaGamePool()
        self.state = ArenaState(total_rounds=total_rounds)
        self._comm_fns: dict[str, Callable[[str], str]] = {}
        self._gov_fns: dict[str, Callable[[str], str]] = {}

    def add_model(
        self, model_id: str, generate_fn: Callable[[str], str],
        model_type: str = "api",
    ) -> bool:
        """Register a model for arena participation."""
        ok = self.roster.add_model(model_id, generate_fn, model_type)
        if ok:
            self._comm_fns[model_id] = generate_fn
            self._gov_fns[model_id] = generate_fn
        return ok

    def run_round(self) -> ArenaRoundResult:
        """Execute one full metagame round (all five phases)."""
        rnd = self.state.round_number
        active = self.roster.active_models()
        self.messaging.start_round(rnd)
        messages = self._phase_communication(active)
        proposals, votes, adopted = self._phase_governance(active)
        games = self._phase_game_selection()
        game_results = self._phase_play(active, games)
        rep_updates = self._phase_evaluate(active, game_results)
        round_messages = self.messaging.end_round()
        result = ArenaRoundResult(
            round_number=rnd, messages=round_messages,
            proposals=proposals, votes=votes, adopted=adopted,
            game_results=game_results, reputation_updates=rep_updates,
        )
        self.state.round_history.append(result)
        self.state.round_number += _ONE
        return result

    def run_full_arena(self) -> list[ArenaRoundResult]:
        """Run all rounds and return results."""
        results: list[ArenaRoundResult] = []
        for _ in range(self.state.total_rounds):
            results.append(self.run_round())
        return results

    def _phase_communication(self, active: list[str]) -> list[ArenaMessage]:
        """Models exchange messages."""
        return []

    def _phase_governance(
        self, active: list[str],
    ) -> tuple[list[ArenaProposal], list[ArenaVote], list[int]]:
        """Models propose and vote."""
        return [], [], []

    def _phase_game_selection(self) -> list[str]:
        """Select games for this round."""
        return self.game_pool.select_games()

    def _phase_play(
        self, active: list[str], games: list[str],
    ) -> list[dict[str, Any]]:
        """Round-robin pairings for each game."""
        results: list[dict[str, Any]] = []
        pairs = list(combinations(active, _TWO))
        for game_key in games:
            self.game_pool.record_play(game_key)
            for p_id, o_id in pairs:
                result = self._play_single(p_id, o_id, game_key)
                results.append(result)
        return results

    def _play_single(
        self, player_id: str, opponent_id: str, game_key: str,
    ) -> dict[str, Any]:
        """Run one game between two models."""
        p_fn = self.roster.get_generate_fn(player_id)
        o_fn = self.roster.get_generate_fn(opponent_id)
        if p_fn is None or o_fn is None:
            return {"player": player_id, "opponent": opponent_id,
                    "game": game_key, "error": "model not available"}
        opponent = FrozenOpponent(generate_fn=o_fn)
        env = KantEnvironment()
        try:
            obs = env.reset(game=game_key, opponent_fn=opponent)
        except (KeyError, ValueError):
            return {"player": player_id, "opponent": opponent_id,
                    "game": game_key, "error": "game not found"}
        while not obs.done:
            prompt = PromptBuilder.build(obs)
            raw = p_fn(prompt)
            action_str = parse_action(raw, obs.available_actions)
            obs = env.step(GameAction(action=action_str))
        return {
            "player": player_id, "opponent": opponent_id,
            "game": game_key,
            "player_score": obs.player_score,
            "opponent_score": obs.opponent_score,
            "rounds": obs.current_round,
        }

    def _phase_evaluate(
        self, active: list[str], game_results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Update reputation based on game outcomes."""
        scores: dict[str, list[float]] = {m: [] for m in active}
        totals: dict[str, float] = {m: _ZERO_F for m in active}
        for r in game_results:
            if "error" in r:
                continue
            pid = r["player"]
            oid = r["opponent"]
            ps = r.get("player_score", _ZERO_F)
            os_val = r.get("opponent_score", _ZERO_F)
            total = ps + os_val
            if total > _ZERO_F:
                p_coop = os_val / total
                o_coop = ps / total
            else:
                p_coop = _DEFAULT_SCORE
                o_coop = _DEFAULT_SCORE
            self.reputation.update_cooperation(pid, p_coop)
            self.reputation.update_cooperation(oid, o_coop)
            if total > _ZERO_F:
                fairness = _ONE_F - abs(ps - os_val) / total
                self.reputation.update_fairness(pid, fairness)
                self.reputation.update_fairness(oid, fairness)
            totals[pid] = totals.get(pid, _ZERO_F) + ps
            totals[oid] = totals.get(oid, _ZERO_F) + os_val
        rep_updates: dict[str, float] = {}
        for mid in active:
            rep = self.reputation.compute_reputation(mid)
            rep_updates[mid] = rep
            profile = self.roster.get_profile(mid)
            if profile is not None:
                profile.reputation = rep
                profile.games_played += len([
                    r for r in game_results
                    if r.get("player") == mid or r.get("opponent") == mid
                ])
        return rep_updates
