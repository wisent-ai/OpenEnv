"""N-player game environment."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from common.games_meta.nplayer_config import NPlayerGameConfig, get_nplayer_game
from env.nplayer.models import (
    NPlayerAction,
    NPlayerGameState,
    NPlayerObservation,
    NPlayerRoundResult,
)
from env.nplayer.strategies import get_nplayer_strategy, NPlayerStrategy

_ONE = int(bool(True))
_ZERO = int()
_ZERO_F = float()


class NPlayerEnvironment:
    """Game-theory environment for N-player games.

    Player zero is the primary agent controlled via ``step()``.
    Players one through N-minus-one are auto-played by strategies or
    caller-provided functions (``opponent_fns``).
    """

    def __init__(self) -> None:
        self._game: Optional[NPlayerGameConfig] = None
        self._strategies: list[Optional[NPlayerStrategy]] = []
        self._opponent_fns: list[Optional[Callable[[NPlayerObservation], NPlayerAction]]] = []
        self._state: NPlayerGameState = NPlayerGameState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        game: str,
        *,
        num_rounds: Optional[int] = None,
        opponent_strategies: Optional[list[str]] = None,
        opponent_fns: Optional[list[Optional[Callable[[NPlayerObservation], NPlayerAction]]]] = None,
        episode_id: Optional[str] = None,
    ) -> NPlayerObservation:
        """Start a new episode.

        Parameters
        ----------
        game:
            Key in ``NPLAYER_GAMES``.
        num_rounds:
            Override the default round count.
        opponent_strategies:
            Strategy names for players one through N-minus-one.  If shorter
            than needed, the last entry is repeated. Defaults to all
            ``"random"``.
        opponent_fns:
            Callable opponents for players one through N-minus-one.  ``None``
            entries fall back to the corresponding strategy.
        episode_id:
            Optional identifier for the episode.
        """
        self._game = get_nplayer_game(game)
        n = self._game.num_players
        num_opponents = n - _ONE

        # Resolve strategies
        if opponent_strategies is None:
            strat_names = ["random"] * num_opponents
        else:
            strat_names = list(opponent_strategies)
            while len(strat_names) < num_opponents:
                strat_names.append(strat_names[-_ONE])
        self._strategies = [get_nplayer_strategy(s) for s in strat_names]

        # Resolve opponent fns
        if opponent_fns is None:
            self._opponent_fns = [None] * num_opponents
        else:
            fns: list[Optional[Callable]] = list(opponent_fns)
            while len(fns) < num_opponents:
                fns.append(None)
            self._opponent_fns = fns

        rounds = num_rounds if num_rounds is not None else self._game.default_rounds

        self._state = NPlayerGameState(
            episode_id=episode_id or str(uuid.uuid4()),
            game_name=game,
            total_rounds=rounds,
            num_players=n,
            scores=[_ZERO_F] * n,
        )

        return self._build_observation(_ZERO)

    def step(self, action: NPlayerAction) -> NPlayerObservation:
        """Execute one round.

        The caller supplies the action for player zero. Opponents are
        auto-played.
        """
        if self._game is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.is_done:
            raise RuntimeError("Episode already finished. Call reset().")
        if action.action not in self._game.actions:
            raise ValueError(
                f"Invalid action '{action.action}'. "
                f"Choose from: {self._game.actions}"
            )

        # Collect all actions: player zero first, then opponents
        all_actions: list[str] = [action.action]
        for idx in range(len(self._strategies)):
            player_idx = idx + _ONE
            opp_action = self._get_opponent_action(idx, player_idx)
            all_actions.append(opp_action)

        actions_tuple = tuple(all_actions)
        payoffs_tuple = self._game.payoff_fn(actions_tuple)

        new_round = len(self._state.history) + _ONE
        result = NPlayerRoundResult(
            round_number=new_round,
            actions=list(all_actions),
            payoffs=list(payoffs_tuple),
        )

        history = list(self._state.history) + [result]
        new_scores = [
            s + p for s, p in zip(self._state.scores, payoffs_tuple)
        ]
        done = new_round >= self._state.total_rounds

        self._state = NPlayerGameState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + _ONE,
            game_name=self._state.game_name,
            current_round=new_round,
            total_rounds=self._state.total_rounds,
            num_players=self._state.num_players,
            scores=new_scores,
            history=history,
            is_done=done,
        )

        return self._build_observation(
            _ZERO,
            reward=payoffs_tuple[_ZERO],
            last_round=result,
            done=done,
        )

    @property
    def state(self) -> NPlayerGameState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_opponent_action(self, opp_idx: int, player_idx: int) -> str:
        """Get the action for opponent at opp_idx (player player_idx)."""
        assert self._game is not None
        fn = self._opponent_fns[opp_idx]
        if fn is not None:
            obs = self._build_observation(player_idx)
            opp_action = fn(obs)
            if opp_action.action not in self._game.actions:
                raise ValueError(
                    f"Opponent {player_idx} returned invalid action "
                    f"'{opp_action.action}'. Choose from: {self._game.actions}"
                )
            return opp_action.action

        strategy = self._strategies[opp_idx]
        assert strategy is not None
        obs = self._build_observation(player_idx)
        return strategy.choose_action(obs)

    def _build_observation(
        self,
        player_index: int,
        reward: float = _ZERO_F,
        last_round: Optional[NPlayerRoundResult] = None,
        done: bool = False,
    ) -> NPlayerObservation:
        assert self._game is not None
        return NPlayerObservation(
            done=done,
            reward=reward,
            game_name=self._state.game_name,
            game_description=self._game.description,
            available_actions=list(self._game.actions),
            current_round=self._state.current_round,
            total_rounds=self._state.total_rounds,
            history=list(self._state.history),
            scores=list(self._state.scores),
            num_players=self._state.num_players,
            player_index=player_index,
            last_round=last_round,
        )
