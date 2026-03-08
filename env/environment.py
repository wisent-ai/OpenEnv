"""Core KantBench environment implementing the OpenEnv Environment interface."""
from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from openenv.core.env_server.interfaces import Environment
from env.models import GameAction, GameObservation, GameState, RoundResult
from common.games import GameConfig, get_game, GAMES
from common.strategies import get_strategy, STRATEGIES, OpponentStrategy
from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS

_ONE = int(bool(True))
_ZERO_F = float()


class KantEnvironment(Environment[GameObservation, GameAction, GameState]):
    """Game-theory environment hosting multiple classic games.

    The agent plays against a built-in opponent strategy or another agent
    function. The opponent's move is computed automatically inside ``step()``
    via the selected strategy or the provided ``opponent_fn``.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._game: Optional[GameConfig] = None
        self._strategy: Optional[OpponentStrategy] = None
        self._strategy_name: str = ""
        self._opponent_fn: Optional[Callable[[GameObservation], GameAction]] = None
        self._state: GameState = GameState()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GameObservation:
        game_name: str = kwargs.get("game", "prisoners_dilemma")
        strategy_name: str = kwargs.get("strategy", "tit_for_tat")
        num_rounds: Optional[int] = kwargs.get("num_rounds")
        opponent_fn: Optional[Callable[[GameObservation], GameAction]] = kwargs.get(
            "opponent_fn",
        )

        self._game = get_game(game_name)
        self._opponent_fn = opponent_fn
        if opponent_fn is not None:
            self._strategy = None
            self._strategy_name = "agent"
        else:
            self._strategy = get_strategy(strategy_name)
            self._strategy_name = strategy_name

        rounds = num_rounds if num_rounds is not None else self._game.default_rounds

        self._state = GameState(
            episode_id=episode_id or str(uuid.uuid4()),
            game_name=game_name,
            opponent_strategy=strategy_name,
            total_rounds=rounds,
        )

        return self._build_observation()

    def step(
        self,
        action: GameAction,
        **kwargs: Any,
    ) -> GameObservation:
        if self._game is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.is_done:
            raise RuntimeError("Episode already finished. Call reset().")
        if action.action not in self._game.actions:
            raise ValueError(
                f"Invalid action '{action.action}'. "
                f"Choose from: {self._game.actions}"
            )

        player_action = action.action
        opponent_action = self._auto_play_opponent(player_action)

        p_pay, o_pay = self._game.payoff_fn(player_action, opponent_action)

        new_round = len(self._state.history) + _ONE
        result = RoundResult(
            round_number=new_round,
            player_action=player_action,
            opponent_action=opponent_action,
            player_payoff=p_pay,
            opponent_payoff=o_pay,
        )

        history = list(self._state.history) + [result]
        p_score = self._state.player_score + p_pay
        o_score = self._state.opponent_score + o_pay
        done = new_round >= self._state.total_rounds

        self._state = GameState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + _ONE,
            game_name=self._state.game_name,
            opponent_strategy=self._state.opponent_strategy,
            current_round=new_round,
            total_rounds=self._state.total_rounds,
            player_score=p_score,
            opponent_score=o_score,
            history=history,
            is_done=done,
        )

        return self._build_observation(reward=p_pay, last_round=result, done=done)

    @property
    def state(self) -> GameState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_play_opponent(self, player_action: str) -> str:
        assert self._game is not None

        if self._opponent_fn is not None:
            opp_obs = self._build_opponent_observation()
            opp_action = self._opponent_fn(opp_obs)
            opp_actions = self._opponent_actions()
            if opp_action.action not in opp_actions:
                raise ValueError(
                    f"Opponent returned invalid action '{opp_action.action}'. "
                    f"Choose from: {opp_actions}"
                )
            return opp_action.action

        assert self._strategy is not None
        hist = [
            {
                "player_action": r.player_action,
                "opponent_action": r.opponent_action,
            }
            for r in self._state.history
        ]
        opp_actions = self._opponent_actions()
        return self._strategy.choose_action(
            self._game.game_type, opp_actions, hist,
        )

    def _opponent_actions(self) -> list[str]:
        assert self._game is not None
        if self._game.opponent_actions is not None:
            return list(self._game.opponent_actions)
        gt = self._game.game_type
        if gt == "ultimatum":
            return ["accept", "reject"]
        if gt == "trust":
            return _trust_return_actions()
        # matrix, public_goods, auction, commons, dictator, centipede,
        # stackelberg, and all generated games share action space
        return list(self._game.actions)

    def _build_opponent_observation(self) -> GameObservation:
        """Build a GameObservation from the opponent's perspective.

        Swaps player/opponent in history, scores, and payoffs so the opponent
        agent sees itself as the "player".
        """
        assert self._game is not None
        flipped_history = [
            RoundResult(
                round_number=r.round_number,
                player_action=r.opponent_action,
                opponent_action=r.player_action,
                player_payoff=r.opponent_payoff,
                opponent_payoff=r.player_payoff,
            )
            for r in self._state.history
        ]
        opp_actions = self._opponent_actions()
        return GameObservation(
            done=False,
            reward=_ZERO_F,
            game_name=self._state.game_name,
            game_description=self._game.description,
            available_actions=opp_actions,
            current_round=self._state.current_round,
            total_rounds=self._state.total_rounds,
            history=flipped_history,
            player_score=self._state.opponent_score,
            opponent_score=self._state.player_score,
            opponent_strategy="agent",
        )

    def _build_observation(
        self,
        reward: float = _ZERO_F,
        last_round: Optional[RoundResult] = None,
        done: bool = False,
    ) -> GameObservation:
        assert self._game is not None
        return GameObservation(
            done=done,
            reward=reward,
            game_name=self._state.game_name,
            game_description=self._game.description,
            available_actions=list(self._game.actions),
            current_round=self._state.current_round,
            total_rounds=self._state.total_rounds,
            history=list(self._state.history),
            player_score=self._state.player_score,
            opponent_score=self._state.opponent_score,
            opponent_strategy=self._strategy_name,
            last_round=last_round,
        )


def _trust_return_actions() -> list[str]:
    from constant_definitions.game_constants import TRUST_ENDOWMENT, TRUST_MULTIPLIER
    cap = TRUST_ENDOWMENT * TRUST_MULTIPLIER
    return [f"return_{i}" for i in range(cap + _ONE)]
