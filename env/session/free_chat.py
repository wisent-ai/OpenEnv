"""2-phase free-chat round helpers for KantEnvironment.

Extracted from env/environment.py to keep that file under the 300-line
per-file cap. The semantics: a free_chat round is split into

    phase = 'message'  -> both players submit a free-form text message
    (env reveals both)
    phase = 'action'   -> both players submit an action token; payoff
                          is computed; round advances; phase resets.

`_step_free_chat(env, action)` is the single per-call dispatcher invoked
from KantEnvironment.step when the active GameConfig has the 'free_chat'
applied variant. It mutates env state (env._phase, pending message
buffers, env._state on round completion) and returns the next observation.

The helpers take the env as their first argument (rather than being
methods on the class) because env/environment.py is already at the cap;
splitting these out keeps the variant code reviewable in isolation
without growing the env file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from env.models import GameAction, GameObservation, GameState, RoundResult

if TYPE_CHECKING:
    from env.environment import KantEnvironment

_ONE = int(bool(True))
_ZERO_F = float()


def step_free_chat(env: "KantEnvironment", action: GameAction) -> GameObservation:
    """Two-phase free-chat round dispatcher. See module docstring."""
    assert env._game is not None
    if env._phase == "message":
        env._pending_player_message = (action.metadata or {}).get("message", "") or ""
        env._pending_opp_message = _auto_play_opponent_message(env)
        env._phase = "action"
        return _build_obs_message_phase_complete(env)

    # phase == 'action'
    if action.action not in env._game.actions:
        raise ValueError(
            f"Invalid action '{action.action}'. Choose from: {env._game.actions}"
        )
    opp_action = _auto_play_opponent_action_only(env, action.action)
    p_pay, o_pay = env._game.payoff_fn(action.action, opp_action)
    new_round = len(env._state.history) + _ONE
    result = RoundResult(
        round_number=new_round,
        player_action=action.action,
        opponent_action=opp_action,
        player_payoff=p_pay,
        opponent_payoff=o_pay,
        player_message=env._pending_player_message,
        opponent_message=env._pending_opp_message,
    )
    history = list(env._state.history) + [result]
    done = new_round >= env._state.total_rounds
    env._state = GameState(
        episode_id=env._state.episode_id,
        step_count=env._state.step_count + _ONE,
        game_name=env._state.game_name,
        opponent_strategy=env._state.opponent_strategy,
        current_round=new_round,
        total_rounds=env._state.total_rounds,
        player_score=env._state.player_score + p_pay,
        opponent_score=env._state.opponent_score + o_pay,
        history=history,
        is_done=done,
    )
    env._phase = "message"
    env._pending_player_message = ""
    env._pending_opp_message = ""
    return env._build_observation(reward=p_pay, last_round=result, done=done)


def _auto_play_opponent_message(env: "KantEnvironment") -> str:
    """Ask the opponent_fn for their message-phase emission. Scripted
    strategies (no opponent_fn) return empty string — they have no
    language; the game still works but only one side speaks."""
    if env._opponent_fn is None:
        return ""
    opp_obs = env._build_opponent_observation()
    opp_obs.metadata["phase"] = "message"
    opp_obs.metadata["free_chat"] = True
    opp_action = env._opponent_fn(opp_obs)
    return (opp_action.metadata or {}).get("message", "") or ""


def _auto_play_opponent_action_only(env: "KantEnvironment", player_action: str) -> str:
    """Action-phase opponent call. Surfaces the just-pending player_message
    via obs.metadata['last_opp_message'] so the opponent can react to
    what was said this round."""
    if env._opponent_fn is not None:
        opp_obs = env._build_opponent_observation()
        opp_obs.metadata["phase"] = "action"
        opp_obs.metadata["free_chat"] = True
        opp_obs.metadata["last_opp_message"] = env._pending_player_message
        opp_obs.metadata["last_player_message"] = env._pending_opp_message
        opp_action = env._opponent_fn(opp_obs)
        opp_actions = env._opponent_actions()
        if opp_action.action not in opp_actions:
            raise ValueError(
                f"Opponent returned invalid action '{opp_action.action}'. "
                f"Choose from: {opp_actions}"
            )
        return opp_action.action
    assert env._strategy is not None
    hist = [
        {"player_action": r.player_action, "opponent_action": r.opponent_action}
        for r in env._state.history
    ]
    opp_actions = env._opponent_actions()
    return env._strategy.choose_action(env._game.game_type, opp_actions, hist)


def _build_obs_message_phase_complete(env: "KantEnvironment") -> GameObservation:
    """Observation returned at the end of phase=message: both messages
    are now visible; agent's next call must produce an ACTION."""
    assert env._game is not None
    meta = {
        "free_chat": True,
        "phase": "action",
        "last_opp_message": env._pending_opp_message,
        "last_player_message": env._pending_player_message,
    }
    return GameObservation(
        done=False,
        reward=_ZERO_F,
        game_name=env._state.game_name,
        game_description=env._game.description,
        available_actions=list(env._game.actions),
        current_round=env._state.current_round + _ONE,
        total_rounds=env._state.total_rounds,
        history=list(env._state.history),
        player_score=env._state.player_score,
        opponent_score=env._state.opponent_score,
        opponent_strategy=env._strategy_name,
        metadata=meta,
    )
