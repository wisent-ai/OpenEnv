"""Trainer-side rollouts for 2-phase free_chat games.

The legacy `_play_batch_interactive_episodes` in train.py assumes one
env.step() per round and one parsed action. apply_free_chat games run
on a 2-phase per-round protocol (message → reveal → action) where each
env.step needs to be called TWICE per round and the model is queried
TWICE per round (once for prose, once for the action token).

This module provides `_play_batch_free_chat_episodes` with the same
return shape as the legacy rollout, plus a per-round trajectory list
that captures both message-phase and action-phase emissions. Training
reward stays the action-segment payoff (apply_free_chat's payoff_fn is
identical to the base game); messages do not directly affect reward,
so any message-side behaviour the policy learns emerges naturally
from the joint distribution that GRPO optimizes.
"""
from __future__ import annotations

import logging
from typing import Any

from env.environment import KantEnvironment
from env.models import GameAction
from train.agent import (
    LLMAgent,
    PromptBuilder,
    parse_action,
    parse_free_chat,
)

logger = logging.getLogger(__name__)


def _local_coop_rate(history) -> float:
    """Cooperation rate over the player's actions in this trajectory."""
    if not history:
        return 0.0
    coop = {"cooperate", "stag", "dove", "contribute"}
    n = sum(1 for r in history if any(c in r.player_action for c in coop))
    return n / len(history)


def play_batch_free_chat_episodes(
    envs: list[KantEnvironment],
    episode_configs: list[tuple[str, str, str]],
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    batch_generate_fn,
) -> list[dict | None]:
    """Run N free_chat episodes in lockstep, batching message-phase and
    action-phase generations across envs.

    `batch_generate_fn(model, tokenizer, obs_list, device)` is the same
    helper the legacy rollout uses to generate one completion per active
    env in parallel. We call it twice per round: once with phase=message
    obs (model emits prose), once with phase=action obs (model emits
    action token after seeing the opponent's revealed message).

    Returns a list of episode result dicts shaped to match the legacy
    `_play_batch_interactive_episodes` return for drop-in compatibility,
    plus a `trajectory` field listing the (player_message,
    opponent_message, player_action, opponent_action) tuple per round.
    """
    n = len(episode_configs)
    results: list[dict | None] = [None] * n
    obs_list: list[Any] = [None] * n
    active = [True] * n

    for i, (game_key, strategy, _first_action) in enumerate(episode_configs):
        try:
            obs_list[i] = envs[i].reset(game=game_key, strategy=strategy)
        except Exception as exc:
            logger.debug("free_chat init error %s/%s: %s", game_key, strategy, exc)
            active[i] = False

    max_rounds = 40  # message+action phases double the step count per round
    for _ in range(max_rounds):
        # ---- message phase ---------------------------------------------------
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break

        active_obs = [obs_list[i] for i in active_indices]
        for obs in active_obs:
            obs.metadata.setdefault("phase", "message")
            obs.metadata.setdefault("free_chat", True)
        msg_completions = batch_generate_fn(model, tokenizer, active_obs, device)
        for j, i in enumerate(active_indices):
            try:
                obs_list[i] = envs[i].step(GameAction(
                    action=obs_list[i].available_actions[0],  # sentinel; ignored in msg phase
                    metadata={"message": (msg_completions[j] or "").strip()},
                ))
            except Exception as exc:
                logger.debug("free_chat msg step err episode %d: %s", i, exc)
                active[i] = False

        # ---- action phase ---------------------------------------------------
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break
        action_obs = [obs_list[i] for i in active_indices]
        act_completions = batch_generate_fn(model, tokenizer, action_obs, device)
        for j, i in enumerate(active_indices):
            try:
                # action phase: parse a bare action token
                act_str = parse_action(
                    (act_completions[j] or "").strip(),
                    obs_list[i].available_actions,
                )
                obs_list[i] = envs[i].step(GameAction(action=act_str))
                if obs_list[i].done:
                    active[i] = False
                    obs = obs_list[i]
                    coop = {"cooperate", "stag", "dove", "contribute"}
                    opp_coop = (
                        sum(1 for r in obs.history
                            if any(c in r.opponent_action for c in coop))
                        / len(obs.history)
                        if obs.history else 0.0
                    )
                    trajectory = [
                        {
                            "round": r.round_number,
                            "player_message": r.player_message,
                            "opponent_message": r.opponent_message,
                            "player_action": r.player_action,
                            "opponent_action": r.opponent_action,
                            "player_payoff": r.player_payoff,
                        }
                        for r in obs.history
                    ]
                    results[i] = {
                        "player_score": obs.player_score,
                        "opponent_score": obs.opponent_score,
                        "cooperation_rate": _local_coop_rate(obs.history),
                        "opponent_cooperation_rate": opp_coop,
                        "rounds": obs.current_round,
                        "strategy": episode_configs[i][1],
                        "trajectory": trajectory,
                    }
            except Exception as exc:
                logger.debug("free_chat act step err episode %d: %s", i, exc)
                active[i] = False

    return results


def is_free_chat_game(game_key: str) -> bool:
    """True if the game key is a free_chat variant per the registry. Used
    by the trainer dispatch to pick the 2-phase rollout when needed.
    Falls back to a key-prefix check when the registry import fails so
    this can be called from cold-start contexts."""
    if game_key.startswith("free_chat_"):
        return True
    try:
        from common.games import GAMES
        cfg = GAMES.get(game_key)
        if cfg is None:
            return False
        return "free_chat" in (getattr(cfg, "applied_variants", ()) or ())
    except Exception:
        return False
