"""Parts of train.py, split by the tama size splitter; train.py imports every name back."""

from __future__ import annotations
import logging
import json
import os
from typing import Any
import torch
logger = logging.getLogger(__name__)
from train_parts.system_prompt import REWARD_STRATEGIES
from train_parts.parse_args import _play_batch_interactive_episodes


def make_reward_fn(model=None, tokenizer=None):
    """Returns a GRPO reward function that plays INTERACTIVE episodes
    using the LOCAL environment (no network round-trips).

    The model plays adaptively round-by-round instead of repeating a
    fixed action, enabling learning of conditional strategies.
    Uses local KantEnvironment for ~100x faster episode play.
    """

    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Create a pool of local envs (one per concurrent episode)
    from env.environment import KantEnvironment as _KantEnv
    env_pool = [_KantEnv() for _ in range(len(REWARD_STRATEGIES) * 32)]

    # Per-rollout trajectory persistence for the dynamics study. When
    # WISENT_TRAJECTORY_LOG is set in the environment, each reward_fn
    # call appends one JSONL row per (completion, strategy) episode
    # that carries a trajectory list (free_chat games only — single-
    # phase games have no messages so there's nothing emergent to log
    # at this level). Step counter is closure-captured so plotting
    # over training step works without TRL exposing the trainer state.
    _trajectory_log_path = os.environ.get("WISENT_TRAJECTORY_LOG", "")
    _step_counter = [0]

    def reward_fn(
        completions: list[str],
        prompts: list[str],
        **kwargs: Any,
    ) -> list[float]:
        game_keys = kwargs.get("game_key", ["prisoners_dilemma"] * len(completions))
        variants = kwargs.get("variant", [""] * len(completions))
        available_moves_batch = kwargs.get(
            "available_moves", [["cooperate", "defect"]] * len(completions)
        )

        # Parse all first actions. For free_chat games skip — completions are NL
        # messages or 2-phase emissions, not bare action tokens. play_batch_free_chat_
        # episodes parses them per phase and ignores first_action.
        from train.free_to_ppo.free_chat_rollouts import is_free_chat_game as _is_fc_game
        first_actions = []
        for i, (completion, moves) in enumerate(zip(completions, available_moves_batch)):
            if _is_fc_game(game_keys[i]):
                action = moves[0]  # sentinel; play_batch_free_chat_episodes ignores it
            else:
                action = parse_action(completion.strip(), moves)
            first_actions.append(action)
            if i < 3:
                logger.info(
                    "Completion [%d] game=%s moves=%s -> parsed=%s | raw=%r",
                    i, game_keys[i], moves, action, completion[:200],
                )

        # Build ALL episode configs: each completion × 3 strategies
        episode_configs = []  # (game_key, strategy, first_action)
        completion_map = []   # maps episode index → completion index
        for i, (game_key, first_action) in enumerate(zip(game_keys, first_actions)):
            for strat in REWARD_STRATEGIES:
                episode_configs.append((game_key, strat, first_action))
                completion_map.append(i)

        # Play ALL episodes in batched mode
        episode_results = _play_batch_interactive_episodes(
            env_pool, episode_configs, model, tokenizer, device,
        )

        # Persist per-step trajectory data when WISENT_TRAJECTORY_LOG is
        # set. Only writes free_chat episodes (those that carry the full
        # message+action sequence) — non-free_chat results have no
        # messages to study. One JSONL row per episode, keyed by step.
        if _trajectory_log_path:
            _step_counter[0] += 1
            step = _step_counter[0]
            os.makedirs(os.path.dirname(_trajectory_log_path) or ".", exist_ok=True)  # parent dir may not exist on a fresh resumed agent VM (FileNotFoundError crashed 724084db restart14 @step~1160)
            with open(_trajectory_log_path, "a", encoding="utf-8") as fh:
                for ep_idx, ep in enumerate(episode_results):
                    if ep is None or "trajectory" not in ep:
                        continue
                    fh.write(json.dumps({
                        "step": step,
                        "completion_idx": completion_map[ep_idx],
                        "game": episode_configs[ep_idx][0],
                        "strategy": episode_configs[ep_idx][1],
                        "rounds": ep["rounds"],
                        "player_score": ep["player_score"],
                        "trajectory": ep["trajectory"],
                    }, ensure_ascii=False) + "\n")

        # Group results by completion and compute 5 metrics
        rewards = []
        n_strats = len(REWARD_STRATEGIES)
        for i in range(len(completions)):
            # Gather this completion's 3 episode results
            episodes = {}
            for j in range(n_strats):
                ep_idx = i * n_strats + j
                strat = REWARD_STRATEGIES[j]
                if episode_results[ep_idx] is not None:
                    episodes[strat] = episode_results[ep_idx]

            if not episodes:
                rewards.append(-1.0)
                continue

            # Mean per-round self-payoff across the cross-strategy episodes.
            # Same scalar train.rewards_to_trajectory.rewards.episode_reward returns per episode --
            # no cooperation / Pareto / fairness shaping.
            total_score = sum(ep["player_score"] for ep in episodes.values())
            total_rounds = sum(ep["rounds"] for ep in episodes.values())
            reward = total_score / total_rounds if total_rounds > 0 else 0.0
            rewards.append(reward)

        return rewards

    return reward_fn
