"""Self-play GRPO trainer for multi-agent training."""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Callable, Dict, List, Optional

from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import LLMAgent, PromptBuilder, parse_action
from train.rewards import episode_reward
from train.trajectory import TrajectoryCollector, EpisodeTrajectory
from train.self_play.opponents import FrozenOpponent, OpponentPool
from train.self_play.config import SelfPlayConfig
from constant_definitions.train.agent_constants import SYSTEM_PROMPT
from constant_definitions.train.grpo_constants import GRPO_LOG_EVERY
from constant_definitions.game_constants import EVAL_ZERO_FLOAT
from constant_definitions.var.meta.self_play_constants import (
    SELF_PLAY_COOP_WEIGHT_DENOMINATOR,
    SELF_PLAY_COOP_WEIGHT_NUMERATOR,
    SELF_PLAY_EXPLOIT_WEIGHT_DENOMINATOR,
    SELF_PLAY_EXPLOIT_WEIGHT_NUMERATOR,
    SELF_PLAY_FAIRNESS_WEIGHT_DENOMINATOR,
    SELF_PLAY_FAIRNESS_WEIGHT_NUMERATOR,
    SELF_PLAY_PARETO_WEIGHT_DENOMINATOR,
    SELF_PLAY_PARETO_WEIGHT_NUMERATOR,
    SELF_PLAY_ADAPT_WEIGHT_DENOMINATOR,
    SELF_PLAY_ADAPT_WEIGHT_NUMERATOR,
    SELF_PLAY_OPPONENT_LABEL,
)

logger = logging.getLogger(__name__)

_ZERO = int()
_ONE = int(bool(True))


def _self_play_weights() -> Dict[str, float]:
    """Return reward weights tuned for self-play training."""
    return {
        "exploitation_resistance": (
            SELF_PLAY_EXPLOIT_WEIGHT_NUMERATOR
            / SELF_PLAY_EXPLOIT_WEIGHT_DENOMINATOR
        ),
        "cooperation_rate": (
            SELF_PLAY_COOP_WEIGHT_NUMERATOR
            / SELF_PLAY_COOP_WEIGHT_DENOMINATOR
        ),
        "pareto_efficiency": (
            SELF_PLAY_PARETO_WEIGHT_NUMERATOR
            / SELF_PLAY_PARETO_WEIGHT_DENOMINATOR
        ),
        "fairness_index": (
            SELF_PLAY_FAIRNESS_WEIGHT_NUMERATOR
            / SELF_PLAY_FAIRNESS_WEIGHT_DENOMINATOR
        ),
        "adaptability": (
            SELF_PLAY_ADAPT_WEIGHT_NUMERATOR
            / SELF_PLAY_ADAPT_WEIGHT_DENOMINATOR
        ),
    }


class SelfPlayTrainer:
    """GRPO training with self-play opponents.

    Training loop:
    1. Collect trajectories: training model vs frozen opponent
    2. Compute GRPO rewards from episode outcomes
    3. Update training model via TRL GRPOTrainer
    4. Periodically refresh frozen opponent from training model
    5. Add old opponent to pool for diversity

    Parameters
    ----------
    config : SelfPlayConfig
        Training configuration.
    model : object
        HuggingFace model to train.
    tokenizer : object
        Tokenizer for the model.
    env : KantEnvironment, optional
        Game environment instance.
    """

    def __init__(
        self,
        config: SelfPlayConfig,
        model: object,
        tokenizer: object,
        env: Optional[KantEnvironment] = None,
    ) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._env = env or KantEnvironment()
        self._pool = OpponentPool(max_size=config.pool_max_size)
        self._frozen = FrozenOpponent.from_model(model, tokenizer)
        self._pool.add(self._frozen)
        self._step_count = _ZERO

    def _model_generate(self, prompt: str) -> str:
        """Generate a completion from the training model."""
        import torch

        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_len = len(inputs["input_ids"][_ZERO])
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_completion_length,
            )
            return self._tokenizer.decode(
                outputs[_ZERO][input_len:],
                skip_special_tokens=True,
            )

    def collect_trajectories(
        self,
        games: List[str],
        num_episodes: int,
    ) -> List[EpisodeTrajectory]:
        """Collect episodes with current frozen opponent."""
        agent = LLMAgent(generate_fn=self._model_generate)
        collector = TrajectoryCollector(
            env=self._env,
            agent=agent,
            reward_fn=lambda ps, os, cr, tr: episode_reward(
                ps, os, cr, tr, weights=_self_play_weights(),
            ),
        )
        trajectories: List[EpisodeTrajectory] = []
        for _ep in range(num_episodes):
            game = random.choice(games)
            opponent = self._pool.sample()
            traj = collector.collect_episode(
                game=game,
                strategy=SELF_PLAY_OPPONENT_LABEL,
                opponent_fn=opponent,
            )
            trajectories.append(traj)
        return trajectories

    def make_reward_fn(self) -> Callable[..., List[float]]:
        """Create GRPO reward function using self-play episodes."""
        pool = self._pool
        env = self._env
        weights = _self_play_weights()

        def reward_fn(
            completions: List[str],
            prompts: List[str],
            **kwargs: Any,
        ) -> List[float]:
            rewards: List[float] = []
            game_keys = kwargs.get(
                "game_key",
                ["prisoners_dilemma"] * len(completions),
            )
            moves_batch = kwargs.get(
                "available_moves",
                [["cooperate", "defect"]] * len(completions),
            )
            for completion, game_key, moves in zip(
                completions, game_keys, moves_batch,
            ):
                action_str = parse_action(completion.strip(), moves)
                opponent = pool.sample()
                obs = env.reset(
                    game=game_key, opponent_fn=opponent,
                )
                while not obs.done:
                    obs = env.step(GameAction(action=action_str))
                reward = episode_reward(
                    obs.player_score,
                    obs.opponent_score,
                    _compute_coop_rate(obs),
                    obs.current_round,
                    weights=weights,
                )
                rewards.append(reward)
            return rewards

        return reward_fn

    def refresh_opponent(self) -> None:
        """Copy current training model to a new frozen opponent."""
        frozen_model = copy.deepcopy(self._model)
        frozen_model.eval()
        new_opponent = FrozenOpponent.from_model(
            frozen_model, self._tokenizer,
        )
        self._pool.add(new_opponent)
        self._frozen = new_opponent
        logger.info(
            "Refreshed opponent. Pool size: %d", self._pool.size,
        )

    def train(self, games: List[str]) -> None:
        """Main self-play training loop.

        Parameters
        ----------
        games : list of str
            Game keys to train on.
        """
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        import torch

        trajectories = self.collect_trajectories(
            games, self._config.warmup_episodes,
        )
        samples = []
        for traj in trajectories:
            for step in traj.steps:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": step.prompt},
                ]
                formatted = self._tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                )
                samples.append({
                    "prompt": formatted,
                    "game_key": traj.game,
                    "available_moves": ["cooperate", "defect"],
                })
        dataset = Dataset.from_list(samples)

        reward_fn = self.make_reward_fn()

        trl_config = GRPOConfig(
            output_dir=self._config.output_dir,
            num_generations=self._config.num_generations,
            max_completion_length=self._config.max_completion_length,
            per_device_train_batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
            max_steps=self._config.max_steps,
            logging_steps=GRPO_LOG_EVERY,
            save_steps=self._config.opponent_update_interval,
            bf16=torch.cuda.is_available(),
        )

        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=reward_fn,
            args=trl_config,
            train_dataset=dataset,
            processing_class=self._tokenizer,
        )

        trainer.train()
        trainer.save_model(self._config.output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove"})


def _compute_coop_rate(obs: GameObservation) -> float:
    """Fraction of cooperative moves in an episode."""
    if not obs.history:
        return EVAL_ZERO_FLOAT
    total = len(obs.history)
    count = _ZERO
    for rnd in obs.history:
        if rnd.player_action in _COOPERATIVE_ACTIONS:
            count += _ONE
    return count / total
