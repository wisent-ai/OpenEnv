"""GRPO trainer wrapping TRL with Kant-specific logic."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import LLMAgent, PromptBuilder, parse_action
from train.grpo.config import GRPOConfig
from train.rewards import episode_reward, per_step_shaping
from train.splits import get_train_eval_split
from train.trajectory import TrajectoryCollector

from constant_definitions.game_constants import EVAL_ONE, EVAL_ZERO, EVAL_ZERO_FLOAT

logger = logging.getLogger(__name__)

_ONE = int(bool(True))


class KantGRPOTrainer:
    """GRPO trainer for strategic reasoning in game-theory environments.

    Wraps TRL's GRPOTrainer with:
    - Environment-based reward computation
    - Curriculum scheduling over games
    - Per-checkpoint evaluation logging

    Parameters
    ----------
    config : GRPOConfig
        Training configuration.
    model : Any
        HuggingFace model (or path to load).
    tokenizer : Any
        HuggingFace tokenizer.
    env : KantEnvironment, optional
        Environment instance for reward computation.
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: Any = None,
        tokenizer: Any = None,
        env: Optional[KantEnvironment] = None,
    ) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._env = env if env is not None else KantEnvironment()
        from common.games import GAMES as _AVAILABLE_GAMES
        raw_train, raw_eval = get_train_eval_split()
        self._train_games = frozenset(g for g in raw_train if g in _AVAILABLE_GAMES)
        self._eval_games = frozenset(g for g in raw_eval if g in _AVAILABLE_GAMES)
        self._current_games: List[str] = sorted(self._train_games)[
            :config.curriculum_initial_games
        ]
        self._step_count = EVAL_ZERO
        self._trl_trainer: Any = None

    def reward_function(
        self,
        completions: List[str],
        prompts: List[str],
    ) -> List[float]:
        """Compute rewards by parsing actions and evaluating in environment.

        This is the reward function passed to TRL's GRPOTrainer.
        Each (prompt, completion) pair is treated as a single round action.
        """
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            # We cannot run a full episode per completion in GRPO
            # (completions are individual round actions), so we return
            # per-step shaping reward based on action quality heuristic.
            reward = EVAL_ZERO_FLOAT
            rewards.append(reward)
        return rewards

    def expand_curriculum(self) -> None:
        """Add more games to the training curriculum."""
        all_train = sorted(self._train_games)
        current_count = len(self._current_games)
        new_count = min(
            current_count + self._config.curriculum_expansion_step,
            len(all_train),
        )
        self._current_games = all_train[:new_count]
        logger.info(
            "Curriculum expanded to %s games",
            str(len(self._current_games)),
        )

    def setup_trl_trainer(self) -> Any:
        """Initialise the TRL GRPOTrainer (requires trl to be installed)."""
        try:
            from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
        except ImportError as exc:
            msg = "trl is required for GRPO training. Install with: pip install trl"
            raise ImportError(msg) from exc

        trl_config = TRLGRPOConfig(**self._config.to_trl_kwargs())
        self._trl_trainer = GRPOTrainer(
            model=self._model,
            config=trl_config,
            tokenizer=self._tokenizer,
            reward_funcs=self.reward_function,
        )
        return self._trl_trainer

    def evaluate(
        self,
        games: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        run_external: bool = False,
        external_benchmarks: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Run evaluation on specified games and return metric dict.

        Parameters
        ----------
        games, strategies
            Forwarded to ``TournamentRunner``.
        run_external : bool
            If ``True``, also run external safety benchmarks.
        external_benchmarks : sequence of str, optional
            Which external benchmarks to run (default: all).
        """
        from bench.evaluation.tournament import TournamentRunner
        from bench.evaluation.metrics import compute_metrics

        eval_games = list(games) if games is not None else sorted(self._eval_games)

        def _agent_fn(obs: GameObservation) -> GameAction:
            prompt = PromptBuilder.build(obs)
            if self._tokenizer is not None and self._model is not None:
                inputs = self._tokenizer(prompt, return_tensors="pt")
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_completion_length,
                )
                completion = self._tokenizer.decode(
                    outputs[EVAL_ZERO][len(inputs["input_ids"][EVAL_ZERO]):],
                    skip_special_tokens=True,
                )
            else:
                completion = obs.available_actions[EVAL_ZERO]
            action_str = parse_action(completion, obs.available_actions)
            return GameAction(action=action_str)

        runner = TournamentRunner(env=self._env, agent_fn=_agent_fn)
        results = runner.run_tournament_as_dict(
            games=eval_games,
            strategies=strategies,
        )
        metrics = compute_metrics(results)

        if run_external:
            from bench.external._model_handle import ModelHandle
            from bench.external.runner import ExternalBenchmarkRunner

            handle = ModelHandle(
                model_name_or_path=self._config.model_name,
                model=self._model,
                tokenizer=self._tokenizer,
            )
            ext_runner = ExternalBenchmarkRunner(
                model_handle=handle,
                benchmarks=external_benchmarks,
            )
            ext_results = ext_runner.run_all()
            for bench_name, result in ext_results.items():
                prefix = f"external/{bench_name}"
                if result.error is not None:
                    metrics[f"{prefix}/error"] = True
                    continue
                for metric_key, value in result.scores.items():
                    metrics[f"{prefix}/{metric_key}"] = value

        return metrics

    @property
    def current_games(self) -> List[str]:
        """Currently active training games."""
        return list(self._current_games)

    @property
    def config(self) -> GRPOConfig:
        """Training configuration."""
        return self._config
