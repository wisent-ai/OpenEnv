"""DPO trainer wrapping TRL with Kant-specific preference learning."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from env.environment import KantEnvironment
from env.models import GameAction, GameObservation
from train.agent import LLMAgent, PromptBuilder, parse_action
from train.dpo.config import DPOConfig
from train.dpo.pairs import generate_preference_pairs
from train.splits import get_train_eval_split
from train.trajectory import EpisodeTrajectory

from constant_definitions.game_constants import EVAL_ZERO

logger = logging.getLogger(__name__)


class KantDPOTrainer:
    """DPO trainer for strategic reasoning via preference learning.

    Wraps TRL's DPOTrainer with:
    - Preference pair generation from trajectory rankings
    - Per-checkpoint evaluation on held-out games
    - Optional LoRA/QLoRA support via PEFT

    Parameters
    ----------
    config : DPOConfig
        Training configuration.
    model : Any
        HuggingFace model (or path to load).
    tokenizer : Any
        HuggingFace tokenizer.
    ref_model : Any, optional
        Reference model for DPO. If None, uses a copy of the policy model.
    """

    def __init__(
        self,
        config: DPOConfig,
        model: Any = None,
        tokenizer: Any = None,
        ref_model: Any = None,
    ) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._ref_model = ref_model
        self._train_games, self._eval_games = get_train_eval_split()
        self._trl_trainer: Any = None

    def prepare_dataset(
        self,
        trajectories: List[EpisodeTrajectory],
    ) -> List[Dict[str, Any]]:
        """Generate preference pairs from collected trajectories."""
        return generate_preference_pairs(
            trajectories,
            min_margin_numerator=self._config.min_reward_margin_numerator,
            min_margin_denominator=self._config.min_reward_margin_denominator,
        )

    def setup_trl_trainer(
        self,
        train_dataset: Any,
    ) -> Any:
        """Initialise the TRL DPOTrainer (requires trl to be installed)."""
        try:
            from trl import DPOTrainer, DPOConfig as TRLDPOConfig
        except ImportError as exc:
            msg = "trl is required for DPO training. Install with: pip install trl"
            raise ImportError(msg) from exc

        trl_config = TRLDPOConfig(**self._config.to_trl_kwargs())
        self._trl_trainer = DPOTrainer(
            model=self._model,
            ref_model=self._ref_model,
            args=trl_config,
            tokenizer=self._tokenizer,
            train_dataset=train_dataset,
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

        env = KantEnvironment()
        eval_games = list(games) if games is not None else sorted(self._eval_games)

        def _agent_fn(obs: GameObservation) -> GameAction:
            prompt = PromptBuilder.build(obs)
            if self._tokenizer is not None and self._model is not None:
                inputs = self._tokenizer(prompt, return_tensors="pt")
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_length,
                )
                completion = self._tokenizer.decode(
                    outputs[EVAL_ZERO][len(inputs["input_ids"][EVAL_ZERO]):],
                    skip_special_tokens=True,
                )
            else:
                completion = obs.available_actions[EVAL_ZERO]
            action_str = parse_action(completion, obs.available_actions)
            return GameAction(action=action_str)

        runner = TournamentRunner(env=env, agent_fn=_agent_fn)
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
    def config(self) -> DPOConfig:
        """Training configuration."""
        return self._config
