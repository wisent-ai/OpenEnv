"""Trajectory collection for training data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from env.models import GameAction, GameObservation, RoundResult
from env.environment import KantEnvironment
from constant_definitions.game_constants import EVAL_ZERO_FLOAT


@dataclass
class StepRecord:
    """A single step within an episode trajectory."""

    prompt: str
    completion: str
    action: str
    reward: float
    player_payoff: float
    opponent_payoff: float
    round_number: int


@dataclass
class EpisodeTrajectory:
    """Complete trajectory of one episode."""

    game: str
    strategy: str
    steps: List[StepRecord] = field(default_factory=list)
    episode_reward: float = EVAL_ZERO_FLOAT
    player_score: float = EVAL_ZERO_FLOAT
    opponent_score: float = EVAL_ZERO_FLOAT
    cooperation_rate: float = EVAL_ZERO_FLOAT
    rounds_played: int = int()
    metrics: Dict[str, float] = field(default_factory=dict)


class TrajectoryCollector:
    """Runs episodes and collects trajectories for training.

    Parameters
    ----------
    env : KantEnvironment
        The game environment instance.
    agent : LLMAgent
        An agent with ``last_prompt`` / ``last_completion`` properties,
        callable with ``(GameObservation) -> GameAction``.
    reward_fn : callable, optional
        Function(player_score, opponent_score, cooperation_rate, rounds) -> float.
    step_reward_fn : callable, optional
        Function(player_payoff, opponent_payoff, payoff_min, payoff_max) -> float.
    """

    def __init__(
        self,
        env: KantEnvironment,
        agent: Any,
        reward_fn: Optional[Callable[..., float]] = None,
        step_reward_fn: Optional[Callable[..., float]] = None,
    ) -> None:
        self._env = env
        self._agent = agent
        self._reward_fn = reward_fn
        self._step_reward_fn = step_reward_fn

    def collect_episode(
        self,
        game: str,
        strategy: str = "tit_for_tat",
        opponent_fn: Optional[Callable] = None,
    ) -> EpisodeTrajectory:
        """Run a single episode and return its trajectory."""
        if opponent_fn is not None:
            obs = self._env.reset(game=game, opponent_fn=opponent_fn)
        else:
            obs = self._env.reset(game=game, strategy=strategy)
        steps: List[StepRecord] = []

        while not obs.done:
            action = self._agent(obs)

            # Capture prompt/completion from agent
            prompt = getattr(self._agent, "last_prompt", "")
            completion = getattr(self._agent, "last_completion", "")

            next_obs = self._env.step(action)

            # Compute step reward
            step_reward = EVAL_ZERO_FLOAT
            if self._step_reward_fn is not None and next_obs.last_round is not None:
                step_reward = self._step_reward_fn(
                    next_obs.last_round.player_payoff,
                    next_obs.last_round.opponent_payoff,
                    EVAL_ZERO_FLOAT,
                    EVAL_ZERO_FLOAT,
                )

            # Record step
            last_rnd = next_obs.last_round
            steps.append(StepRecord(
                prompt=prompt,
                completion=completion,
                action=action.action,
                reward=step_reward,
                player_payoff=(
                    last_rnd.player_payoff if last_rnd is not None
                    else EVAL_ZERO_FLOAT
                ),
                opponent_payoff=(
                    last_rnd.opponent_payoff if last_rnd is not None
                    else EVAL_ZERO_FLOAT
                ),
                round_number=next_obs.current_round,
            ))
            obs = next_obs

        # Compute cooperation rate (reusing tournament logic pattern)
        coop_rate = _compute_cooperation_rate(obs)

        # Compute episode reward
        ep_reward = EVAL_ZERO_FLOAT
        if self._reward_fn is not None:
            ep_reward = self._reward_fn(
                obs.player_score,
                obs.opponent_score,
                coop_rate,
                obs.current_round,
            )

        return EpisodeTrajectory(
            game=game,
            strategy=strategy,
            steps=steps,
            episode_reward=ep_reward,
            player_score=obs.player_score,
            opponent_score=obs.opponent_score,
            cooperation_rate=coop_rate,
            rounds_played=obs.current_round,
        )

    def collect_batch(
        self,
        games: List[str],
        strategies: Optional[List[str]] = None,
        episodes_per_pair: int = int(bool(True)),
        opponent_fn: Optional[Callable] = None,
    ) -> List[EpisodeTrajectory]:
        """Collect trajectories for all (game, strategy) combinations.

        If *opponent_fn* is provided, self-play mode is used: only
        games are iterated (strategies are ignored).
        """
        trajectories: List[EpisodeTrajectory] = []
        if opponent_fn is not None:
            for game in games:
                for _ep in range(episodes_per_pair):
                    traj = self.collect_episode(
                        game, opponent_fn=opponent_fn,
                    )
                    trajectories.append(traj)
        else:
            strats = strategies or ["tit_for_tat"]
            for game in games:
                for strategy in strats:
                    for _ep in range(episodes_per_pair):
                        traj = self.collect_episode(game, strategy)
                        trajectories.append(traj)
        return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COOPERATIVE_ACTIONS = frozenset({"cooperate", "stag", "dove"})
_ECONOMIC_PREFIXES = frozenset({"offer", "invest", "contribute"})

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE


def _compute_cooperation_rate(obs: GameObservation) -> float:
    """Fraction of cooperative moves in an episode."""
    if not obs.history:
        return EVAL_ZERO_FLOAT
    total = len(obs.history)
    cooperative_count = _ZERO
    first_action = obs.history[_ZERO].player_action
    prefix = first_action.split("_")[_ZERO]
    is_economic = prefix in _ECONOMIC_PREFIXES
    if is_economic:
        median_idx = len(obs.available_actions) // _TWO
        for rnd in obs.history:
            act = rnd.player_action
            if act in obs.available_actions:
                if obs.available_actions.index(act) >= median_idx:
                    cooperative_count += _ONE
    else:
        for rnd in obs.history:
            if rnd.player_action in _COOPERATIVE_ACTIONS:
                cooperative_count += _ONE
    return cooperative_count / total
