"""KantBench Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import KantBenchAction, KantBenchObservation


class KantBenchEnv(
    EnvClient[KantBenchAction, KantBenchObservation]
):
    """
    Client for the KantBench game theory environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with KantBenchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.game_name)
        ...     print(result.observation.available_moves)
        ...
        ...     result = client.step(KantBenchAction(move="cooperate"))
        ...     print(result.observation.your_payoff)

    Example with HF Space:
        >>> with KantBenchEnv(base_url="https://openenv-community-kantbench.hf.space") as client:
        ...     result = client.reset()
        ...     result = client.step(KantBenchAction(move="cooperate"))
    """

    def _step_payload(self, action: KantBenchAction) -> Dict:
        return {"move": action.move}

    def _parse_result(self, payload: Dict) -> StepResult[KantBenchObservation]:
        obs_data = payload.get("observation", {})
        observation = KantBenchObservation(
            game_name=obs_data.get("game_name", ""),
            game_description=obs_data.get("game_description", ""),
            available_moves=obs_data.get("available_moves", []),
            your_move=obs_data.get("your_move", ""),
            opponent_move=obs_data.get("opponent_move", ""),
            your_payoff=obs_data.get("your_payoff", 0.0),
            opponent_payoff=obs_data.get("opponent_payoff", 0.0),
            cumulative_score=obs_data.get("cumulative_score", 0.0),
            round_number=obs_data.get("round_number", 0),
            max_rounds=obs_data.get("max_rounds", 10),
            opponent_strategy=obs_data.get("opponent_strategy", ""),
            history=obs_data.get("history", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
