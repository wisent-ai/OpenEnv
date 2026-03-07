"""Training pipeline for strategic reasoning via game-theory environments."""

__all__ = [
    "LLMAgent",
    "PromptBuilder",
    "parse_action",
    "episode_reward",
    "get_train_eval_split",
    "EpisodeTrajectory",
    "StepRecord",
    "TrajectoryCollector",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid pulling in openenv at package load time."""
    if name in ("LLMAgent", "PromptBuilder", "parse_action"):
        from train.agent import LLMAgent, PromptBuilder, parse_action
        _map = {
            "LLMAgent": LLMAgent,
            "PromptBuilder": PromptBuilder,
            "parse_action": parse_action,
        }
        return _map[name]
    if name == "episode_reward":
        from train.rewards import episode_reward
        return episode_reward
    if name == "get_train_eval_split":
        from train.splits import get_train_eval_split
        return get_train_eval_split
    if name in ("EpisodeTrajectory", "StepRecord", "TrajectoryCollector"):
        from train.trajectory import (
            EpisodeTrajectory, StepRecord, TrajectoryCollector,
        )
        _map = {
            "EpisodeTrajectory": EpisodeTrajectory,
            "StepRecord": StepRecord,
            "TrajectoryCollector": TrajectoryCollector,
        }
        return _map[name]
    msg = f"module 'train' has no attribute {name!r}"
    raise AttributeError(msg)
