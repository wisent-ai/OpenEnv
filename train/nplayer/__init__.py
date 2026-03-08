"""N-player and coalition LLM agents for game-theory environments."""

__all__ = [
    "NPlayerLLMAgent",
    "NPlayerPromptBuilder",
    "CoalitionLLMAgent",
    "CoalitionPromptBuilder",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid pulling in heavy dependencies at load time."""
    if name in ("NPlayerLLMAgent", "NPlayerPromptBuilder"):
        from train.nplayer.nplayer_agent import (
            NPlayerLLMAgent,
            NPlayerPromptBuilder,
        )
        _map = {
            "NPlayerLLMAgent": NPlayerLLMAgent,
            "NPlayerPromptBuilder": NPlayerPromptBuilder,
        }
        return _map[name]
    if name in ("CoalitionLLMAgent", "CoalitionPromptBuilder"):
        from train.nplayer.coalition_agent import (
            CoalitionLLMAgent,
            CoalitionPromptBuilder,
        )
        _map = {
            "CoalitionLLMAgent": CoalitionLLMAgent,
            "CoalitionPromptBuilder": CoalitionPromptBuilder,
        }
        return _map[name]
    msg = f"module 'train.nplayer' has no attribute {name!r}"
    raise AttributeError(msg)
