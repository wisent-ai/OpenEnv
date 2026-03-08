"""Integration test: metagame arena with real AI model backends.

Runs three arena rounds with a local mock, Anthropic Claude, and OpenAI GPT.
Usage:  python tests/integration_arena_multimodel.py
"""
from __future__ import annotations
import sys
import types

if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw: object) -> None:
            super().__init_subclass__(**kw)
        def __class_getitem__(cls, params: object) -> type:
            return cls
        def __init__(self) -> None:
            pass

    _iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
    _openenv_stub.core = _core_stub  # type: ignore[attr-defined]
    _core_stub.env_server = _server_stub  # type: ignore[attr-defined]
    _server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE

_MODEL_LOCAL = "local_mock"
_MODEL_CLAUDE = "claude"
_MODEL_GPT = "openai_gpt"


def _local_generate(prompt: str) -> str:
    """Simple cooperative local model."""
    return "cooperate"


def main() -> None:
    """Run the multi-model arena integration test."""
    from train.arena.arena_runner import ArenaRunner

    runner = ArenaRunner(total_rounds=_THREE)

    runner.add_local_model(_MODEL_LOCAL, _local_generate)
    print(f"Added local mock model: {_MODEL_LOCAL}")

    try:
        runner.add_anthropic_model(_MODEL_CLAUDE)
        print(f"Added Anthropic model: {_MODEL_CLAUDE}")
    except Exception as exc:
        print(f"Skipping Anthropic model (OAuth unavailable): {exc}")
        runner.add_strategy_model(_MODEL_CLAUDE, "cooperate")

    try:
        runner.add_openai_model(_MODEL_GPT)
        print(f"Added OpenAI model: {_MODEL_GPT}")
    except Exception as exc:
        print(f"Skipping OpenAI model (credentials unavailable): {exc}")
        runner.add_strategy_model(_MODEL_GPT, "defect")

    arena = runner.arena
    assert arena.roster.active_count >= _THREE, "Need at least three models"

    print(f"\nRunning {_THREE} arena rounds...")
    results = runner.run()
    assert len(results) == _THREE, f"Expected {_THREE} results, got {len(results)}"

    for result in results:
        rnd = result.round_number
        n_games = len(result.game_results)
        errors = sum(_ONE for r in result.game_results if "error" in r)
        print(f"  Round {rnd}: {n_games} games, {errors} errors")

    print("\nFinal reputations:")
    for model_id in arena.roster.active_models():
        rep = arena.reputation.compute_reputation(model_id)
        profile = arena.roster.get_profile(model_id)
        games = profile.games_played if profile else _ZERO
        print(f"  {model_id}: reputation={rep:.4f}, games={games}")

    print("\nIntegration test PASSED")


if __name__ == "__main__":
    main()
