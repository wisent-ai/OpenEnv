"""List the live game registry.

Imports ``common.games`` (which triggers extension loading) and prints
the union of ``GAMES.keys()`` and ``GAME_FACTORIES.keys()`` -- i.e. the
set of game names an LLM agent connecting to ``env.app`` can actually
reset the environment to.

Run from the repo root:

    PYTHONPATH=. python3 scripts/diag/list_games.py
"""

from __future__ import annotations

import sys
import types

# The env.* modules import openenv.core; common.games does not, but
# extensions sometimes do, so stub it the same way the test suite does.
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
        ("openenv", _openenv_stub),
        ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

from common.games import GAMES, GAME_FACTORIES  # noqa: E402


def main() -> None:
    static_keys = sorted(GAMES.keys())
    factory_keys = sorted(GAME_FACTORIES.keys())
    union = sorted(set(static_keys) | set(factory_keys))
    print(f"Static GAMES entries:    {len(static_keys)}")
    print(f"GAME_FACTORIES entries:  {len(factory_keys)}")
    print(f"Union (live registry):   {len(union)}")
    print()
    print("All game keys:")
    for key in union:
        cfg = GAMES.get(key)
        if cfg is None and key in GAME_FACTORIES:
            cfg = GAME_FACTORIES[key]()
        rounds = cfg.default_rounds if cfg is not None else "?"
        n_actions = len(cfg.actions) if cfg is not None else "?"
        np = cfg.num_players if cfg is not None else "?"
        print(f"  {key:40s}  players={np}  rounds={rounds}  actions={n_actions}")


if __name__ == "__main__":
    main()
