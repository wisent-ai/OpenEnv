"""Tests for N-player built-in game payoff functions."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

# Stub the openenv package
_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")
class _EnvironmentStub:
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
    def __class_getitem__(cls, params: object) -> type:
        return cls
    def __init__(self) -> None:
        pass
_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv_stub.core = _core_stub  # type: ignore[attr-defined]
_core_stub.env_server = _server_stub  # type: ignore[attr-defined]
_server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
for _name, _mod in [
    ("openenv", _openenv_stub),
    ("openenv.core", _core_stub),
    ("openenv.core.env_server", _server_stub),
    ("openenv.core.env_server.interfaces", _iface_stub),
]:
    sys.modules[_name] = _mod

import pytest

from constant_definitions.nplayer.nplayer_constants import (
    NPG_ENDOWMENT,
    NPG_MULTIPLIER_NUMERATOR,
    NPG_MULTIPLIER_DENOMINATOR,
    NVD_BENEFIT,
    NVD_COST,
    NVD_NO_VOLUNTEER,
    NEF_ATTEND_REWARD,
    NEF_CROWD_PENALTY,
    NEF_STAY_HOME,
)
from common.games_meta.nplayer_config import NPLAYER_GAMES
import common.games_meta.nplayer_games  # noqa: F401 -- register built-ins

# ── test-local numeric helpers ──────────────────────────────────────────
_ZERO = int()
_ONE = int(bool(True))


class TestPublicGoodsPayoff:
    def test_all_contribute_full(self) -> None:
        game = NPLAYER_GAMES["nplayer_public_goods"]
        n = game.num_players
        action = f"contribute_{NPG_ENDOWMENT}"
        actions = tuple([action] * n)
        payoffs = game.payoff_fn(actions)
        total = n * NPG_ENDOWMENT
        pool = total * NPG_MULTIPLIER_NUMERATOR / NPG_MULTIPLIER_DENOMINATOR
        share = pool / n
        expected = float(NPG_ENDOWMENT - NPG_ENDOWMENT + share)
        for p in payoffs:
            assert p == pytest.approx(expected)

    def test_all_contribute_zero(self) -> None:
        game = NPLAYER_GAMES["nplayer_public_goods"]
        n = game.num_players
        actions = tuple(["contribute_0"] * n)
        payoffs = game.payoff_fn(actions)
        for p in payoffs:
            assert p == pytest.approx(float(NPG_ENDOWMENT))

    def test_free_rider_advantage(self) -> None:
        game = NPLAYER_GAMES["nplayer_public_goods"]
        n = game.num_players
        actions_list = ["contribute_0"] + [f"contribute_{NPG_ENDOWMENT}"] * (n - _ONE)
        payoffs = game.payoff_fn(tuple(actions_list))
        assert payoffs[_ZERO] > payoffs[_ONE]


class TestVolunteerDilemmaPayoff:
    def test_all_volunteer(self) -> None:
        game = NPLAYER_GAMES["nplayer_volunteer_dilemma"]
        n = game.num_players
        actions = tuple(["volunteer"] * n)
        payoffs = game.payoff_fn(actions)
        expected = float(NVD_BENEFIT - NVD_COST)
        for p in payoffs:
            assert p == pytest.approx(expected)

    def test_one_volunteer(self) -> None:
        game = NPLAYER_GAMES["nplayer_volunteer_dilemma"]
        n = game.num_players
        actions = tuple(["volunteer"] + ["abstain"] * (n - _ONE))
        payoffs = game.payoff_fn(actions)
        assert payoffs[_ZERO] == pytest.approx(float(NVD_BENEFIT - NVD_COST))
        for i in range(_ONE, n):
            assert payoffs[i] == pytest.approx(float(NVD_BENEFIT))

    def test_nobody_volunteers(self) -> None:
        game = NPLAYER_GAMES["nplayer_volunteer_dilemma"]
        n = game.num_players
        actions = tuple(["abstain"] * n)
        payoffs = game.payoff_fn(actions)
        for p in payoffs:
            assert p == pytest.approx(float(NVD_NO_VOLUNTEER))


class TestElFarolPayoff:
    def test_all_stay_home(self) -> None:
        game = NPLAYER_GAMES["nplayer_el_farol"]
        n = game.num_players
        actions = tuple(["stay_home"] * n)
        payoffs = game.payoff_fn(actions)
        for p in payoffs:
            assert p == pytest.approx(float(NEF_STAY_HOME))

    def test_few_attend(self) -> None:
        game = NPLAYER_GAMES["nplayer_el_farol"]
        n = game.num_players
        actions = tuple(["attend"] + ["stay_home"] * (n - _ONE))
        payoffs = game.payoff_fn(actions)
        assert payoffs[_ZERO] == pytest.approx(float(NEF_ATTEND_REWARD))
        for i in range(_ONE, n):
            assert payoffs[i] == pytest.approx(float(NEF_STAY_HOME))

    def test_all_attend_crowded(self) -> None:
        game = NPLAYER_GAMES["nplayer_el_farol"]
        n = game.num_players
        actions = tuple(["attend"] * n)
        payoffs = game.payoff_fn(actions)
        for p in payoffs:
            assert p == pytest.approx(float(NEF_CROWD_PENALTY))
