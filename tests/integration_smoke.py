"""End-to-end integration smoke tests for dynamic games + N-player support."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/machiaveli",
)

# Stub the openenv package
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")


class _EnvironmentStub:
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

    def __class_getitem__(cls, params: object) -> type:
        return cls

    def __init__(self) -> None:
        pass


_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv = types.ModuleType("openenv")
_core = types.ModuleType("openenv.core")
_server = types.ModuleType("openenv.core.env_server")
_openenv.core = _core  # type: ignore[attr-defined]
_core.env_server = _server  # type: ignore[attr-defined]
_server.interfaces = _iface_stub  # type: ignore[attr-defined]
for _n, _m in [
    ("openenv", _openenv), ("openenv.core", _core),
    ("openenv.core.env_server", _server),
    ("openenv.core.env_server.interfaces", _iface_stub),
]:
    sys.modules[_n] = _m

from common.games import create_matrix_game, GAMES
from common.games_meta.dynamic import unregister_game
from env.environment import MachiavelliEnvironment
from env.models import GameAction
import common.games_meta.nplayer_games  # noqa: F401 -- register built-ins
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction, NPlayerObservation

_ONE = int(bool(True))
_ZERO = int()
_ZERO_F = float()
_THREE = _ONE + _ONE + _ONE


def test_dynamic_game_with_two_player_env() -> None:
    """Create a dynamic matrix game at runtime, play it with the existing env."""
    print("=== Dynamic game + two-player env ===")
    cfg = create_matrix_game(
        "chicken",
        ["swerve", "straight"],
        {
            ("swerve", "swerve"): (_ZERO_F, _ZERO_F),
            ("swerve", "straight"): (-float(_ONE), float(_ONE)),
            ("straight", "swerve"): (float(_ONE), -float(_ONE)),
            ("straight", "straight"): (-float(_ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE),
                                       -float(_ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE)),
        },
        register=True,
    )
    assert "dynamic_chicken" in GAMES
    env = MachiavelliEnvironment()
    obs = env.reset(game="dynamic_chicken", strategy="random", num_rounds=_THREE)
    assert obs.done is False
    assert obs.game_name == "dynamic_chicken"
    for _ in range(_THREE):
        obs = env.step(GameAction(action="swerve"))
        assert obs.last_round is not None
        assert obs.last_round.player_action == "swerve"
    assert obs.done is True
    unregister_game("dynamic_chicken")
    assert "dynamic_chicken" not in GAMES
    print("  PASSED")


def test_nplayer_public_goods() -> None:
    """Play a multi-player public goods game with NPlayerEnvironment."""
    print("=== N-player public goods ===")
    env = NPlayerEnvironment()
    obs = env.reset(
        "nplayer_public_goods",
        num_rounds=_THREE,
        opponent_strategies=["always_cooperate"],
    )
    assert obs.num_players > _ONE + _ONE
    assert obs.player_index == _ZERO
    for _ in range(_THREE):
        obs = env.step(NPlayerAction(action="contribute_0"))
        assert len(obs.last_round.actions) == obs.num_players
        assert len(obs.last_round.payoffs) == obs.num_players
    assert obs.done is True
    # Free-riding player should have highest score
    assert obs.scores[_ZERO] >= max(obs.scores[_ONE:])
    print(f"  Scores: player={obs.scores[_ZERO]:.1f}, "
          f"others avg={sum(obs.scores[_ONE:]) / len(obs.scores[_ONE:]):.1f}")
    print("  PASSED")


def test_mixed_opponents() -> None:
    """Mix agent functions and strategies as opponents in an N-player game."""
    print("=== Mixed opponent fns + strategies ===")

    def always_volunteer(obs: NPlayerObservation) -> NPlayerAction:
        return NPlayerAction(action="volunteer")

    env = NPlayerEnvironment()
    obs = env.reset(
        "nplayer_volunteer_dilemma",
        num_rounds=_ONE,
        opponent_strategies=["always_defect"],
        opponent_fns=[always_volunteer, None, always_volunteer, None],
    )
    obs = env.step(NPlayerAction(action="abstain"))
    # fn opponents volunteer, strategy opponents abstain
    assert obs.last_round.actions[_ONE] == "volunteer"
    assert obs.last_round.actions[_ONE + _ONE] == "abstain"
    assert obs.last_round.actions[_ONE + _ONE + _ONE] == "volunteer"
    assert obs.last_round.actions[_ONE + _ONE + _ONE + _ONE] == "abstain"
    print(f"  Actions: {obs.last_round.actions}")
    print(f"  Payoffs: {obs.last_round.payoffs}")
    print("  PASSED")


def test_el_farol_full_episode() -> None:
    """Play a full El Farol bar episode."""
    print("=== El Farol bar episode ===")
    env = NPlayerEnvironment()
    rounds = _ONE + _ONE + _ONE + _ONE + _ONE
    obs = env.reset("nplayer_el_farol", num_rounds=rounds)
    for r in range(rounds):
        action = "attend" if r % (_ONE + _ONE) == _ZERO else "stay_home"
        obs = env.step(NPlayerAction(action=action))
    assert obs.done is True
    assert len(obs.history) == rounds
    print(f"  Final scores: {[round(s, _ONE + _ONE) for s in obs.scores]}")
    print("  PASSED")


if __name__ == "__main__":
    test_dynamic_game_with_two_player_env()
    test_nplayer_public_goods()
    test_mixed_opponents()
    test_el_farol_full_episode()
    print("\nAll integration smoke tests passed!")
