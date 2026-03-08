"""Environment integration tests for meta-gaming variants."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
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
    sys.modules.setdefault(_name, _mod)

import pytest

from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
)
from constant_definitions.var.meta.meta_rule_constants import (
    COOP_BONUS_NUMERATOR, COOP_BONUS_DENOMINATOR,
    EQUAL_SPLIT_DENOMINATOR,
    META_RACCEPT_PREFIX, META_RREJECT_PREFIX, META_RPROP_PREFIX,
)
from env.models import GameAction, GameObservation
from env.environment import KantEnvironment
from common.games import GAMES
from common.meta.variants_meta import (
    apply_proposer_responder, apply_constitutional,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE

_CC = float(PD_CC_PAYOFF)
_CD = float(PD_CD_PAYOFF)
_DC = float(PD_DC_PAYOFF)
_DD = float(PD_DD_PAYOFF)
_COOP_B = COOP_BONUS_NUMERATOR / COOP_BONUS_DENOMINATOR


@pytest.fixture()
def env() -> KantEnvironment:
    return KantEnvironment()


class TestMetaEnvironment:
    def test_reset_with_rule_proposal(self, env: KantEnvironment) -> None:
        obs = env.reset(
            game="rule_proposal_prisoners_dilemma",
            strategy="always_cooperate",
        )
        assert not obs.done
        assert len(obs.available_actions) > _TWO

    def test_step_valid_action(self, env: KantEnvironment) -> None:
        obs = env.reset(
            game="rule_proposal_prisoners_dilemma",
            strategy="always_cooperate",
        )
        action = obs.available_actions[_ZERO]
        obs = env.step(GameAction(action=action))
        assert obs.current_round == _ONE

    def test_invalid_action_raises(self, env: KantEnvironment) -> None:
        env.reset(
            game="rule_proposal_prisoners_dilemma",
            strategy="always_cooperate",
        )
        with pytest.raises(ValueError):
            env.step(GameAction(action="bad_action"))

    def test_signal_payoff_is_base(self, env: KantEnvironment) -> None:
        env.reset(
            game="rule_signal_prisoners_dilemma",
            strategy="always_cooperate",
            num_rounds=_ONE,
        )
        obs = env.step(GameAction(action="sig_equalsplit_cooperate"))
        assert obs.reward == _CC


class TestProposerResponderEnv:
    _game = apply_proposer_responder(
        GAMES["prisoners_dilemma"], base_key="prisoners_dilemma",
    )

    def test_opponent_gets_responder_actions(self) -> None:
        assert self._game.opponent_actions is not None
        opp = list(self._game.opponent_actions)
        assert any(a.startswith(META_RACCEPT_PREFIX) for a in opp)
        assert any(a.startswith(META_RREJECT_PREFIX) for a in opp)

    def test_player_gets_proposer_actions(self) -> None:
        assert all(a.startswith(META_RPROP_PREFIX) for a in self._game.actions)


class TestConstitutionalEnv:
    def test_locks_across_rounds(self) -> None:
        base = GAMES["prisoners_dilemma"]
        game = apply_constitutional(base, base_key="prisoners_dilemma")
        p1, _ = game.payoff_fn(
            "const_coopbonus_cooperate", "const_coopbonus_cooperate",
        )
        assert p1 == _CC + _COOP_B
        p2, _ = game.payoff_fn(
            "const_none_defect", "const_none_defect",
        )
        assert p2 == _DD

    def test_resets_between_episodes(self) -> None:
        base = GAMES["prisoners_dilemma"]
        game_a = apply_constitutional(base, base_key="prisoners_dilemma")
        game_a.payoff_fn(
            "const_coopbonus_cooperate", "const_coopbonus_cooperate",
        )
        game_b = apply_constitutional(base, base_key="prisoners_dilemma")
        p, _ = game_b.payoff_fn(
            "const_none_cooperate", "const_equalsplit_cooperate",
        )
        assert p == _CC
