"""Tests for adaptive payoff game factories and environment integration."""
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

import pytest
from common.games import GameConfig, GAME_FACTORIES, get_game
from env.environment import KantEnvironment
from env.models import GameAction
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_DD_PAYOFF, HD_HH_PAYOFF, EVAL_ZERO_FLOAT,
)
from constant_definitions.var.meta.adaptive_constants import (
    ADAPTIVE_GAME_TYPE, ADAPTIVE_DEFAULT_ROUNDS,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FIVE = _THREE + _TWO

_ADAPTIVE_KEYS = [
    "adaptive_prisoners_dilemma", "arms_race",
    "trust_erosion", "market_dynamics", "reputation_payoffs",
]


class TestGameFactoryRegistry:
    """Verify all adaptive games are registered and produce fresh state."""

    def test_all_factories_registered(self) -> None:
        for key in _ADAPTIVE_KEYS:
            assert key in GAME_FACTORIES

    @pytest.mark.parametrize("key", _ADAPTIVE_KEYS)
    def test_get_game_returns_game_config(self, key: str) -> None:
        cfg = get_game(key)
        assert isinstance(cfg, GameConfig)
        assert cfg.game_type == ADAPTIVE_GAME_TYPE

    @pytest.mark.parametrize("key", _ADAPTIVE_KEYS)
    def test_factory_returns_fresh_config(self, key: str) -> None:
        cfg_a = get_game(key)
        cfg_b = get_game(key)
        assert cfg_a is not cfg_b

    def test_state_isolation_between_calls(self) -> None:
        cfg_a = get_game("adaptive_prisoners_dilemma")
        for _ in range(_FIVE):
            cfg_a.payoff_fn("cooperate", "cooperate")
        cfg_b = get_game("adaptive_prisoners_dilemma")
        p_b, _ = cfg_b.payoff_fn("cooperate", "cooperate")
        # Fresh config has initial multiplier, so CC = base PD_CC
        assert p_b == float(PD_CC_PAYOFF)


class TestAdaptivePD:
    """Adaptive PD: multiplier grows with cooperation."""

    def test_mutual_coop_increases_multiplier(self) -> None:
        cfg = get_game("adaptive_prisoners_dilemma")
        p_first, _ = cfg.payoff_fn("cooperate", "cooperate")
        p_second, _ = cfg.payoff_fn("cooperate", "cooperate")
        assert p_second > p_first

    def test_mutual_defect_decreases_multiplier(self) -> None:
        cfg = get_game("adaptive_prisoners_dilemma")
        p_first, _ = cfg.payoff_fn("defect", "defect")
        p_second, _ = cfg.payoff_fn("defect", "defect")
        assert p_second < p_first

    def test_mixed_outcome_no_change(self) -> None:
        cfg = get_game("adaptive_prisoners_dilemma")
        p_first, _ = cfg.payoff_fn("cooperate", "defect")
        p_second, _ = cfg.payoff_fn("cooperate", "defect")
        assert p_first == p_second

    def test_actions_are_cooperate_defect(self) -> None:
        cfg = get_game("adaptive_prisoners_dilemma")
        assert cfg.actions == ["cooperate", "defect"]


class TestArmsRace:
    """Arms Race: hawk-hawk costs escalate."""

    def test_hawk_hawk_costs_escalate(self) -> None:
        cfg = get_game("arms_race")
        p_first, _ = cfg.payoff_fn("hawk", "hawk")
        p_second, _ = cfg.payoff_fn("hawk", "hawk")
        assert p_second < p_first

    def test_non_hawk_deescalates(self) -> None:
        cfg = get_game("arms_race")
        cfg.payoff_fn("hawk", "hawk")
        cfg.payoff_fn("hawk", "hawk")
        p_peak, _ = cfg.payoff_fn("hawk", "hawk")
        # De-escalate with several dove rounds
        cfg.payoff_fn("dove", "dove")
        cfg.payoff_fn("dove", "dove")
        cfg.payoff_fn("dove", "dove")
        p_after, _ = cfg.payoff_fn("hawk", "hawk")
        assert p_after > p_peak

    def test_actions_are_hawk_dove(self) -> None:
        cfg = get_game("arms_race")
        assert cfg.actions == ["hawk", "dove"]


class TestTrustErosion:
    """Trust Erosion: multiplier decays after mutual defection."""

    def test_defection_erodes_trust(self) -> None:
        cfg = get_game("trust_erosion")
        p_first, _ = cfg.payoff_fn("defect", "defect")
        p_second, _ = cfg.payoff_fn("defect", "defect")
        assert p_second < p_first

    def test_cooperation_recovers_trust(self) -> None:
        cfg = get_game("trust_erosion")
        cfg.payoff_fn("defect", "defect")
        cfg.payoff_fn("defect", "defect")
        p_low, _ = cfg.payoff_fn("cooperate", "cooperate")
        p_higher, _ = cfg.payoff_fn("cooperate", "cooperate")
        assert p_higher > p_low


class TestMarketDynamics:
    """Market Dynamics: demand shifts based on output."""

    def test_high_output_depresses_demand(self) -> None:
        cfg = get_game("market_dynamics")
        # Use medium to avoid zero-price edge case at intercept=total
        p_first, _ = cfg.payoff_fn("medium", "medium")
        p_second, _ = cfg.payoff_fn("medium", "medium")
        assert p_second < p_first

    def test_low_output_recovers_demand(self) -> None:
        cfg = get_game("market_dynamics")
        cfg.payoff_fn("high", "high")
        cfg.payoff_fn("high", "high")
        p_low_output, _ = cfg.payoff_fn("low", "low")
        p_after_recovery, _ = cfg.payoff_fn("low", "low")
        assert p_after_recovery >= p_low_output

    def test_actions_are_low_medium_high(self) -> None:
        cfg = get_game("market_dynamics")
        assert cfg.actions == ["low", "medium", "high"]


class TestReputationPayoffs:
    """Reputation Payoffs: cooperation bonus grows with history."""

    def test_coop_bonus_grows(self) -> None:
        cfg = get_game("reputation_payoffs")
        p_first, _ = cfg.payoff_fn("cooperate", "cooperate")
        p_second, _ = cfg.payoff_fn("cooperate", "cooperate")
        assert p_second > p_first

    def test_fresh_factory_resets_bonus(self) -> None:
        cfg_a = get_game("reputation_payoffs")
        cfg_a.payoff_fn("cooperate", "cooperate")
        cfg_a.payoff_fn("cooperate", "cooperate")
        cfg_b = get_game("reputation_payoffs")
        p_a, _ = cfg_a.payoff_fn("cooperate", "cooperate")
        p_b, _ = cfg_b.payoff_fn("cooperate", "cooperate")
        assert p_a > p_b


class TestAdaptiveEnvironment:
    """Full episode through KantEnvironment with adaptive games."""

    @pytest.mark.parametrize("key", _ADAPTIVE_KEYS)
    def test_full_episode(self, key: str) -> None:
        env = KantEnvironment()
        obs = env.reset(game=key, strategy="always_cooperate")
        assert not obs.done
        while not obs.done:
            action = GameAction(action=obs.available_actions[_ZERO])
            obs = env.step(action)
        assert obs.done
        assert obs.current_round == ADAPTIVE_DEFAULT_ROUNDS

    def test_payoffs_change_across_rounds(self) -> None:
        env = KantEnvironment()
        obs = env.reset(
            game="adaptive_prisoners_dilemma",
            strategy="always_cooperate",
        )
        payoffs = []
        while not obs.done:
            obs = env.step(GameAction(action="cooperate"))
            if obs.last_round is not None:
                payoffs.append(obs.last_round.player_payoff)
        # With mutual cooperation, multiplier grows
        assert payoffs[-_ONE] > payoffs[_ZERO]

    def test_independent_episode_state(self) -> None:
        env = KantEnvironment()
        obs = env.reset(
            game="adaptive_prisoners_dilemma",
            strategy="always_cooperate",
        )
        while not obs.done:
            obs = env.step(GameAction(action="cooperate"))
        first_final = obs.player_score
        obs = env.reset(
            game="adaptive_prisoners_dilemma",
            strategy="always_cooperate",
        )
        while not obs.done:
            obs = env.step(GameAction(action="cooperate"))
        assert obs.player_score == first_final
