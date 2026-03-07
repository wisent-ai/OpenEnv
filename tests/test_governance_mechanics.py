"""Tests for governance mechanic functions."""
import sys
import types

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")
class _EnvironmentStub:
    def __init_subclass__(cls, **kwargs: object) -> None: super().__init_subclass__(**kwargs)
    def __class_getitem__(cls, params: object) -> type: return cls
    def __init__(self) -> None: pass
_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv_stub.core = _core_stub  # type: ignore[attr-defined]
_core_stub.env_server = _server_stub  # type: ignore[attr-defined]
_server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
for _name, _mod in [("openenv", _openenv_stub), ("openenv.core", _core_stub),
                     ("openenv.core.env_server", _server_stub),
                     ("openenv.core.env_server.interfaces", _iface_stub)]:
    sys.modules[_name] = _mod

import pytest

from constant_definitions.nplayer.governance_constants import (
    MECHANIC_TAXATION, MECHANIC_REDISTRIBUTION, MECHANIC_INSURANCE,
    MECHANIC_QUOTA, MECHANIC_SUBSIDY, MECHANIC_VETO,
    REDISTRIBUTION_EQUAL, REDISTRIBUTION_PROPORTIONAL,
)
from env.nplayer.governance.models import MechanicConfig, RuntimeRules
from env.nplayer.governance.mechanics import (
    _apply_taxation, _apply_redistribution, _apply_insurance,
    _apply_quota, _apply_subsidy, _apply_veto, apply_mechanics,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_ZERO_F = float()
_TEN = _FOUR + _FOUR + _TWO


class TestTaxation:
    def test_equal_redistribution(self) -> None:
        payoffs = [float(_TEN), float(_TEN), float(_TEN)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(tax_rate_numerator=_ONE, tax_rate_denominator=_TEN)
        result = _apply_taxation(payoffs, active, cfg)
        for r in result:
            assert r == pytest.approx(float(_TEN))

    def test_unequal_payoffs(self) -> None:
        payoffs = [float(_TEN + _TEN), float(_TEN), _ZERO_F]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(tax_rate_numerator=_ONE, tax_rate_denominator=_TEN)
        result = _apply_taxation(payoffs, active, cfg)
        total_before = sum(payoffs[i] for i in active)
        total_after = sum(result[i] for i in active)
        assert total_after == pytest.approx(total_before)
        assert result[_ZERO] < payoffs[_ZERO]
        assert result[_TWO] > payoffs[_TWO]

    def test_inactive_unaffected(self) -> None:
        payoffs = [float(_TEN), float(_TEN + _TEN), float(_TEN)]
        active = {_ZERO, _TWO}
        cfg = MechanicConfig(tax_rate_numerator=_ONE, tax_rate_denominator=_TEN)
        result = _apply_taxation(payoffs, active, cfg)
        assert result[_ONE] == payoffs[_ONE]


class TestRedistribution:
    def test_equal_mode(self) -> None:
        payoffs = [float(_TEN + _TEN), float(_TEN), _ZERO_F]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(redistribution_mode=REDISTRIBUTION_EQUAL)
        result = _apply_redistribution(payoffs, active, cfg)
        mean = float(_TEN)
        for i in active:
            assert result[i] == pytest.approx(mean)

    def test_proportional_mode(self) -> None:
        payoffs = [float(_TEN + _TEN), _ZERO_F]
        active = {_ZERO, _ONE}
        cfg = MechanicConfig(
            redistribution_mode=REDISTRIBUTION_PROPORTIONAL,
            damping_numerator=_ONE, damping_denominator=_TWO)
        result = _apply_redistribution(payoffs, active, cfg)
        mean = float(_TEN)
        assert result[_ZERO] == pytest.approx(float(_TEN) + float(_TEN - _TEN // _TWO))
        assert result[_ONE] == pytest.approx(float(_TEN // _TWO))


class TestInsurance:
    def test_below_threshold_receives(self) -> None:
        payoffs = [float(_TEN), float(_TEN), float(_ONE)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(
            insurance_contribution_numerator=_ONE,
            insurance_contribution_denominator=_TEN,
            insurance_threshold_numerator=_ONE,
            insurance_threshold_denominator=_TWO)
        result = _apply_insurance(payoffs, active, cfg)
        assert result[_TWO] > payoffs[_TWO]
        total_before = sum(payoffs[i] for i in active)
        total_after = sum(result[i] for i in active)
        assert total_after == pytest.approx(total_before)

    def test_all_above_threshold(self) -> None:
        payoffs = [float(_TEN), float(_TEN), float(_TEN)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(
            insurance_contribution_numerator=_ONE,
            insurance_contribution_denominator=_TEN,
            insurance_threshold_numerator=_ONE,
            insurance_threshold_denominator=_TWO)
        result = _apply_insurance(payoffs, active, cfg)
        for i in active:
            assert result[i] == pytest.approx(payoffs[i] * (float(_TEN) - float(_ONE)) / float(_TEN))


class TestQuota:
    def test_cap_and_redistribute(self) -> None:
        payoffs = [float(_TEN), float(_TWO), float(_TWO)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(quota_max=float(_FOUR + _TWO))
        result = _apply_quota(payoffs, active, cfg)
        assert result[_ZERO] == pytest.approx(float(_FOUR + _TWO))
        assert result[_ONE] > payoffs[_ONE]
        assert result[_TWO] > payoffs[_TWO]

    def test_no_cap_needed(self) -> None:
        payoffs = [float(_THREE), float(_TWO), float(_ONE)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(quota_max=float(_TEN))
        result = _apply_quota(payoffs, active, cfg)
        for i in active:
            assert result[i] == pytest.approx(payoffs[i])


class TestSubsidy:
    def test_floor_applied(self) -> None:
        payoffs = [float(_TEN), float(_ONE), _ZERO_F]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(
            subsidy_floor=float(_THREE),
            subsidy_fund_rate_numerator=_ONE,
            subsidy_fund_rate_denominator=_TWO)
        result = _apply_subsidy(payoffs, active, cfg)
        assert result[_ONE] > payoffs[_ONE]
        assert result[_TWO] > payoffs[_TWO]
        assert result[_ZERO] < payoffs[_ZERO]


class TestVeto:
    def test_veto_triggers_equalization(self) -> None:
        payoffs = [float(_ONE), float(_TEN), float(_TEN)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(veto_player=_ZERO)
        result = _apply_veto(payoffs, active, cfg)
        mean = sum(payoffs[i] for i in active) / len(active)
        for i in active:
            assert result[i] == pytest.approx(mean)

    def test_veto_no_trigger(self) -> None:
        payoffs = [float(_TEN), float(_ONE), float(_ONE)]
        active = {_ZERO, _ONE, _TWO}
        cfg = MechanicConfig(veto_player=_ZERO)
        result = _apply_veto(payoffs, active, cfg)
        for i in active:
            assert result[i] == pytest.approx(payoffs[i])

    def test_veto_player_inactive(self) -> None:
        payoffs = [float(_ONE), float(_TEN), float(_TEN)]
        active = {_ONE, _TWO}
        cfg = MechanicConfig(veto_player=_ZERO)
        result = _apply_veto(payoffs, active, cfg)
        for i in range(len(payoffs)):
            assert result[i] == pytest.approx(payoffs[i])


class TestMechanicComposition:
    def test_taxation_then_quota(self) -> None:
        payoffs = [float(_TEN + _TEN), float(_TWO), float(_TWO)]
        active = {_ZERO, _ONE, _TWO}
        rules = RuntimeRules(
            mechanics={
                MECHANIC_TAXATION: True, MECHANIC_REDISTRIBUTION: False,
                MECHANIC_INSURANCE: False, MECHANIC_QUOTA: True,
                MECHANIC_SUBSIDY: False, MECHANIC_VETO: False,
            },
            mechanic_config=MechanicConfig(
                tax_rate_numerator=_ONE, tax_rate_denominator=_TEN,
                quota_max=float(_TEN)))
        result = apply_mechanics(payoffs, rules, active)
        assert result[_ZERO] <= float(_TEN)

    def test_no_mechanics_passthrough(self) -> None:
        payoffs = [float(_TEN), float(_TEN)]
        active = {_ZERO, _ONE}
        rules = RuntimeRules(mechanics={MECHANIC_TAXATION: False, MECHANIC_QUOTA: False})
        result = apply_mechanics(payoffs, rules, active)
        for i in range(len(payoffs)):
            assert result[i] == pytest.approx(payoffs[i])

    def test_ordering_matters(self) -> None:
        payoffs = [float(_TEN + _TEN), float(_TWO)]
        active = {_ZERO, _ONE}
        rules_tax_first = RuntimeRules(
            mechanics={
                MECHANIC_TAXATION: True, MECHANIC_REDISTRIBUTION: False,
                MECHANIC_INSURANCE: False, MECHANIC_QUOTA: True,
                MECHANIC_SUBSIDY: False, MECHANIC_VETO: False,
            },
            mechanic_config=MechanicConfig(
                tax_rate_numerator=_THREE, tax_rate_denominator=_TEN,
                quota_max=float(_TEN)))
        result = apply_mechanics(payoffs, rules_tax_first, active)
        total_after = sum(result)
        total_before = sum(payoffs)
        assert total_after == pytest.approx(total_before)
