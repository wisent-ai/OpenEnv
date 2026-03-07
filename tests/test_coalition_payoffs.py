"""Tests for coalition payoff computation and game payoff functions."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

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

from constant_definitions.nplayer.coalition_constants import (
    ENFORCEMENT_CHEAP_TALK,
    ENFORCEMENT_PENALTY,
    ENFORCEMENT_BINDING,
    COALITION_DEFAULT_PENALTY_NUMERATOR,
    COALITION_DEFAULT_PENALTY_DENOMINATOR,
    CARTEL_COLLUDE_HIGH,
    CARTEL_COLLUDE_LOW,
    CARTEL_COMPETE_HIGH,
    CARTEL_COMPETE_LOW,
    ALLIANCE_SUPPORT_POOL,
    ALLIANCE_BETRAY_GAIN,
    ALLIANCE_NO_SUPPORT,
    VOTING_WINNER_PAYOFF,
    VOTING_LOSER_PAYOFF,
    OSTRACISM_BONUS_POOL,
    OSTRACISM_EXCLUDED_PAYOFF,
    OSTRACISM_BASE_PAYOFF,
    TRADE_DIVERSE_PAYOFF,
    TRADE_HOMOGENEOUS_PAYOFF,
    TRADE_MINORITY_BONUS,
    RULE_EQUAL_PAY,
    RULE_WINNER_HIGH,
    RULE_WINNER_LOW,
    COMMONS_LOW_SUSTAINABLE,
    COMMONS_HIGH_SUSTAINABLE,
    COMMONS_LOW_DEPLETED,
    COMMONS_HIGH_DEPLETED,
)
from env.nplayer.coalition.models import ActiveCoalition
from env.nplayer.coalition.payoffs import compute_coalition_payoffs
from common.games_meta.coalition_config import (
    _cartel_payoff,
    _alliance_payoff,
    _coalition_voting_payoff,
    _ostracism_payoff,
    _resource_trading_payoff,
    _rule_voting_payoff,
    _commons_governance_payoff,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE
_SIX = _FIVE + _ONE
_TEN = _FIVE + _FIVE
_ZERO_F = float()
_TWO_F = float(_TWO)
_FOUR_F = float(_FOUR)
_SIX_F = float(_SIX)
_TEN_F = float(_TEN)
_PEN_N = COALITION_DEFAULT_PENALTY_NUMERATOR
_PEN_D = COALITION_DEFAULT_PENALTY_DENOMINATOR


class TestComputeCoalitionPayoffs:
    def test_cheap_talk_no_adjustment(self) -> None:
        base = (_SIX_F, _SIX_F, _TEN_F)
        actions = ("collude", "collude", "compete")
        coalition = ActiveCoalition(
            members=[_ZERO, _ONE], agreed_action="collude",
        )
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base, actions, [coalition], ENFORCEMENT_CHEAP_TALK, _PEN_N, _PEN_D,
        )
        assert adjusted == base
        assert defectors == []
        assert all(p == _ZERO_F for p in penalties)

    def test_penalty_defector_loses(self) -> None:
        base = (_TEN_F, _SIX_F, _SIX_F)
        actions = ("compete", "collude", "collude")
        coalition = ActiveCoalition(
            members=[_ZERO, _ONE, _TWO], agreed_action="collude",
        )
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base, actions, [coalition], ENFORCEMENT_PENALTY, _PEN_N, _PEN_D,
        )
        assert _ZERO in defectors
        assert _ONE not in defectors
        assert _TWO not in defectors
        expected_penalty = _TEN_F * _PEN_N / _PEN_D
        assert penalties[_ZERO] == pytest.approx(expected_penalty)
        assert adjusted[_ZERO] == pytest.approx(_TEN_F - expected_penalty)
        assert adjusted[_ONE] == pytest.approx(_SIX_F)

    def test_binding_no_defectors(self) -> None:
        base = (_SIX_F, _SIX_F)
        actions = ("collude", "collude")
        coalition = ActiveCoalition(
            members=[_ZERO, _ONE], agreed_action="collude",
        )
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base, actions, [coalition], ENFORCEMENT_BINDING, _PEN_N, _PEN_D,
        )
        assert defectors == []
        assert adjusted == base

    def test_side_payments(self) -> None:
        base = (_TEN_F, _SIX_F, _SIX_F)
        actions = ("collude", "collude", "collude")
        coalition = ActiveCoalition(
            members=[_ZERO, _ONE, _TWO],
            agreed_action="collude",
            side_payment=_TWO_F,
        )
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base, actions, [coalition], ENFORCEMENT_CHEAP_TALK, _PEN_N, _PEN_D,
        )
        # Proposer pays side * num_other_members = two * two = four
        assert side_pmts[_ZERO] == pytest.approx(-_FOUR_F)
        assert side_pmts[_ONE] == pytest.approx(_TWO_F)
        assert side_pmts[_TWO] == pytest.approx(_TWO_F)
        assert adjusted[_ZERO] == pytest.approx(_SIX_F)
        eight_f = float(_SIX + _TWO)
        assert adjusted[_ONE] == pytest.approx(eight_f)

    def test_no_coalitions(self) -> None:
        five_f = float(_FIVE)
        base = (five_f, five_f)
        actions = ("collude", "compete")
        adjusted, defectors, penalties, side_pmts = compute_coalition_payoffs(
            base, actions, [], ENFORCEMENT_PENALTY, _PEN_N, _PEN_D,
        )
        assert adjusted == base
        assert defectors == []


class TestCartelPayoff:
    def test_all_collude(self) -> None:
        p = _cartel_payoff(("collude", "collude", "collude", "collude"))
        assert all(v == pytest.approx(float(CARTEL_COLLUDE_HIGH)) for v in p)

    def test_all_compete(self) -> None:
        p = _cartel_payoff(("compete", "compete", "compete", "compete"))
        assert all(v == pytest.approx(float(CARTEL_COMPETE_LOW)) for v in p)

    def test_one_defects(self) -> None:
        p = _cartel_payoff(("collude", "collude", "collude", "compete"))
        assert p[_ZERO] == pytest.approx(float(CARTEL_COLLUDE_HIGH))
        assert p[_THREE] == pytest.approx(float(CARTEL_COMPETE_HIGH))

    def test_cartel_fails(self) -> None:
        p = _cartel_payoff(("collude", "compete", "compete", "compete"))
        assert p[_ZERO] == pytest.approx(float(CARTEL_COLLUDE_LOW))
        assert p[_ONE] == pytest.approx(float(CARTEL_COMPETE_LOW))


class TestAlliancePayoff:
    def test_all_support(self) -> None:
        p = _alliance_payoff(("support", "support", "support", "support"))
        expected = float(ALLIANCE_SUPPORT_POOL) / _FOUR
        assert all(v == pytest.approx(expected) for v in p)

    def test_all_betray(self) -> None:
        p = _alliance_payoff(("betray", "betray", "betray", "betray"))
        assert all(v == pytest.approx(float(ALLIANCE_NO_SUPPORT)) for v in p)

    def test_mixed(self) -> None:
        p = _alliance_payoff(("support", "support", "betray", "betray"))
        assert p[_ZERO] == pytest.approx(float(ALLIANCE_SUPPORT_POOL) / _TWO)
        assert p[_TWO] == pytest.approx(float(ALLIANCE_BETRAY_GAIN))


class TestCoalitionVotingPayoff:
    def test_unanimous_a(self) -> None:
        p = _coalition_voting_payoff(("vote_A",) * _FIVE)
        assert all(v == pytest.approx(float(VOTING_WINNER_PAYOFF)) for v in p)

    def test_majority_a(self) -> None:
        actions = ("vote_A", "vote_A", "vote_A", "vote_B", "vote_B")
        p = _coalition_voting_payoff(actions)
        assert p[_ZERO] == pytest.approx(float(VOTING_WINNER_PAYOFF))
        assert p[_THREE] == pytest.approx(float(VOTING_LOSER_PAYOFF))


class TestOstracismPayoff:
    def test_exclusion(self) -> None:
        actions = ("exclude_2", "exclude_2", "exclude_none", "exclude_2", "exclude_none")
        p = _ostracism_payoff(actions)
        assert p[_TWO] == pytest.approx(float(OSTRACISM_EXCLUDED_PAYOFF))
        expected_share = float(OSTRACISM_BONUS_POOL) / _FOUR
        assert p[_ZERO] == pytest.approx(expected_share)

    def test_no_majority(self) -> None:
        actions = ("exclude_1", "exclude_2", "exclude_3", "exclude_none", "exclude_none")
        p = _ostracism_payoff(actions)
        assert all(v == pytest.approx(float(OSTRACISM_BASE_PAYOFF)) for v in p)


class TestResourceTradingPayoff:
    def test_all_same(self) -> None:
        p = _resource_trading_payoff(("produce_A",) * _FOUR)
        assert all(v == pytest.approx(float(TRADE_HOMOGENEOUS_PAYOFF)) for v in p)

    def test_diverse(self) -> None:
        actions = ("produce_A", "produce_A", "produce_B", "produce_B")
        p = _resource_trading_payoff(actions)
        assert all(v == pytest.approx(float(TRADE_DIVERSE_PAYOFF)) for v in p)

    def test_minority_bonus(self) -> None:
        actions = ("produce_A", "produce_B", "produce_B", "produce_B")
        p = _resource_trading_payoff(actions)
        expected_min = float(TRADE_DIVERSE_PAYOFF) + float(TRADE_MINORITY_BONUS)
        assert p[_ZERO] == pytest.approx(expected_min)
        assert p[_ONE] == pytest.approx(float(TRADE_DIVERSE_PAYOFF))


class TestRuleVotingPayoff:
    def test_equal_wins(self) -> None:
        actions = ("rule_equal", "rule_equal", "rule_winner", "rule_equal")
        p = _rule_voting_payoff(actions)
        assert all(v == pytest.approx(float(RULE_EQUAL_PAY)) for v in p)

    def test_winner_wins(self) -> None:
        actions = ("rule_winner", "rule_winner", "rule_winner", "rule_equal")
        p = _rule_voting_payoff(actions)
        assert p[_ZERO] == pytest.approx(float(RULE_WINNER_HIGH))
        assert p[_THREE] == pytest.approx(float(RULE_WINNER_LOW))


class TestCommonsGovernancePayoff:
    def test_sustainable(self) -> None:
        actions = ("extract_low", "extract_low", "extract_high", "extract_low", "extract_low")
        p = _commons_governance_payoff(actions)
        assert p[_ZERO] == pytest.approx(float(COMMONS_LOW_SUSTAINABLE))
        assert p[_TWO] == pytest.approx(float(COMMONS_HIGH_SUSTAINABLE))

    def test_depleted(self) -> None:
        actions = ("extract_high", "extract_high", "extract_high", "extract_low", "extract_low")
        p = _commons_governance_payoff(actions)
        assert p[_ZERO] == pytest.approx(float(COMMONS_HIGH_DEPLETED))
        assert p[_THREE] == pytest.approx(float(COMMONS_LOW_DEPLETED))
