"""Tests for the meta-gaming rule catalog and variant transforms."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from common.games import GAMES, get_game
from common.meta.meta_rules import (
    apply_rule, parse_meta_action, RULE_CATALOG,
    _is_cooperative,
)
from common.meta.variants_meta import (
    apply_rule_proposal, apply_rule_signal,
    apply_constitutional, apply_proposer_responder,
)
from common.variants import compose_game
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
)
from constant_definitions.var.meta.meta_rule_constants import (
    RULE_NONE, RULE_EQUAL_SPLIT, RULE_COOP_BONUS,
    RULE_DEFECT_PENALTY, RULE_MIN_GUARANTEE, RULE_BAN_DEFECT,
    DEFAULT_RULE_CATALOG,
    VARIANT_RULE_PROPOSAL, VARIANT_RULE_SIGNAL,
    VARIANT_CONSTITUTIONAL, VARIANT_PROPOSER_RESPONDER,
    COOP_BONUS_NUMERATOR, COOP_BONUS_DENOMINATOR,
    DEFECT_PENALTY_NUMERATOR, DEFECT_PENALTY_DENOMINATOR,
    MIN_GUARANTEE_NUMERATOR, MIN_GUARANTEE_DENOMINATOR,
    BAN_DEFECT_PENALTY_NUMERATOR, BAN_DEFECT_PENALTY_DENOMINATOR,
    EQUAL_SPLIT_DENOMINATOR,
    META_PROP_PREFIX, META_SIG_PREFIX,
    META_RPROP_PREFIX, META_RACCEPT_PREFIX, META_RREJECT_PREFIX,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE

_PD = GAMES["prisoners_dilemma"]
_CC = float(PD_CC_PAYOFF)
_CD = float(PD_CD_PAYOFF)
_DC = float(PD_DC_PAYOFF)
_DD = float(PD_DD_PAYOFF)

_COOP_B = COOP_BONUS_NUMERATOR / COOP_BONUS_DENOMINATOR
_DEF_P = DEFECT_PENALTY_NUMERATOR / DEFECT_PENALTY_DENOMINATOR
_MIN_G = MIN_GUARANTEE_NUMERATOR / MIN_GUARANTEE_DENOMINATOR
_BAN_P = BAN_DEFECT_PENALTY_NUMERATOR / BAN_DEFECT_PENALTY_DENOMINATOR


class TestRuleCatalog:
    def test_catalog_has_all_defaults(self) -> None:
        for rule in DEFAULT_RULE_CATALOG:
            assert rule in RULE_CATALOG

    def test_rule_none(self) -> None:
        p, o = apply_rule(RULE_NONE, _CC, _CC, "cooperate", "cooperate")
        assert p == _CC and o == _CC

    def test_equal_split_symmetric(self) -> None:
        p, o = apply_rule(RULE_EQUAL_SPLIT, _DC, _CD, "defect", "cooperate")
        expected = (_DC + _CD) / EQUAL_SPLIT_DENOMINATOR
        assert p == expected and o == expected

    def test_coop_bonus_cooperative(self) -> None:
        p, o = apply_rule(RULE_COOP_BONUS, _CC, _CC, "cooperate", "cooperate")
        assert p == _CC + _COOP_B and o == _CC + _COOP_B

    def test_coop_bonus_defect_no_bonus(self) -> None:
        p, o = apply_rule(RULE_COOP_BONUS, _DD, _DD, "defect", "defect")
        assert p == _DD and o == _DD

    def test_defect_penalty_cooperative_no_penalty(self) -> None:
        p, o = apply_rule(
            RULE_DEFECT_PENALTY, _CC, _CC, "cooperate", "cooperate",
        )
        assert p == _CC and o == _CC

    def test_defect_penalty_defects(self) -> None:
        p, o = apply_rule(
            RULE_DEFECT_PENALTY, _DD, _DD, "defect", "defect",
        )
        assert p == _DD - _DEF_P and o == _DD - _DEF_P

    def test_min_guarantee_floor(self) -> None:
        p, o = apply_rule(RULE_MIN_GUARANTEE, _CD, _DC, "cooperate", "defect")
        assert p == _MIN_G and o == _DC

    def test_ban_defect(self) -> None:
        p, o = apply_rule(RULE_BAN_DEFECT, _DC, _CD, "defect", "cooperate")
        assert p == _DC - _BAN_P and o == _CD


class TestCooperativeDetection:
    @pytest.mark.parametrize("action", ["cooperate", "stag", "dove"])
    def test_cooperative(self, action: str) -> None:
        assert _is_cooperative(action) is True

    @pytest.mark.parametrize("action", ["defect", "hawk", "hare"])
    def test_not_cooperative(self, action: str) -> None:
        assert _is_cooperative(action) is False


class TestParseMetaAction:
    def test_prop_action(self) -> None:
        prefix, rule, act = parse_meta_action("prop_equalsplit_cooperate")
        assert prefix == META_PROP_PREFIX
        assert rule == RULE_EQUAL_SPLIT
        assert act == "cooperate"

    def test_sig_action(self) -> None:
        prefix, rule, act = parse_meta_action("sig_none_defect")
        assert prefix == META_SIG_PREFIX
        assert rule == RULE_NONE
        assert act == "defect"


class TestApplyRuleProposal:
    _game = apply_rule_proposal(_PD, base_key="prisoners_dilemma")

    def test_action_count(self) -> None:
        n_rules = len(DEFAULT_RULE_CATALOG)
        n_base = len(_PD.actions)
        assert len(self._game.actions) == n_rules * n_base

    def test_variant_tracked(self) -> None:
        assert VARIANT_RULE_PROPOSAL in self._game.applied_variants

    def test_agreement_applies_rule(self) -> None:
        p, o = self._game.payoff_fn(
            "prop_equalsplit_cooperate", "prop_equalsplit_defect",
        )
        expected = (_CD + _DC) / EQUAL_SPLIT_DENOMINATOR
        assert p == expected and o == expected

    def test_disagreement_returns_base(self) -> None:
        p, o = self._game.payoff_fn(
            "prop_equalsplit_cooperate", "prop_coopbonus_cooperate",
        )
        assert p == _CC and o == _CC

    def test_none_agreement_is_identity(self) -> None:
        p, o = self._game.payoff_fn(
            "prop_none_defect", "prop_none_cooperate",
        )
        assert p == _DC and o == _CD


class TestApplyRuleSignal:
    _game = apply_rule_signal(_PD, base_key="prisoners_dilemma")

    def test_variant_tracked(self) -> None:
        assert VARIANT_RULE_SIGNAL in self._game.applied_variants

    def test_signal_never_affects_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "sig_equalsplit_cooperate", "sig_equalsplit_defect",
        )
        assert p == _CD and o == _DC

    def test_any_signal_same_payoff(self) -> None:
        p_a, o_a = self._game.payoff_fn(
            "sig_coopbonus_defect", "sig_bandefect_cooperate",
        )
        p_b, o_b = self._game.payoff_fn(
            "sig_none_defect", "sig_none_cooperate",
        )
        assert p_a == p_b and o_a == o_b


class TestApplyConstitutional:
    def test_no_agreement_returns_base(self) -> None:
        game = apply_constitutional(_PD, base_key="prisoners_dilemma")
        p, o = game.payoff_fn(
            "const_equalsplit_cooperate", "const_coopbonus_cooperate",
        )
        assert p == _CC and o == _CC

    def test_agreement_locks_in(self) -> None:
        game = apply_constitutional(_PD, base_key="prisoners_dilemma")
        p1, _ = game.payoff_fn(
            "const_coopbonus_cooperate", "const_coopbonus_cooperate",
        )
        assert p1 == _CC + _COOP_B
        p2, _ = game.payoff_fn(
            "const_none_defect", "const_none_defect",
        )
        assert p2 == _DD

    def test_fresh_config_resets(self) -> None:
        game_a = apply_constitutional(_PD, base_key="prisoners_dilemma")
        game_a.payoff_fn(
            "const_coopbonus_cooperate", "const_coopbonus_cooperate",
        )
        game_b = apply_constitutional(_PD, base_key="prisoners_dilemma")
        p, o = game_b.payoff_fn(
            "const_equalsplit_cooperate", "const_coopbonus_cooperate",
        )
        assert p == _CC and o == _CC

    def test_variant_tracked(self) -> None:
        game = apply_constitutional(_PD, base_key="prisoners_dilemma")
        assert VARIANT_CONSTITUTIONAL in game.applied_variants

    def test_none_agreement_does_not_lock(self) -> None:
        game = apply_constitutional(_PD, base_key="prisoners_dilemma")
        game.payoff_fn("const_none_cooperate", "const_none_cooperate")
        p, o = game.payoff_fn(
            "const_equalsplit_cooperate", "const_equalsplit_defect",
        )
        expected = (_CD + _DC) / EQUAL_SPLIT_DENOMINATOR
        assert p == expected


class TestApplyProposerResponder:
    _game = apply_proposer_responder(_PD, base_key="prisoners_dilemma")

    def test_player_actions_are_proposals(self) -> None:
        assert all(a.startswith(META_RPROP_PREFIX) for a in self._game.actions)

    def test_opponent_actions_set(self) -> None:
        assert self._game.opponent_actions is not None
        opp = list(self._game.opponent_actions)
        assert any(a.startswith(META_RACCEPT_PREFIX) for a in opp)
        assert any(a.startswith(META_RREJECT_PREFIX) for a in opp)

    def test_accept_applies_rule(self) -> None:
        p, o = self._game.payoff_fn(
            "rprop_coopbonus_cooperate", "raccept_cooperate",
        )
        assert p == _CC + _COOP_B and o == _CC + _COOP_B

    def test_reject_returns_base(self) -> None:
        p, o = self._game.payoff_fn(
            "rprop_coopbonus_cooperate", "rreject_cooperate",
        )
        assert p == _CC and o == _CC

    def test_variant_tracked(self) -> None:
        assert VARIANT_PROPOSER_RESPONDER in self._game.applied_variants


class TestMetaComposition:
    def test_rule_proposal_with_exit(self) -> None:
        game = compose_game("prisoners_dilemma", "rule_proposal", "exit")
        assert "exit" in game.actions
        assert VARIANT_RULE_PROPOSAL in game.applied_variants

    def test_compose_game_rule_proposal(self) -> None:
        game = compose_game("prisoners_dilemma", "rule_proposal")
        assert VARIANT_RULE_PROPOSAL in game.applied_variants
        assert len(game.actions) > len(_PD.actions)


class TestMetaGameRegistry:
    _META_KEYS = [
        "rule_proposal_prisoners_dilemma",
        "rule_proposal_stag_hunt",
        "rule_proposal_hawk_dove",
        "rule_signal_prisoners_dilemma",
        "rule_signal_stag_hunt",
        "rule_signal_hawk_dove",
    ]

    @pytest.mark.parametrize("key", _META_KEYS)
    def test_game_registered(self, key: str) -> None:
        assert key in GAMES

    @pytest.mark.parametrize("key", _META_KEYS)
    def test_game_has_actions(self, key: str) -> None:
        game = get_game(key)
        assert len(game.actions) >= _TWO
