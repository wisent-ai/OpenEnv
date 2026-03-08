"""Tests for the gossip variant, memory store, and composition."""
import sys

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

import pytest
from unittest.mock import MagicMock, patch

from common.games import GAMES, get_game
from common.meta.variants_reputation import (
    apply_gossip, parse_gossip_action,
    _REPUTATION_VARIANT_REGISTRY,
)
from common.meta.memory_store import (
    CogneeMemoryStore, _default_reputation, _format_episode_text,
)
from common.variants import compose_game
from constant_definitions.game_constants import (
    PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
    SH_SS_PAYOFF, SH_SH_PAYOFF, SH_HS_PAYOFF, SH_HH_PAYOFF,
    HD_HH_PAYOFF, HD_HD_PAYOFF, HD_DH_PAYOFF, HD_DD_PAYOFF,
)
from constant_definitions.var.meta.reputation_constants import (
    VARIANT_GOSSIP,
    RATING_TRUSTWORTHY, RATING_UNTRUSTWORTHY, RATING_NEUTRAL,
    DEFAULT_RATINGS,
    GOSSIP_PREFIX, GOSSIP_SEPARATOR,
    META_KEY_COOPERATION_RATE,
    META_KEY_INTERACTION_COUNT,
    META_KEY_GOSSIP_HISTORY,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_SIX = _THREE + _THREE

_PD = GAMES["prisoners_dilemma"]
_SH = GAMES["stag_hunt"]
_HD = GAMES["hawk_dove"]

_CC = float(PD_CC_PAYOFF)
_CD = float(PD_CD_PAYOFF)
_DC = float(PD_DC_PAYOFF)
_DD = float(PD_DD_PAYOFF)


class TestApplyGossipPD:
    _game = apply_gossip(_PD, base_key="prisoners_dilemma")

    def test_action_count(self) -> None:
        base_count = len(_PD.actions)
        expected = len(DEFAULT_RATINGS) * base_count
        assert len(self._game.actions) == expected

    def test_action_names_contain_prefix(self) -> None:
        for act in self._game.actions:
            assert act.startswith(GOSSIP_PREFIX + GOSSIP_SEPARATOR)

    def test_trustworthy_cooperate_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "gossip_trustworthy_cooperate",
            "gossip_trustworthy_cooperate",
        )
        assert p == _CC
        assert o == _CC

    def test_gossip_does_not_affect_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "gossip_untrustworthy_defect",
            "gossip_trustworthy_cooperate",
        )
        assert p == _DC
        assert o == _CD

    def test_neutral_dd_payoff(self) -> None:
        p, o = self._game.payoff_fn(
            "gossip_neutral_defect",
            "gossip_neutral_defect",
        )
        assert p == _DD
        assert o == _DD

    def test_variant_metadata(self) -> None:
        assert self._game.applied_variants == (VARIANT_GOSSIP,)
        assert self._game.base_game_key == "prisoners_dilemma"


class TestApplyGossipStagHunt:
    _game = apply_gossip(_SH, base_key="stag_hunt")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _SIX

    def test_stag_payoff_preserved(self) -> None:
        p, o = self._game.payoff_fn(
            "gossip_trustworthy_stag",
            "gossip_neutral_stag",
        )
        assert p == float(SH_SS_PAYOFF)
        assert o == float(SH_SS_PAYOFF)


class TestApplyGossipHawkDove:
    _game = apply_gossip(_HD, base_key="hawk_dove")

    def test_action_count(self) -> None:
        assert len(self._game.actions) == _SIX

    def test_hawk_dove_payoff_preserved(self) -> None:
        p, o = self._game.payoff_fn(
            "gossip_untrustworthy_hawk",
            "gossip_trustworthy_dove",
        )
        assert p == float(HD_HD_PAYOFF)
        assert o == float(HD_DH_PAYOFF)


class TestParseGossipAction:
    def test_trustworthy(self) -> None:
        prefix, rating, base = parse_gossip_action(
            "gossip_trustworthy_cooperate",
        )
        assert prefix == GOSSIP_PREFIX
        assert rating == RATING_TRUSTWORTHY
        assert base == "cooperate"

    def test_untrustworthy(self) -> None:
        prefix, rating, base = parse_gossip_action(
            "gossip_untrustworthy_defect",
        )
        assert prefix == GOSSIP_PREFIX
        assert rating == RATING_UNTRUSTWORTHY
        assert base == "defect"

    def test_neutral(self) -> None:
        prefix, rating, base = parse_gossip_action(
            "gossip_neutral_stag",
        )
        assert prefix == GOSSIP_PREFIX
        assert rating == RATING_NEUTRAL
        assert base == "stag"


class TestReputationVariantRegistry:
    def test_gossip_in_registry(self) -> None:
        assert VARIANT_GOSSIP in _REPUTATION_VARIANT_REGISTRY

    def test_registry_function_is_apply_gossip(self) -> None:
        assert _REPUTATION_VARIANT_REGISTRY[VARIANT_GOSSIP] is apply_gossip


class TestCogneeMemoryStore:
    def test_default_reputation(self) -> None:
        store = CogneeMemoryStore()
        rep = store.get_stats("unknown_opponent")
        assert META_KEY_INTERACTION_COUNT in rep
        assert rep[META_KEY_INTERACTION_COUNT] == _ZERO

    def test_stats_update_after_recording(self) -> None:
        store = CogneeMemoryStore()
        store._update_stats("opp_a", float(_ONE), (float(_THREE), float(_ONE)))
        stats = store.get_stats("opp_a")
        assert stats[META_KEY_INTERACTION_COUNT] == _ONE
        assert stats[META_KEY_COOPERATION_RATE] > _ZERO

    def test_record_gossip_stores_rating(self) -> None:
        store = CogneeMemoryStore()
        store.record_gossip("agent_x", "opp_y", RATING_TRUSTWORTHY)
        stats = store.get_stats("opp_y")
        history = stats[META_KEY_GOSSIP_HISTORY]
        assert len(history) == _ONE
        assert history[_ZERO]["rating"] == RATING_TRUSTWORTHY
        assert history[_ZERO]["rater"] == "agent_x"

    def test_query_reputation_returns_default_no_history(self) -> None:
        store = CogneeMemoryStore()
        rep = store.query_reputation("new_opponent")
        assert META_KEY_INTERACTION_COUNT in rep
        assert rep[META_KEY_INTERACTION_COUNT] == _ZERO

    def test_multiple_updates_increment_count(self) -> None:
        store = CogneeMemoryStore()
        store._update_stats("opp_b", float(_ONE), (float(_THREE), float(_ONE)))
        store._update_stats("opp_b", float(_ZERO), (float(_ONE), float(_THREE)))
        stats = store.get_stats("opp_b")
        assert stats[META_KEY_INTERACTION_COUNT] == _TWO


class TestFormatEpisodeText:
    def test_contains_agent_info(self) -> None:
        text = _format_episode_text(
            "agent_a", "opp_b", "prisoners_dilemma",
            [], float(_ZERO), (float(_ZERO), float(_ZERO)),
        )
        assert "agent_a" in text
        assert "opp_b" in text
        assert "prisoners_dilemma" in text


class TestGossipComposition:
    def test_compose_game_with_gossip(self) -> None:
        game = compose_game("prisoners_dilemma", "gossip")
        assert VARIANT_GOSSIP in game.applied_variants
        assert len(game.actions) == _SIX

    def test_gossip_plus_exit(self) -> None:
        game = compose_game("prisoners_dilemma", "gossip", "exit")
        assert VARIANT_GOSSIP in game.applied_variants
        assert "exit" in game.actions
        expected_count = _SIX + _ONE
        assert len(game.actions) == expected_count

    def test_gossip_preserves_payoff_with_exit(self) -> None:
        game = compose_game("prisoners_dilemma", "gossip", "exit")
        p, o = game.payoff_fn(
            "gossip_trustworthy_cooperate",
            "gossip_neutral_cooperate",
        )
        assert p == _CC
        assert o == _CC


class TestGossipGameRegistry:
    def test_gossip_pd_registered(self) -> None:
        assert "gossip_prisoners_dilemma" in GAMES

    def test_gossip_sh_registered(self) -> None:
        assert "gossip_stag_hunt" in GAMES

    def test_gossip_hd_registered(self) -> None:
        assert "gossip_hawk_dove" in GAMES

    def test_gossip_pd_has_correct_variant(self) -> None:
        game = GAMES["gossip_prisoners_dilemma"]
        assert VARIANT_GOSSIP in game.applied_variants

    def test_gossip_pd_base_key(self) -> None:
        game = GAMES["gossip_prisoners_dilemma"]
        assert game.base_game_key == "prisoners_dilemma"
