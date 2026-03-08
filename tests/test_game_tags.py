"""Tests for the game tag / category system."""

from __future__ import annotations

import pytest

from constant_definitions.batch4.tag_constants import (
    CATEGORIES,
    NO_COMMUNICATION,
    CHEAP_TALK,
    COSTLY_SIGNALING,
    BINDING_COMMITMENT,
    MEDIATED,
    COMPLETE_INFORMATION,
    ZERO_SUM,
    SOCIAL_DILEMMA,
    SIMULTANEOUS,
    SEQUENTIAL,
    COORDINATION,
    AUCTION,
    VOTING,
    SECURITY,
    EVOLUTIONARY,
    BINARY_CHOICE,
    MULTIPLAYER,
    COALITION_FORMATION,
    PENALTY_ENFORCEMENT,
    BINDING_ENFORCEMENT,
    META_GOVERNANCE,
)
from common.games import GAMES
from common.games_meta.nplayer_config import NPLAYER_GAMES
import common.games_meta.nplayer_games  # noqa: F401 – trigger registration
import common.games_meta.coalition_config  # noqa: F401 – trigger dual-registration
from common.games_meta.game_tags import (
    GAME_TAGS,
    get_games_by_tag,
    get_games_by_tags,
    list_tags,
    list_categories,
)

# ---------------------------------------------------------------------------
# Named constants for test thresholds
# ---------------------------------------------------------------------------
_MIN_TAGS_PER_GAME = int(bool(True)) + int(bool(True)) + int(bool(True))  # each game needs >= this many tags
_NPLAYER_ONLY = set(NPLAYER_GAMES) - set(GAMES)
_ALL_GAME_KEYS = set(GAMES) | _NPLAYER_ONLY
_EXPECTED_TOTAL_GAMES = len(_ALL_GAME_KEYS)
_ONE = int(bool(True))
_ZERO = int()

# ── Communication tag set (every game must have exactly one) ──
_COMM_TAGS = frozenset({NO_COMMUNICATION, CHEAP_TALK, COSTLY_SIGNALING, BINDING_COMMITMENT, MEDIATED})


class TestGameTagCoverage:
    """Every registered game must appear in GAME_TAGS."""

    def test_all_games_have_tags(self):
        missing = _ALL_GAME_KEYS - set(GAME_TAGS)
        assert not missing, f"Games missing from GAME_TAGS: {missing}"

    def test_no_extra_games_in_tags(self):
        extra = set(GAME_TAGS) - _ALL_GAME_KEYS
        assert not extra, f"GAME_TAGS has keys not in any registry: {extra}"

    def test_tag_count_equals_game_count(self):
        assert len(GAME_TAGS) == _EXPECTED_TOTAL_GAMES


class TestTagValidity:
    """Tags must be well-formed and drawn from the taxonomy."""

    def test_every_game_has_communication_tag(self):
        for key, tags in GAME_TAGS.items():
            has_comm = tags & _COMM_TAGS
            assert has_comm, f"{key} has no communication tag"

    def test_every_game_has_minimum_tags(self):
        for key, tags in GAME_TAGS.items():
            assert len(tags) >= _MIN_TAGS_PER_GAME, (
                f"{key} only has {len(tags)} tags (need >= {_MIN_TAGS_PER_GAME})"
            )

    def test_all_tags_belong_to_taxonomy(self):
        valid_tags: set[str] = set()
        for tag_list in CATEGORIES.values():
            valid_tags.update(tag_list)
        for key, tags in GAME_TAGS.items():
            invalid = tags - valid_tags
            assert not invalid, f"{key} has unknown tags: {invalid}"


class TestKnownMappings:
    """Spot-check specific games for expected tags."""

    def test_prisoners_dilemma_no_communication(self):
        assert NO_COMMUNICATION in GAME_TAGS["prisoners_dilemma"]

    def test_prisoners_dilemma_social_dilemma(self):
        assert SOCIAL_DILEMMA in GAME_TAGS["prisoners_dilemma"]

    def test_cheap_talk_pd_has_cheap_talk(self):
        assert CHEAP_TALK in GAME_TAGS["cheap_talk_pd"]

    def test_cheap_talk_game_has_cheap_talk(self):
        assert CHEAP_TALK in GAME_TAGS["cheap_talk"]

    def test_mediated_game_has_mediated(self):
        assert MEDIATED in GAME_TAGS["mediated_game"]

    def test_binding_commitment_has_binding(self):
        assert BINDING_COMMITMENT in GAME_TAGS["binding_commitment"]

    def test_beer_quiche_costly_signaling(self):
        assert COSTLY_SIGNALING in GAME_TAGS["beer_quiche"]

    def test_matching_pennies_zero_sum(self):
        assert ZERO_SUM in GAME_TAGS["matching_pennies"]

    def test_rock_paper_scissors_zero_sum(self):
        assert ZERO_SUM in GAME_TAGS["rock_paper_scissors"]

    def test_rpsls_zero_sum(self):
        assert ZERO_SUM in GAME_TAGS["rpsls"]

    def test_penalty_shootout_zero_sum(self):
        assert ZERO_SUM in GAME_TAGS["penalty_shootout"]

    def test_stag_hunt_coordination(self):
        assert COORDINATION in GAME_TAGS["stag_hunt"]

    def test_first_price_auction_tag(self):
        assert AUCTION in GAME_TAGS["first_price_auction"]

    def test_jury_voting_tag(self):
        assert VOTING in GAME_TAGS["jury_voting"]

    def test_evolutionary_pd_evolutionary(self):
        assert EVOLUTIONARY in GAME_TAGS["evolutionary_pd"]

    def test_security_game_security(self):
        assert SECURITY in GAME_TAGS["security_game"]


class TestFilterFunctions:
    """get_games_by_tag and get_games_by_tags return correct results."""

    def test_get_games_by_tag_cheap_talk(self):
        results = get_games_by_tag(CHEAP_TALK)
        assert "cheap_talk_pd" in results
        assert "cheap_talk" in results
        assert "prisoners_dilemma" not in results

    def test_get_games_by_tag_zero_sum(self):
        results = get_games_by_tag(ZERO_SUM)
        assert "matching_pennies" in results
        assert "rock_paper_scissors" in results
        assert "rpsls" in results
        assert "prisoners_dilemma" not in results

    def test_get_games_by_tags_intersection(self):
        results = get_games_by_tags(NO_COMMUNICATION, SOCIAL_DILEMMA)
        assert "prisoners_dilemma" in results
        assert "stag_hunt" in results
        # cheap_talk_pd has CHEAP_TALK, not NO_COMMUNICATION
        assert "cheap_talk_pd" not in results

    def test_get_games_by_tags_narrow(self):
        results = get_games_by_tags(ZERO_SUM, BINARY_CHOICE)
        assert "matching_pennies" in results
        assert "inspection_game" in results
        # rock_paper_scissors is SMALL_CHOICE
        assert "rock_paper_scissors" not in results

    def test_get_games_by_tag_returns_list(self):
        result = get_games_by_tag(SOCIAL_DILEMMA)
        assert isinstance(result, list)

    def test_get_games_by_tags_empty_on_impossible(self):
        # No game is both zero_sum and coordination
        results = get_games_by_tags(ZERO_SUM, COORDINATION)
        assert results == []


class TestListFunctions:
    """list_tags and list_categories produce valid output."""

    def test_list_tags_returns_sorted(self):
        tags = list_tags()
        assert tags == sorted(tags)

    def test_list_tags_nonempty(self):
        assert len(list_tags()) > _ZERO

    def test_list_categories_has_all_dimensions(self):
        cats = list_categories()
        for dim in ("communication", "information", "structure",
                     "payoff_type", "domain", "action_space",
                     "player_count", "coalition", "enforcement",
                     "governance"):
            assert dim in cats, f"Missing dimension: {dim}"

    def test_no_empty_categories(self):
        cats = list_categories()
        for dim, tags in cats.items():
            assert len(tags) >= _ONE, f"Dimension {dim} is empty"

    def test_every_category_tag_used_by_at_least_one_game(self):
        all_used: set[str] = set()
        for tags in GAME_TAGS.values():
            all_used |= tags
        cats = list_categories()
        for dim, tag_list in cats.items():
            for tag in tag_list:
                assert tag in all_used, (
                    f"Tag {tag!r} from {dim} not used by any game"
                )


class TestNPlayerTags:
    """Spot-checks for N-player and coalition game tags."""

    def test_nplayer_public_goods_multiplayer(self):
        assert MULTIPLAYER in GAME_TAGS["nplayer_public_goods"]

    def test_nplayer_el_farol_multiplayer(self):
        assert MULTIPLAYER in GAME_TAGS["nplayer_el_farol"]

    def test_coalition_cartel_tags(self):
        tags = GAME_TAGS["coalition_cartel"]
        assert MULTIPLAYER in tags
        assert COALITION_FORMATION in tags
        assert PENALTY_ENFORCEMENT in tags
        assert META_GOVERNANCE in tags

    def test_coalition_voting_binding(self):
        tags = GAME_TAGS["coalition_voting"]
        assert BINDING_ENFORCEMENT in tags
        assert COALITION_FORMATION in tags

    def test_multiplayer_tag_returns_all_ten(self):
        results = get_games_by_tag(MULTIPLAYER)
        assert len(results) == len(_NPLAYER_ONLY)

    def test_coalition_penalty_filter(self):
        results = get_games_by_tags(COALITION_FORMATION, PENALTY_ENFORCEMENT)
        expected = {"coalition_cartel", "coalition_ostracism", "coalition_commons"}
        assert set(results) == expected
