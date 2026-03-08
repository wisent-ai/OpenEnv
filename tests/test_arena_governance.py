"""Tests for ArenaGovernance: proposals, weighted voting, and action resolution."""
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

    _iface_stub.Environment = _EnvironmentStub
    _openenv_stub.core = _core_stub
    _core_stub.env_server = _server_stub
    _server_stub.interfaces = _iface_stub
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

sys.path.insert(int(), "/Users/lukaszbartoszcze/Documents/OpenEnv/kant")

import common.games  # noqa: F401 — breaks circular import before governance loads
import pytest
from env.arena.subsystems.governance import ArenaGovernance
from env.arena.models import ArenaProposal, ArenaVote
from constant_definitions.arena.arena_constants import (
    PROPOSAL_BAN, PROPOSAL_ADD, PROPOSAL_RULE, PROPOSAL_NEW_GAME,
    MAX_PROPOSALS_PER_ROUND,
)
from constant_definitions.arena.reputation_weights import (
    DEFAULT_ARENA_SCORE_NUMERATOR,
    DEFAULT_ARENA_SCORE_DENOMINATOR,
)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_DEFAULT_WEIGHT = DEFAULT_ARENA_SCORE_NUMERATOR / DEFAULT_ARENA_SCORE_DENOMINATOR
_HEAVY_WEIGHT = float(_TWO)
_LIGHT_WEIGHT = _DEFAULT_WEIGHT / float(_FOUR) / float(_TWO)

_MODELS = ["alpha", "beta", "gamma"]


def _gov() -> ArenaGovernance:
    return ArenaGovernance()


# ---------------------------------------------------------------------------
# submit_proposals
# ---------------------------------------------------------------------------

class TestSubmitProposals:
    def test_valid_ban_accepted(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta")
        accepted = gov.submit_proposals([prop], _MODELS)
        assert len(accepted) == _ONE
        assert len(gov.pending_proposals) == _ONE

    def test_valid_rule_accepted(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="beta", proposal_type=PROPOSAL_RULE,
                             rule_description="no collusion")
        assert len(gov.submit_proposals([prop], _MODELS)) == _ONE

    def test_valid_new_game_accepted(self) -> None:
        gov = _gov()
        game_def: dict = {"name": "test", "actions": [], "payoff_matrix": {}}
        prop = ArenaProposal(proposer="gamma", proposal_type=PROPOSAL_NEW_GAME,
                             game_definition=game_def)
        assert len(gov.submit_proposals([prop], _MODELS)) == _ONE

    def test_unknown_proposer_rejected(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="outsider", proposal_type=PROPOSAL_BAN,
                             target_model="alpha")
        assert len(gov.submit_proposals([prop], _MODELS)) == _ZERO

    def test_invalid_proposal_type_rejected(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type="invalid_type",
                             target_model="beta")
        assert len(gov.submit_proposals([prop], _MODELS)) == _ZERO

    def test_ban_without_target_rejected(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model=None)
        assert len(gov.submit_proposals([prop], _MODELS)) == _ZERO

    def test_rule_without_description_rejected(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_RULE,
                             rule_description=None)
        assert len(gov.submit_proposals([prop], _MODELS)) == _ZERO

    def test_new_game_without_definition_rejected(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_NEW_GAME,
                             game_definition=None)
        assert len(gov.submit_proposals([prop], _MODELS)) == _ZERO

    def test_max_proposals_limit_enforced(self) -> None:
        gov = _gov()
        proposals = [
            ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta"),
            ArenaProposal(proposer="beta", proposal_type=PROPOSAL_RULE,
                          rule_description="rule one"),
            ArenaProposal(proposer="gamma", proposal_type=PROPOSAL_ADD, target_model="delta"),
            ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_RULE,
                          rule_description="rule two"),
        ]
        accepted = gov.submit_proposals(proposals, _MODELS)
        assert len(accepted) == MAX_PROPOSALS_PER_ROUND
        assert len(gov.pending_proposals) == MAX_PROPOSALS_PER_ROUND

    def test_exactly_max_proposals_all_accepted(self) -> None:
        gov = _gov()
        proposals = [
            ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta"),
            ArenaProposal(proposer="beta", proposal_type=PROPOSAL_RULE,
                          rule_description="fairness"),
            ArenaProposal(proposer="gamma", proposal_type=PROPOSAL_ADD, target_model="delta"),
        ]
        assert len(gov.submit_proposals(proposals, _MODELS)) == _THREE


# ---------------------------------------------------------------------------
# tally_votes
# ---------------------------------------------------------------------------

class TestTallyVotes:
    def _load_ban(self, gov: ArenaGovernance) -> None:
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta")
        gov.submit_proposals([prop], _MODELS)

    def _load_rule(self, gov: ArenaGovernance) -> None:
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_RULE,
                             rule_description="test rule")
        gov.submit_proposals([prop], _MODELS)

    def _equal_votes(self, approve_alpha: bool, approve_beta: bool,
                     approve_gamma: bool) -> list[ArenaVote]:
        return [
            ArenaVote(voter="alpha", proposal_index=_ZERO,
                      approve=approve_alpha, weight=_DEFAULT_WEIGHT),
            ArenaVote(voter="beta", proposal_index=_ZERO,
                      approve=approve_beta, weight=_DEFAULT_WEIGHT),
            ArenaVote(voter="gamma", proposal_index=_ZERO,
                      approve=approve_gamma, weight=_DEFAULT_WEIGHT),
        ]

    def test_ban_passes_with_supermajority(self) -> None:
        # Two of three equal-weight voters approve — meets two-thirds threshold.
        gov = _gov()
        self._load_ban(gov)
        adopted, rejected = gov.tally_votes(
            self._equal_votes(True, True, False), _MODELS
        )
        assert _ZERO in adopted
        assert _ZERO not in rejected

    def test_ban_fails_below_supermajority(self) -> None:
        # One of three approves — below two-thirds ban threshold.
        gov = _gov()
        self._load_ban(gov)
        adopted, rejected = gov.tally_votes(
            self._equal_votes(True, False, False), _MODELS
        )
        assert _ZERO in rejected
        assert _ZERO not in adopted

    def test_rule_passes_with_simple_majority(self) -> None:
        # Two of three approve — exceeds one-half rule threshold.
        gov = _gov()
        self._load_rule(gov)
        adopted, _ = gov.tally_votes(self._equal_votes(True, True, False), _MODELS)
        assert _ZERO in adopted

    def test_rule_fails_with_no_approvals(self) -> None:
        gov = _gov()
        self._load_rule(gov)
        _, rejected = gov.tally_votes(self._equal_votes(False, False, False), _MODELS)
        assert _ZERO in rejected

    def test_votes_from_inactive_models_ignored(self) -> None:
        # Outsider vote does not count; only alpha's rejection matters.
        gov = _gov()
        self._load_rule(gov)
        votes = [
            ArenaVote(voter="outsider", proposal_index=_ZERO,
                      approve=True, weight=_DEFAULT_WEIGHT),
            ArenaVote(voter="alpha", proposal_index=_ZERO,
                      approve=False, weight=_DEFAULT_WEIGHT),
        ]
        _, rejected = gov.tally_votes(votes, _MODELS)
        assert _ZERO in rejected

    def test_pending_cleared_after_tally(self) -> None:
        gov = _gov()
        self._load_rule(gov)
        assert len(gov.pending_proposals) == _ONE
        gov.tally_votes([], _MODELS)
        assert len(gov.pending_proposals) == _ZERO

    def test_history_recorded_after_tally(self) -> None:
        gov = _gov()
        self._load_rule(gov)
        assert len(gov.history) == _ZERO
        gov.tally_votes([], _MODELS)
        assert len(gov.history) == _ONE

    def test_weighted_heavy_voter_passes_ban_alone(self) -> None:
        # Alpha carries heavy weight exceeding two-thirds of total weight alone.
        gov = _gov()
        self._load_ban(gov)
        votes = [
            ArenaVote(voter="alpha", proposal_index=_ZERO,
                      approve=True, weight=_HEAVY_WEIGHT),
            ArenaVote(voter="beta", proposal_index=_ZERO,
                      approve=False, weight=_LIGHT_WEIGHT),
            ArenaVote(voter="gamma", proposal_index=_ZERO,
                      approve=False, weight=_LIGHT_WEIGHT),
        ]
        adopted, _ = gov.tally_votes(votes, _MODELS)
        assert _ZERO in adopted

    def test_second_tally_appends_to_history(self) -> None:
        gov = _gov()
        self._load_rule(gov)
        gov.tally_votes([], _MODELS)
        self._load_rule(gov)
        gov.tally_votes([], _MODELS)
        assert len(gov.history) == _TWO


# ---------------------------------------------------------------------------
# apply_adopted
# ---------------------------------------------------------------------------

class TestApplyAdopted:
    def test_single_adopted_returns_action(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta")
        actions = gov.apply_adopted([_ZERO], [prop])
        assert len(actions) == _ONE
        assert actions[_ZERO]["type"] == PROPOSAL_BAN
        assert actions[_ZERO]["target_model"] == "beta"

    def test_out_of_range_index_skipped(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="beta")
        assert len(gov.apply_adopted([_ONE], [prop])) == _ZERO

    def test_empty_adopted_returns_empty(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_RULE,
                             rule_description="no defect")
        assert len(gov.apply_adopted([], [prop])) == _ZERO

    def test_rule_action_carries_description(self) -> None:
        gov = _gov()
        prop = ArenaProposal(proposer="beta", proposal_type=PROPOSAL_RULE,
                             rule_description="always cooperate")
        actions = gov.apply_adopted([_ZERO], [prop])
        assert actions[_ZERO]["rule_description"] == "always cooperate"

    def test_multiple_adopted_indices(self) -> None:
        gov = _gov()
        props = [
            ArenaProposal(proposer="alpha", proposal_type=PROPOSAL_BAN, target_model="gamma"),
            ArenaProposal(proposer="beta", proposal_type=PROPOSAL_RULE,
                          rule_description="rule two"),
        ]
        actions = gov.apply_adopted([_ZERO, _ONE], props)
        assert len(actions) == _TWO
        assert actions[_ZERO]["type"] == PROPOSAL_BAN
        assert actions[_ONE]["type"] == PROPOSAL_RULE

    def test_initial_state_is_clean(self) -> None:
        gov = _gov()
        assert gov.pending_proposals == []
        assert gov.history == []
