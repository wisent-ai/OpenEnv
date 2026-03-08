"""LLM agent for coalition formation and meta-governance environments."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionObservation,
    CoalitionProposal, CoalitionResponse,
)
from env.nplayer.governance.models import GovernanceProposal, GovernanceVote
from env.nplayer.models import NPlayerAction
from train.agent import parse_action
from constant_definitions.train.agent_constants import (
    COALITION_PROMPT_SECTION_COALITIONS,
    COALITION_PROMPT_SECTION_PHASE,
    COALITION_PROMPT_SECTION_PROPOSALS,
    COALITION_SYSTEM_PROMPT,
    GOVERNANCE_PROMPT_SECTION_PENDING,
    GOVERNANCE_PROMPT_SECTION_RULES,
    MAX_PROMPT_HISTORY_ROUNDS,
    NPLAYER_PROMPT_SECTION_ALL_SCORES,
    PROMPT_SECTION_ACTIONS, PROMPT_SECTION_GAME,
    PROMPT_SECTION_HISTORY, PROMPT_SECTION_INSTRUCTION,
)

_ZERO = int()
_ONE = int(bool(True))
_NL = "\n"
_SEP = "\n\n"
_BO = "["
_BC = "]"
_CS = ": "
_DS = "- "
_PP = "Player "
_RP = "Round "
_PS = " | "
_PL = " played: "
_PY = " payoff: "


class CoalitionPromptBuilder:
    """Formats CoalitionObservation into structured text prompts."""

    @staticmethod
    def build_negotiate(obs: CoalitionObservation) -> str:
        """Build a negotiate-phase prompt."""
        sections: List[str] = []
        base = obs.base
        sections.append(
            _BO + PROMPT_SECTION_GAME + _BC + _NL
            + base.game_name + _NL + base.game_description
        )
        sections.append(
            _BO + COALITION_PROMPT_SECTION_PHASE + _BC + _NL
            + obs.phase + _NL + "Enforcement" + _CS + obs.enforcement
        )
        if obs.pending_proposals:
            prop_lines = [
                str(idx) + _CS + "proposer=" + str(p.proposer)
                + " members=" + str(p.members)
                + " action=" + p.agreed_action
                for idx, p in enumerate(obs.pending_proposals)
            ]
            sections.append(
                _BO + COALITION_PROMPT_SECTION_PROPOSALS + _BC
                + _NL + _NL.join(prop_lines)
            )
        if obs.active_coalitions:
            coal_lines = [
                "members=" + str(c.members) + " action=" + c.agreed_action
                for c in obs.active_coalitions
            ]
            sections.append(
                _BO + COALITION_PROMPT_SECTION_COALITIONS + _BC
                + _NL + _NL.join(coal_lines)
            )
        if obs.current_rules is not None:
            rules = obs.current_rules
            active_mechs = [k for k, v in rules.mechanics.items() if v]
            sections.append(
                _BO + GOVERNANCE_PROMPT_SECTION_RULES + _BC + _NL
                + "enforcement" + _CS + rules.enforcement + _NL
                + "active_mechanics" + _CS + str(active_mechs)
            )
        if obs.pending_governance:
            gov_lines = [
                str(i) + _CS + gp.proposal_type + " by " + _PP + str(gp.proposer)
                for i, gp in enumerate(obs.pending_governance)
            ]
            sections.append(
                _BO + GOVERNANCE_PROMPT_SECTION_PENDING + _BC
                + _NL + _NL.join(gov_lines)
            )
        score_lines = [
            _PP + str(i) + _CS + str(s)
            for i, s in enumerate(obs.adjusted_scores)
        ]
        sections.append(
            _BO + NPLAYER_PROMPT_SECTION_ALL_SCORES + _BC
            + _NL + _NL.join(score_lines)
        )
        action_lines = [_DS + a for a in base.available_actions]
        sections.append(
            _BO + PROMPT_SECTION_ACTIONS + _BC + _NL + _NL.join(action_lines)
        )
        sections.append(
            _BO + PROMPT_SECTION_INSTRUCTION + _BC + _NL + COALITION_SYSTEM_PROMPT
        )
        return _SEP.join(sections)

    @staticmethod
    def build_action(obs: CoalitionObservation) -> str:
        """Build an action-phase prompt."""
        sections: List[str] = []
        base = obs.base
        sections.append(
            _BO + PROMPT_SECTION_GAME + _BC + _NL
            + base.game_name + _NL + base.game_description
        )
        sections.append(
            _BO + COALITION_PROMPT_SECTION_PHASE + _BC + _NL + obs.phase
        )
        my_coals = [
            "members=" + str(c.members) + " agreed_action=" + c.agreed_action
            for c in obs.active_coalitions
            if base.player_index in c.members
        ]
        if my_coals:
            sections.append(
                _BO + COALITION_PROMPT_SECTION_COALITIONS + _BC
                + _NL + _NL.join(my_coals)
            )
        if base.history:
            h_lines: List[str] = []
            for rnd in base.history[-MAX_PROMPT_HISTORY_ROUNDS:]:
                parts = [_RP + str(rnd.round_number)]
                for pidx, (act, pay) in enumerate(zip(rnd.actions, rnd.payoffs)):
                    parts.append(
                        _PP + str(pidx) + _PL + act + _PY + str(pay)
                    )
                h_lines.append(_PS.join(parts))
            sections.append(
                _BO + PROMPT_SECTION_HISTORY + _BC + _NL + _NL.join(h_lines)
            )
        action_lines = [_DS + a for a in base.available_actions]
        sections.append(
            _BO + PROMPT_SECTION_ACTIONS + _BC + _NL + _NL.join(action_lines)
        )
        sections.append(
            _BO + PROMPT_SECTION_INSTRUCTION + _BC + _NL
            + "Choose your action. Respond with ONLY the action name."
        )
        return _SEP.join(sections)


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from LLM output, return None on failure."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= _ZERO and end > start:
        try:
            return json.loads(stripped[start:end + _ONE])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


class CoalitionLLMAgent:
    """LLM-based agent for coalition environments.

    Implements the negotiate + act protocol expected by
    CoalitionTournamentRunner.
    """

    def __init__(
        self, generate_fn: Callable[[str], str],
        player_index: int = _ZERO,
        prompt_builder: Optional[CoalitionPromptBuilder] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._player_index = player_index
        self._prompt_builder = prompt_builder or CoalitionPromptBuilder()

    def negotiate(self, obs: CoalitionObservation) -> CoalitionAction:
        """Generate coalition proposals and responses to pending ones."""
        prompt = self._prompt_builder.build_negotiate(obs)
        completion = self._generate_fn(prompt)
        parsed = _safe_json_parse(completion)
        if parsed is not None:
            proposals = self._extract_proposals(parsed, obs)
            responses = self._extract_responses(parsed, obs)
        else:
            proposals = []
            responses = self._default_responses(obs)
        return CoalitionAction(proposals=proposals, responses=responses)

    def act(self, obs: CoalitionObservation) -> NPlayerAction:
        """Select a game action during the action phase."""
        prompt = self._prompt_builder.build_action(obs)
        completion = self._generate_fn(prompt)
        action_str = parse_action(completion, obs.base.available_actions)
        return NPlayerAction(action=action_str)

    def _extract_proposals(
        self, data: Dict[str, Any], obs: CoalitionObservation,
    ) -> List[CoalitionProposal]:
        raw = data.get("proposals", [])
        if not isinstance(raw, list):
            return []
        result: List[CoalitionProposal] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            members = item.get("members", [])
            action = item.get("agreed_action", "")
            if isinstance(members, list) and action in obs.base.available_actions:
                result.append(CoalitionProposal(
                    proposer=self._player_index,
                    members=members, agreed_action=action,
                ))
        return result

    def _extract_responses(
        self, data: Dict[str, Any], obs: CoalitionObservation,
    ) -> List[CoalitionResponse]:
        raw = data.get("responses", {})
        if not isinstance(raw, dict):
            return self._default_responses(obs)
        result: List[CoalitionResponse] = []
        for idx in range(len(obs.pending_proposals)):
            accepted = raw.get(str(idx), True)
            result.append(CoalitionResponse(
                responder=self._player_index,
                proposal_index=idx, accepted=bool(accepted),
            ))
        return result

    def _default_responses(
        self, obs: CoalitionObservation,
    ) -> List[CoalitionResponse]:
        return [
            CoalitionResponse(
                responder=self._player_index,
                proposal_index=idx, accepted=True,
            )
            for idx in range(len(obs.pending_proposals))
        ]
