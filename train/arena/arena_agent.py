"""ArenaPromptBuilder and ArenaAgent for metagame arena phases."""
from __future__ import annotations

import re
from typing import Callable, Optional

from train.agent import parse_action
from constant_definitions.arena.arena_constants import (
    PHASE_COMMUNICATION,
    PHASE_GOVERNANCE,
    PHASE_PLAY,
    PROPOSAL_BAN,
    PROPOSAL_ADD,
    PROPOSAL_RULE,
    PROPOSAL_NEW_GAME,
)
from constant_definitions.arena.messaging_constants import (
    MSG_TYPE_DIRECT,
    MSG_TYPE_BROADCAST,
    MSG_TYPE_GOSSIP,
)
from env.arena.models import ArenaMessage, ArenaProposal, ArenaVote

_ZERO = int()
_ONE = int(bool(True))

_RE_TO = re.compile(r"TO:\s*(\S+)\s+(.*)", re.IGNORECASE)
_RE_BROADCAST = re.compile(r"BROADCAST:\s*(.*)", re.IGNORECASE)
_RE_GOSSIP = re.compile(r"GOSSIP:\s*(\S+)\s+(\S+)", re.IGNORECASE)
_RE_PROPOSE = re.compile(
    r"PROPOSE\s+(BAN|ADD|RULE|NEW_GAME)\s*(.*)", re.IGNORECASE,
)
_RE_VOTE = re.compile(r"(APPROVE|REJECT)\s+(\d+)", re.IGNORECASE)


class ArenaPromptBuilder:
    """Builds phase-specific prompts for arena models."""

    @staticmethod
    def build_communication(
        model_id: str, active_models: list[str],
        message_context: str, round_number: int,
    ) -> str:
        """Build a prompt for the communication phase."""
        others = [m for m in active_models if m != model_id]
        return (
            f"[Arena State] Round {round_number}\n"
            f"[Your ID] {model_id}\n"
            f"[Active Models] {', '.join(active_models)}\n"
            f"[Messages]\n{message_context}\n\n"
            f"[Available Actions]\n"
            f"Send messages to other models. Formats:\n"
            f"  TO: <model_id> <message>\n"
            f"  BROADCAST: <message>\n"
            f"  GOSSIP: <target_model> trustworthy|untrustworthy|neutral\n"
            f"You may send multiple messages, one per line."
        )

    @staticmethod
    def build_governance(
        model_id: str, active_models: list[str],
        proposals_text: str, round_number: int,
    ) -> str:
        """Build a prompt for the governance phase."""
        return (
            f"[Arena State] Round {round_number}\n"
            f"[Your ID] {model_id}\n"
            f"[Active Models] {', '.join(active_models)}\n"
            f"[Proposals]\n{proposals_text}\n\n"
            f"[Available Actions]\n"
            f"Propose changes or vote:\n"
            f"  PROPOSE BAN <model_id>\n"
            f"  PROPOSE ADD <model_id>\n"
            f"  PROPOSE RULE <description>\n"
            f"  PROPOSE NEW_GAME <definition>\n"
            f"  APPROVE <proposal_index>\n"
            f"  REJECT <proposal_index>\n"
        )


class ArenaAgent:
    """Wraps a generate_fn to participate in all arena phases."""

    def __init__(self, model_id: str, generate_fn: Callable[[str], str]) -> None:
        self.model_id = model_id
        self._generate_fn = generate_fn

    def communicate(
        self, active_models: list[str],
        message_context: str, round_number: int,
    ) -> list[ArenaMessage]:
        """Generate communication messages."""
        prompt = ArenaPromptBuilder.build_communication(
            self.model_id, active_models, message_context, round_number,
        )
        raw = self._generate_fn(prompt)
        return self._parse_messages(raw)

    def govern(
        self, active_models: list[str],
        proposals_text: str, round_number: int,
    ) -> tuple[list[ArenaProposal], list[ArenaVote]]:
        """Generate governance proposals and votes."""
        prompt = ArenaPromptBuilder.build_governance(
            self.model_id, active_models, proposals_text, round_number,
        )
        raw = self._generate_fn(prompt)
        return self._parse_governance(raw)

    def _parse_messages(self, raw: str) -> list[ArenaMessage]:
        messages: list[ArenaMessage] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            m = _RE_TO.match(line)
            if m:
                messages.append(ArenaMessage(
                    sender=self.model_id, recipients=[m.group(_ONE)],
                    msg_type=MSG_TYPE_DIRECT, content=m.group(_ONE + _ONE),
                ))
                continue
            m = _RE_BROADCAST.match(line)
            if m:
                messages.append(ArenaMessage(
                    sender=self.model_id, msg_type=MSG_TYPE_BROADCAST,
                    content=m.group(_ONE),
                ))
                continue
            m = _RE_GOSSIP.match(line)
            if m:
                messages.append(ArenaMessage(
                    sender=self.model_id, msg_type=MSG_TYPE_GOSSIP,
                    gossip_target=m.group(_ONE),
                    gossip_rating=m.group(_ONE + _ONE),
                ))
        return messages

    def _parse_governance(
        self, raw: str,
    ) -> tuple[list[ArenaProposal], list[ArenaVote]]:
        proposals: list[ArenaProposal] = []
        votes: list[ArenaVote] = []
        _type_map = {
            "BAN": PROPOSAL_BAN, "ADD": PROPOSAL_ADD,
            "RULE": PROPOSAL_RULE, "NEW_GAME": PROPOSAL_NEW_GAME,
        }
        for line in raw.strip().split("\n"):
            line = line.strip()
            m = _RE_PROPOSE.match(line)
            if m:
                ptype = _type_map.get(m.group(_ONE).upper(), PROPOSAL_BAN)
                detail = m.group(_ONE + _ONE).strip()
                prop = ArenaProposal(proposer=self.model_id, proposal_type=ptype)
                if ptype in (PROPOSAL_BAN, PROPOSAL_ADD):
                    prop.target_model = detail
                elif ptype == PROPOSAL_RULE:
                    prop.rule_description = detail
                proposals.append(prop)
                continue
            m = _RE_VOTE.match(line)
            if m:
                approve = m.group(_ONE).upper() == "APPROVE"
                idx = int(m.group(_ONE + _ONE))
                votes.append(ArenaVote(
                    voter=self.model_id, proposal_index=idx, approve=approve,
                ))
        return proposals, votes
