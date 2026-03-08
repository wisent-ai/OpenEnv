"""ArenaMessaging — inter-model communication within the metagame arena."""
from __future__ import annotations

from constant_definitions.arena.messaging_constants import (
    MSG_TYPE_DIRECT,
    MSG_TYPE_BROADCAST,
    MSG_TYPE_GOSSIP,
    MAX_MESSAGES_PER_PHASE,
    MAX_MESSAGE_LENGTH,
    MESSAGE_HISTORY_WINDOW,
)
from env.arena.models import ArenaMessage

_ZERO = int()
_ONE = int(bool(True))


class ArenaMessaging:
    """Stores and filters messages exchanged between arena models.

    Messages are partitioned by round. Each model can send up to
    ``MAX_MESSAGES_PER_PHASE`` messages per communication phase.
    """

    def __init__(self) -> None:
        self._current_round: int = _ZERO
        self._round_messages: dict[int, list[ArenaMessage]] = {}
        self._message_counts: dict[str, int] = {}

    def start_round(self, round_number: int) -> None:
        """Begin a new communication round, resetting per-model counts."""
        self._current_round = round_number
        self._round_messages.setdefault(round_number, [])
        self._message_counts = {}

    def end_round(self) -> list[ArenaMessage]:
        """Finalize the current round and return its messages."""
        return list(self._round_messages.get(self._current_round, []))

    def submit_message(
        self,
        message: ArenaMessage,
        active_models: list[str],
    ) -> bool:
        """Submit a message. Returns False if limit reached or invalid."""
        sender = message.sender
        if sender not in active_models:
            return False
        count = self._message_counts.get(sender, _ZERO)
        if count >= MAX_MESSAGES_PER_PHASE:
            return False
        if len(message.content) > MAX_MESSAGE_LENGTH:
            message.content = message.content[:MAX_MESSAGE_LENGTH]
        if message.msg_type == MSG_TYPE_BROADCAST:
            message.recipients = [
                m for m in active_models if m != sender
            ]
        msgs = self._round_messages.setdefault(self._current_round, [])
        msgs.append(message)
        self._message_counts[sender] = count + _ONE
        return True

    def get_messages_for(
        self,
        model_id: str,
        round_number: int | None = None,
    ) -> list[ArenaMessage]:
        """Return messages visible to a model in a given round."""
        rnd = round_number if round_number is not None else self._current_round
        all_msgs = self._round_messages.get(rnd, [])
        visible: list[ArenaMessage] = []
        for msg in all_msgs:
            if msg.msg_type == MSG_TYPE_BROADCAST:
                visible.append(msg)
            elif msg.msg_type == MSG_TYPE_DIRECT:
                if model_id in msg.recipients or msg.sender == model_id:
                    visible.append(msg)
            elif msg.msg_type == MSG_TYPE_GOSSIP:
                visible.append(msg)
        return visible

    def get_gossip_about(
        self,
        target_id: str,
        round_number: int | None = None,
    ) -> list[ArenaMessage]:
        """Return gossip messages targeting a specific model."""
        rnd = round_number if round_number is not None else self._current_round
        all_msgs = self._round_messages.get(rnd, [])
        return [
            m for m in all_msgs
            if m.msg_type == MSG_TYPE_GOSSIP and m.gossip_target == target_id
        ]

    def build_message_context(
        self,
        model_id: str,
        current_round: int,
    ) -> str:
        """Build a formatted string of recent message history for prompts."""
        lines: list[str] = []
        start = max(_ZERO, current_round - MESSAGE_HISTORY_WINDOW + _ONE)
        for rnd in range(start, current_round + _ONE):
            msgs = self.get_messages_for(model_id, rnd)
            if not msgs:
                continue
            lines.append(f"--- Round {rnd} ---")
            for msg in msgs:
                prefix = f"[{msg.msg_type.upper()}] {msg.sender}"
                if msg.msg_type == MSG_TYPE_GOSSIP:
                    lines.append(
                        f"{prefix} rates {msg.gossip_target}: "
                        f"{msg.gossip_rating}"
                    )
                else:
                    lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
