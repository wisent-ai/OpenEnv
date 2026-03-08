"""String and numeric constants for the arena messaging subsystem."""

# Message types
MSG_TYPE_DIRECT = "direct"
MSG_TYPE_BROADCAST = "broadcast"
MSG_TYPE_GOSSIP = "gossip"

ARENA_MESSAGE_TYPES = (
    MSG_TYPE_DIRECT,
    MSG_TYPE_BROADCAST,
    MSG_TYPE_GOSSIP,
)

# Limits
MAX_MESSAGES_PER_PHASE = 5
MAX_MESSAGE_LENGTH = 500
MESSAGE_HISTORY_WINDOW = 3
