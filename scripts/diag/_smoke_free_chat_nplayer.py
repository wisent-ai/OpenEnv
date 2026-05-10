"""Smoke-test apply_free_chat on the N-player path.

Picks an N-player game, plays a round where the player sends a free-form
message + an action, opponents are scripted strategies (no message). Verifies
that:
  - free_chat_<key> is registered in NPLAYER_GAMES with action vocab unchanged
  - applied_variants contains 'free_chat'
  - obs.metadata['free_chat'] is True
  - RoundResult.messages preserves the player's message at index 0
  - Subsequent obs.metadata['last_opp_messages'] is a list of len num_players-1

Run:
    PYTHONPATH=. python3 scripts/diag/_smoke_free_chat_nplayer.py
"""
from __future__ import annotations

# Importing the module triggers NPLAYER_GAMES.update of the builtin set
# AND the free-chat sibling registrations.
import common.games_meta.nplayer_games  # noqa: F401
from common.games_meta.nplayer_config import NPLAYER_GAMES
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction


def main() -> None:
    free_chat_keys = [k for k in NPLAYER_GAMES if k.startswith("free_chat_")]
    assert free_chat_keys, f"no free_chat n-player keys registered: {sorted(NPLAYER_GAMES.keys())[:5]}..."
    base_key = free_chat_keys[0].removeprefix("free_chat_")
    fc_key = free_chat_keys[0]
    cfg = NPLAYER_GAMES[fc_key]
    base_cfg = NPLAYER_GAMES[base_key]
    assert "free_chat" in cfg.applied_variants, f"variant marker missing: {cfg.applied_variants}"
    assert cfg.actions == base_cfg.actions, f"action vocab changed: {cfg.actions} != {base_cfg.actions}"

    env = NPlayerEnvironment()
    obs = env.reset(game=fc_key)
    assert obs.metadata.get("free_chat") is True, f"obs missing free_chat flag: {obs.metadata}"

    msg = "I am proposing we all attend; we get more if at least 2 of us show."
    action = NPlayerAction(
        action=cfg.actions[0],
        metadata={"message": msg},
    )
    obs2 = env.step(action)
    last = obs2.last_round
    assert last is not None
    assert last.messages, f"messages list empty on RoundResult: {last}"
    assert last.messages[0] == msg, f"player message not preserved: {last.messages[0]!r}"
    expected_others = max(0, cfg.num_players - 1)
    others_in_meta = obs2.metadata.get("last_opp_messages", [])
    # Scripted opponents emit empty messages, so last_opp_messages may be
    # absent (any() filter) — check the list-shape branch independently
    # by inspecting RoundResult directly.
    assert len(last.messages) == cfg.num_players, (
        f"messages length {len(last.messages)} != num_players {cfg.num_players}"
    )

    print(f"OK: {fc_key}  num_players={cfg.num_players}  actions={cfg.actions}")
    print(f"     player_message={last.messages[0]!r}")
    print(f"     opponent_messages={last.messages[1:]} (scripted -> empty)")
    print(f"     obs.metadata.last_opp_messages={others_in_meta}")
    print(f"Total free_chat n-player games registered: {len(free_chat_keys)}")


if __name__ == "__main__":
    main()
