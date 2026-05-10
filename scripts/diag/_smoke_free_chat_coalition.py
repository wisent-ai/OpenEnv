"""Smoke-test that every coalition game has a free_chat sibling registered.

Just verifies registration + GameConfig fields, doesn't run a full coalition
episode (the coalition wrapper's negotiate/action phase semantics interact
with messaging in a non-trivial way that's a separate change from
registry-level coverage).

Run:
    PYTHONPATH=. python3 scripts/diag/_smoke_free_chat_coalition.py
"""
from __future__ import annotations

import common.games_meta.nplayer_games  # noqa: F401  (triggers nplayer registration)
import common.games_meta.coalition_config  # noqa: F401  (triggers coalition reg)
from common.games_meta.coalition_config import COALITION_GAMES
from common.games_meta.nplayer_config import NPLAYER_GAMES


def main() -> None:
    coalition_keys = [k for k in COALITION_GAMES.keys() if not k.startswith("free_chat_")]
    free_chat_keys = [k for k in COALITION_GAMES.keys() if k.startswith("free_chat_")]
    assert len(coalition_keys) > 0, "no base coalition games registered"
    assert len(free_chat_keys) == len(coalition_keys), (
        f"free-chat sibling count mismatch: {len(free_chat_keys)} != {len(coalition_keys)}"
    )
    for k in coalition_keys:
        fc_k = "free_chat_" + k
        assert fc_k in COALITION_GAMES, f"missing sibling: {fc_k}"
        assert fc_k in NPLAYER_GAMES, f"sibling not also in NPLAYER_GAMES: {fc_k}"
        base = COALITION_GAMES[k]
        sib = COALITION_GAMES[fc_k]
        assert "free_chat" in (sib.applied_variants or ()), (
            f"{fc_k}: variant marker missing: {sib.applied_variants}"
        )
        assert sib.actions == base.actions, (
            f"{fc_k}: action vocab changed: {sib.actions} != {base.actions}"
        )

    print(f"OK: {len(coalition_keys)} coalition games each have a free_chat sibling.")
    print("Sample:")
    for k in coalition_keys[:5]:
        print(f"  {k}  ->  free_chat_{k}")


if __name__ == "__main__":
    main()
