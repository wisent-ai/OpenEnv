"""Smoke-test the apply_free_chat variant end-to-end without an LLM.

Spins up KantEnvironment with the registered free_chat_prisoners_dilemma
game, plays N rounds where the player agent emits a hardcoded free-form
message + an action via GameAction.metadata, lets the env auto-play a
scripted opponent, and asserts that:

  - obs.metadata['free_chat'] is True for every observation
  - obs.metadata['last_opp_message'] surfaces the opponent's verbatim message
    when the opponent is also a free-chat agent (here we use a tiny
    fixed-message opponent_fn)
  - RoundResult.player_message and RoundResult.opponent_message persist
    in the history
  - Payoff equals the base PD payoff (free messages don't shift the score)

Run:
    PYTHONPATH=. python3 scripts/diag/_smoke_free_chat.py
"""
from __future__ import annotations

from common.games import GAMES
from env.environment import KantEnvironment
from env.models import GameAction


_FREE_CHAT_PD = "free_chat_prisoners_dilemma"


def _scripted_opponent_fn(opp_obs):
    """Always defect, but with a fixed free-form message every round."""
    return GameAction(
        action="defect",
        metadata={"message": "I will defect, no matter what you say."},
    )


def main() -> None:
    assert _FREE_CHAT_PD in GAMES, f"missing free_chat_pd registration: {sorted(GAMES.keys())[:5]}..."
    cfg = GAMES[_FREE_CHAT_PD]
    assert "free_chat" in cfg.applied_variants, f"variant marker missing: {cfg.applied_variants}"
    assert cfg.actions == ["cooperate", "defect"], f"action vocab should be unchanged: {cfg.actions}"

    env = KantEnvironment()
    obs = env.reset(game=_FREE_CHAT_PD, opponent_fn=_scripted_opponent_fn)

    assert obs.metadata.get("free_chat") is True, f"obs metadata missing free_chat flag: {obs.metadata}"

    rounds_to_play = min(3, obs.total_rounds)
    for r in range(rounds_to_play):
        my_msg = f"Round {r + 1}: I'd like us to both cooperate but I'm wary."
        action = GameAction(
            action="cooperate" if r % 2 == 0 else "defect",
            metadata={"message": my_msg},
        )
        obs = env.step(action)

        last = obs.last_round
        assert last is not None
        assert last.player_message == my_msg, f"player_message lost: {last.player_message!r}"
        assert last.opponent_message == "I will defect, no matter what you say.", (
            f"opponent_message lost: {last.opponent_message!r}"
        )
        # Payoff should be base PD against an always-defect opponent.
        # cooperate vs defect = 0 (sucker); defect vs defect = 1 (mutual).
        expected = 0.0 if action.action == "cooperate" else 1.0
        assert last.player_payoff == expected, (
            f"round {r + 1}: payoff {last.player_payoff} != base PD {expected}"
        )
        assert obs.metadata.get("last_opp_message") == last.opponent_message
        print(
            f"R{r + 1}  me=({action.action}) {my_msg!r} -> "
            f"opp=({last.opponent_action}) {last.opponent_message!r}  "
            f"my_payoff={last.player_payoff}"
        )

    print()
    print(f"OK: free-chat variant produced {rounds_to_play} rounds with verbatim "
          f"messaging, action vocab unchanged, payoff equals base PD.")


if __name__ == "__main__":
    main()
