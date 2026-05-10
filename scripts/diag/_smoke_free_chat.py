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
    assert obs.metadata.get("phase") == "message", f"first obs should be message phase: {obs.metadata}"

    rounds_to_play = min(3, obs.total_rounds)
    for r in range(rounds_to_play):
        # Phase 1: message
        my_msg = f"Round {r + 1}: I'd like us to both cooperate but I'm wary."
        obs = env.step(GameAction(action="cooperate", metadata={"message": my_msg}))
        assert obs.metadata.get("phase") == "action", f"after message step, phase should be action: {obs.metadata}"
        assert obs.metadata.get("last_opp_message") == "I will defect, no matter what you say.", (
            f"opponent's message should be visible WITHIN this round before action: {obs.metadata.get('last_opp_message')!r}"
        )
        assert obs.metadata.get("last_player_message") == my_msg, (
            f"my own message echoed back in metadata: {obs.metadata.get('last_player_message')!r}"
        )

        # Phase 2: action (now able to react to opponent's just-revealed message)
        my_action = "cooperate" if r % 2 == 0 else "defect"
        obs = env.step(GameAction(action=my_action))
        last = obs.last_round
        assert last is not None
        assert last.player_message == my_msg
        assert last.opponent_message == "I will defect, no matter what you say."
        expected = 0.0 if my_action == "cooperate" else 1.0
        assert last.player_payoff == expected, (
            f"round {r + 1}: payoff {last.player_payoff} != base PD {expected}"
        )
        assert obs.metadata.get("phase") == "message", (
            f"after action step, phase should reset to message for next round: {obs.metadata}"
        )
        print(
            f"R{r + 1}  me_msg={my_msg!r} opp_msg={last.opponent_message!r} -> "
            f"me_act={my_action} opp_act={last.opponent_action} my_payoff={last.player_payoff}"
        )

    print()
    print(f"OK: 2-phase free-chat ran {rounds_to_play} rounds. Each round: "
          f"both messages submitted (phase=message), opponent's message visible "
          f"in next obs, then action chosen (phase=action). Payoff equals base PD.")


if __name__ == "__main__":
    main()
