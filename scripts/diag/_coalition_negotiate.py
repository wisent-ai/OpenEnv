"""LLM-driven coalition negotiation for the Llama runner.

Extracted from _episode_play.py to keep that file under the per-file
line cap. The negotiation flow this module supports:

* For player zero (the "agent" we're evaluating): play_episode_coalition
  in _episode_play.py calls _llm_negotiate per round. The LLM sees the
  pending proposals and returns a comma-separated list of indices to
  accept (or 'none'). The CoalitionAction returned has responses only;
  no proposals (the agent never offers its own coalitions yet).

* For opponent slots (in --mode self / cross): make_coalition_strategy
  returns a CoalitionStrategy whose negotiate() reuses _llm_negotiate
  and whose respond_to_proposal() asks the LLM 'accept' or 'reject'
  on a single inline proposal (the env calls this when the agent's
  CoalitionAction.proposals targets that opponent).
"""

from __future__ import annotations

from typing import Callable

from env.nplayer.coalition.models import CoalitionAction, CoalitionResponse


def _build_negotiate_prompt(obs):
    base = obs.base
    me = base.player_index
    lines = [
        f"[Game] {base.game_name}",
        f"[You are] P{me} of {base.num_players}",
        f"[Round] {base.current_round} of {base.total_rounds}",
        "[Scores] " + ", ".join(f"P{i}={s:g}" for i, s in enumerate(base.scores)),
    ]
    if obs.pending_proposals:
        lines.append("[Pending coalition proposals against you]")
        for i, p in enumerate(obs.pending_proposals):
            members = "{" + ", ".join(f"P{m}" for m in p.members) + "}"
            lines.append(
                f"  {i}: P{p.proposer} proposes {members} agreeing to play "
                f"'{p.agreed_action}', side payment {p.side_payment:g}"
            )
        lines.append(
            "[Instruction] Reply with comma-separated indices to ACCEPT, "
            "or 'none'. Example: 0,2"
        )
    else:
        lines.append("[No pending proposals] Reply 'none'.")
    return "\n".join(lines)


def _parse_acceptance_indices(completion, num_proposals):
    s = (completion or "").strip().lower()
    if "none" in s and not any(c.isdigit() for c in s):
        return []
    accepted, cur = [], ""
    for ch in s + ",":
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                idx = int(cur)
                if 0 <= idx < num_proposals:
                    accepted.append(idx)
                cur = ""
    return sorted(set(accepted))


def llm_negotiate(generate_fn: Callable[[str], str], obs) -> CoalitionAction:
    """One LLM call -> CoalitionAction (responses to pending proposals only)."""
    completion = generate_fn(_build_negotiate_prompt(obs))
    accept = set(_parse_acceptance_indices(completion, len(obs.pending_proposals)))
    me = obs.base.player_index
    responses = [
        CoalitionResponse(responder=me, proposal_index=i, accepted=(i in accept))
        for i in range(len(obs.pending_proposals))
    ]
    return CoalitionAction(responses=responses)


def make_coalition_strategy(generate_fn: Callable[[str], str], nplayer_agent_fn):
    """CoalitionStrategy that drives every method via the LLM.

    nplayer_agent_fn is the prebuilt NPlayerObservation -> NPlayerAction
    closure (built once in _episode_play.make_nplayer_agent). It is
    threaded in rather than rebuilt here to share the same parse-action
    counter wrapping.
    """
    class _LLMCoalitionStrategy:
        def negotiate(self, observation):
            return llm_negotiate(generate_fn, observation)

        def respond_to_proposal(self, observation, proposal):
            base = observation.base
            members = "{" + ", ".join(f"P{m}" for m in proposal.members) + "}"
            prompt = (
                f"[Game] {base.game_name}\n"
                f"[You are] P{base.player_index}\n"
                f"[Pending proposal] P{proposal.proposer} proposes coalition "
                f"{members} agreeing to play '{proposal.agreed_action}', "
                f"side payment {proposal.side_payment:g}\n"
                "[Instruction] Reply 'accept' or 'reject'."
            )
            reply = (generate_fn(prompt) or "").strip().lower()
            return "accept" in reply and not reply.startswith("reject")

        def choose_action(self, observation):
            return nplayer_agent_fn(observation.base).action

    return _LLMCoalitionStrategy()
