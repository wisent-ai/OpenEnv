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

from typing import Callable, Optional

from constant_definitions.nplayer.coalition_constants import (
    COALITION_DEFAULT_SIDE_PAYMENT,
)
from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionProposal, CoalitionResponse,
)

# Cap side-payments parsed out of LLM output to a small bounded range so
# the model can't propose nonsensical transfers if it hallucinates a
# huge number. Lower bound: zero (no negative payments to the partner).
_SIDE_PAYMENT_MAX_NUMERATOR = 5
_SIDE_PAYMENT_MAX_DENOMINATOR = 1
_SIDE_PAYMENT_MAX = float(_SIDE_PAYMENT_MAX_NUMERATOR / _SIDE_PAYMENT_MAX_DENOMINATOR)


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


def _build_propose_prompt(obs):
    base = obs.base
    me = base.player_index
    others = ", ".join(f"P{i}" for i in range(base.num_players) if i != me)
    actions = ", ".join(base.available_actions)
    return (
        f"[Game] {base.game_name}\n"
        f"[You are] P{me} of {base.num_players}\n"
        f"[Round] {base.current_round} of {base.total_rounds}\n"
        f"[Other players] {others}\n"
        f"[Available actions] {actions}\n"
        f"[Side-payment range] 0 to {_SIDE_PAYMENT_MAX:g}\n"
        "[Instruction] Optionally propose a coalition. Reply "
        "'P<i> [P<j> ...] <action> pay <amount>' to invite one or more "
        "other players to agree on that action with that side payment, "
        "or 'none' to skip. Examples: 'P2 cooperate pay 1', "
        "'P2 P3 cooperate pay 1'."
    )


def _scan_first_number(text: str) -> Optional[str]:
    """Return the first contiguous decimal number in *text*, or None."""
    cur, started = "", False
    for ch in text:
        if ch.isdigit() or (ch == "." and started and "." not in cur):
            cur += ch
            started = True
        elif started:
            break
    return cur or None


def _parse_proposal(completion, obs):
    s = (completion or "").strip().lower()
    if "none" in s and not any(c.isdigit() for c in s):
        return None
    me = obs.base.player_index

    # Earliest available-action token in the completion -- everything
    # before it is read as player indices, everything after as the
    # optional side payment.
    chosen_pos, chosen = -1, None
    for a in obs.base.available_actions:
        idx = s.find(a.lower())
        if idx >= 0 and (chosen_pos == -1 or idx < chosen_pos):
            chosen_pos, chosen = idx, a
    if chosen is None:
        return None

    # All distinct, in-range, non-self integer targets BEFORE the action.
    head = s[:chosen_pos]
    targets: list[int] = []
    cur = ""
    for ch in head + " ":
        if ch.isdigit():
            cur += ch
        elif cur:
            t = int(cur)
            if t != me and 0 <= t < obs.base.num_players and t not in targets:
                targets.append(t)
            cur = ""
    if not targets:
        return None

    # First number AFTER the chosen action = side payment (default 0).
    after_act = s[chosen_pos + len(chosen):]
    pay_str = _scan_first_number(after_act)
    payment = float(COALITION_DEFAULT_SIDE_PAYMENT)
    if pay_str is not None:
        try:
            payment = max(0.0, min(_SIDE_PAYMENT_MAX, float(pay_str)))
        except ValueError:
            pass

    return CoalitionProposal(
        proposer=me, members=[me, *targets],
        agreed_action=chosen, side_payment=payment,
    )


def llm_negotiate(generate_fn: Callable[[str], str], obs) -> CoalitionAction:
    """Two LLM calls per round: respond to pending proposals + optionally propose one."""
    resp_completion = generate_fn(_build_negotiate_prompt(obs))
    accept = set(_parse_acceptance_indices(resp_completion, len(obs.pending_proposals)))
    me = obs.base.player_index
    responses = [
        CoalitionResponse(responder=me, proposal_index=i, accepted=(i in accept))
        for i in range(len(obs.pending_proposals))
    ]
    proposal = _parse_proposal(generate_fn(_build_propose_prompt(obs)), obs)
    proposals = [proposal] if proposal is not None else []
    return CoalitionAction(responses=responses, proposals=proposals)


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
