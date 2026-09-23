"""Trainer-side rollouts for 2-phase free_chat games.

The legacy `_play_batch_interactive_episodes` in train.py assumes one
env.step() per round and one parsed action. apply_free_chat games run
on a 2-phase per-round protocol (message → reveal → action) where each
env.step needs to be called TWICE per round and the model is queried
TWICE per round (once for prose, once for the action token).

This module provides `_play_batch_free_chat_episodes` with the same
return shape as the legacy rollout, plus a per-round trajectory list
that captures both message-phase and action-phase emissions. Training
reward stays the action-segment payoff (apply_free_chat's payoff_fn is
identical to the base game); messages do not directly affect reward,
so any message-side behaviour the policy learns emerges naturally
from the joint distribution that GRPO optimizes.
"""
from __future__ import annotations

import logging
from typing import Any

from env.environment import KantEnvironment
from env.models import GameAction
from train.agent import (
    LLMAgent,
    PromptBuilder,
    parse_action,
    parse_free_chat,
)

logger = logging.getLogger(__name__)


def _local_coop_rate(history) -> float:
    """Cooperation rate over the player's actions in this trajectory."""
    if not history:
        return 0.0
    coop = {"cooperate", "stag", "dove", "contribute"}
    n = sum(1 for r in history if any(c in r.player_action for c in coop))
    return n / len(history)


def _constrained_action_generate(model, tokenizer, obs_list, device):
    """Force model.generate to emit one of the available action words exactly.

    A step-0-only constraint isn't enough: small models commit to a
    valid first-token like " co" then drift to "cooking" instead of
    "cooperate", and parse_action's substring match needs the FULL
    action word. So we make the constraint STATEFUL — at every step
    of generation, the only allowed next-tokens are those that keep
    `generated` as a prefix of at least one action's full token
    sequence. Once an action is fully spelled out, allow EOS.

    The constraint is format-only; the policy still chooses WHICH
    action to commit to at step 0 (and any action whose first-token
    has higher logit wins under greedy decode), so the emergent
    strategic dynamics (lying, lie-detection, cooperation) are
    preserved — only the FORMAT is forced into an action vocab.
    """
    import torch

    completions: list[str] = []
    for obs in obs_list:
        actions = obs.available_actions
        history_lines = []
        for r in (obs.history or [])[-3:]:
            history_lines.append(
                f"R{r.round_number}: you={r.player_action} opp={r.opponent_action}"
                f" payoff={r.player_payoff}"
            )
        history_block = ("\n[Recent rounds]\n" + "\n".join(history_lines)) if history_lines else ""
        prompt = (
            f"You are playing {obs.game_name}.{history_block}\n"
            f"\n[Round]\n{obs.current_round} of {obs.total_rounds}"
            f"\n\n[Instruction]\nReply with EXACTLY ONE word: "
            f"{' or '.join(actions)}. Just the single word."
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build the set of viable FULL-word token sequences. Try a few
        # surface forms (leading space, capitalization) since chat templates
        # and tokenizers vary in where the assistant turn starts.
        action_seqs: list[list[int]] = []
        for a in actions:
            for variant in (" " + a, a, " " + a.capitalize(), a.capitalize()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    action_seqs.append(ids)
        if not action_seqs:
            raise RuntimeError(
                f"_constrained_action_generate: tokenizer produced no "
                f"token sequence for any surface form of actions {actions!r}."
            )
        eos_id = tokenizer.eos_token_id

        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        def _prefix_fn(batch_id, input_ids,
                       _plen=prompt_len, _seqs=action_seqs, _eos=eos_id):
            gen = input_ids[_plen:].tolist() if input_ids.ndim == 1 else input_ids[0, _plen:].tolist()
            k = len(gen)
            next_allowed: set[int] = set()
            done_seen = False
            for seq in _seqs:
                if k > len(seq):
                    continue
                if gen == seq[:k]:
                    if k < len(seq):
                        next_allowed.add(seq[k])
                    else:
                        done_seen = True
            if done_seen and _eos is not None:
                next_allowed.add(_eos)
            if not next_allowed:
                return [_eos] if _eos is not None else [0]
            return sorted(next_allowed)

        max_seq_len = max(len(s) for s in action_seqs)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_seq_len + 1,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=_prefix_fn,
            )
        completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        completions.append(completion.strip())
    return completions


def _build_result_from_obs(obs, strategy: str, partial: bool) -> dict:
    """Build the result dict for an episode from its current obs.history.

    Called from both the normal terminal path (obs.done) and the error
    path (parse miss / env.step exception). When `partial=True` the
    episode was cut short — record whatever rounds did complete so the
    trajectory analyzer can still see honest/lie/coop behaviour from the
    completed prefix. Without this, partial rollouts produced result=None
    and the trainer's WISENT_TRAJECTORY_LOG persistence wrote nothing,
    leaving the dynamics-over-training CSV empty.
    """
    coop = {"cooperate", "stag", "dove", "contribute"}
    history = obs.history or []
    opp_coop = (
        sum(1 for r in history if any(c in r.opponent_action for c in coop))
        / len(history)
        if history else 0.0
    )
    trajectory = [
        {
            "round": r.round_number,
            "player_message": r.player_message,
            "opponent_message": r.opponent_message,
            "player_action": r.player_action,
            "opponent_action": r.opponent_action,
            "player_payoff": r.player_payoff,
        }
        for r in history
    ]
    return {
        "player_score": obs.player_score,
        "opponent_score": obs.opponent_score,
        "cooperation_rate": _local_coop_rate(history),
        "opponent_cooperation_rate": opp_coop,
        "rounds": obs.current_round,
        "strategy": strategy,
        "trajectory": trajectory,
        "partial": partial,
    }


def play_batch_free_chat_episodes(
    envs: list[KantEnvironment],
    episode_configs: list[tuple[str, str, str]],
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    batch_generate_fn,
) -> list[dict | None]:
    """Run N free_chat episodes in lockstep, batching message-phase and
    action-phase generations across envs.

    `batch_generate_fn(model, tokenizer, obs_list, device)` is the same
    helper the legacy rollout uses to generate one completion per active
    env in parallel. We call it twice per round: once with phase=message
    obs (model emits prose), once with phase=action obs (model emits
    action token after seeing the opponent's revealed message).

    Returns a list of episode result dicts shaped to match the legacy
    `_play_batch_interactive_episodes` return for drop-in compatibility,
    plus a `trajectory` field listing the (player_message,
    opponent_message, player_action, opponent_action) tuple per round.
    """
    n = len(episode_configs)
    results: list[dict | None] = [None] * n
    obs_list: list[Any] = [None] * n
    active = [True] * n

    for i, (game_key, strategy, _first_action) in enumerate(episode_configs):
        try:
            obs_list[i] = envs[i].reset(game=game_key, strategy=strategy)
        except Exception as exc:
            logger.debug("free_chat init error %s/%s: %s", game_key, strategy, exc)
            active[i] = False

    max_rounds = 40  # message+action phases double the step count per round
    for _ in range(max_rounds):
        # ---- message phase ---------------------------------------------------
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break

        active_obs = [obs_list[i] for i in active_indices]
        for obs in active_obs:
            obs.metadata.setdefault("phase", "message")
            obs.metadata.setdefault("free_chat", True)
        msg_completions = batch_generate_fn(model, tokenizer, active_obs, device)
        for j, i in enumerate(active_indices):
            try:
                obs_list[i] = envs[i].step(GameAction(
                    action=obs_list[i].available_actions[0],  # sentinel; ignored in msg phase
                    metadata={"message": (msg_completions[j] or "").strip()},
                ))
            except Exception as exc:
                logger.debug("free_chat msg step err episode %d: %s", i, exc)
                active[i] = False

        # ---- action phase ---------------------------------------------------
        active_indices = [i for i in range(n) if active[i]]
        if not active_indices:
            break
        action_obs = [obs_list[i] for i in active_indices]
        # Use constrained generation for the action phase to guarantee a
        # parseable action prefix even for small models. parse_action's
        # substring match downstream then resolves it to a valid action.
        act_completions = _constrained_action_generate(
            model, tokenizer, action_obs, device,
        )
        for j, i in enumerate(active_indices):
            try:
                # action phase: parse a bare action token
                act_str = parse_action(
                    (act_completions[j] or "").strip(),
                    obs_list[i].available_actions,
                )
                obs_list[i] = envs[i].step(GameAction(action=act_str))
                if obs_list[i].done:
                    active[i] = False
                    results[i] = _build_result_from_obs(
                        obs_list[i], episode_configs[i][1], partial=False,
                    )
            except Exception as exc:
                logger.debug("free_chat act step err episode %d: %s", i, exc)
                active[i] = False
                results[i] = _build_result_from_obs(
                    obs_list[i], episode_configs[i][1], partial=True,
                )

    return results


def is_free_chat_game(game_key: str) -> bool:
    """True if the game key is a free_chat variant per the registry. Used
    by the trainer dispatch to pick the 2-phase rollout when needed.
    Falls back to a key-prefix check when the registry import fails so
    this can be called from cold-start contexts."""
    if game_key.startswith("free_chat_"):
        return True
    try:
        from common.games import GAMES
        cfg = GAMES.get(game_key)
        if cfg is None:
            return False
        return "free_chat" in (getattr(cfg, "applied_variants", ()) or ())
    except Exception:
        return False
