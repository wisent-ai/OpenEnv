"""Per-env episode-play helpers for run_llama_kantbench.py.

Three env paths share a (game_key, score, rounds) row contract so the
caller can aggregate uniformly. Each helper takes the player's generate_fn
(prompt -> completion string) and the env-specific opponent spec.

Extracted from the runner because the runner sits at the per-file
300-line cap and N-player support would push it over.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

import train.agent as _train_agent
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation, RoundResult
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction, NPlayerObservation
from env.nplayer.coalition.environment import CoalitionEnvironment
from env.nplayer.coalition.models import CoalitionAction, CoalitionResponse
from constant_definitions.nplayer.coalition_constants import (
    COALITION_PHASE_NEGOTIATE, COALITION_PHASE_ACTION,
)
from train.agent import LLMAgent, PromptBuilder

PARSE_MISS_COUNT = 0
PARSE_TOTAL_COUNT = 0
LOG_COMPLETIONS = False
_JSONL_FH = None
_CTX: dict = {"game": None, "opponent": None}
_LOG = logging.getLogger(__name__)


def set_logging_context(*, game: Optional[str] = None,
                        opponent: Optional[str] = None) -> None:
    """Stamp game / opponent on subsequent parse_action wrapper rows."""
    if game is not None:
        _CTX["game"] = game
    if opponent is not None:
        _CTX["opponent"] = opponent


def open_jsonl_log(path: str) -> None:
    """Open a per-round JSONL sink. Each parse_action call appends one row."""
    global _JSONL_FH
    _JSONL_FH = open(path, "a", encoding="utf-8")


def close_jsonl_log() -> None:
    global _JSONL_FH
    if _JSONL_FH is not None:
        _JSONL_FH.flush()
        _JSONL_FH.close()
        _JSONL_FH = None


def install_parse_action_counter(log_completions: bool = False):
    """Wrap train.agent.parse_action so the runner can report the miss rate."""
    import train.agent as _agent_mod
    global LOG_COMPLETIONS
    LOG_COMPLETIONS = log_completions
    _original = _agent_mod.parse_action

    def _wrapped(response: str, available_actions):
        global PARSE_MISS_COUNT, PARSE_TOTAL_COUNT
        PARSE_TOTAL_COUNT += 1
        stripped = response.strip()
        lower = stripped.lower()
        matched = (
            stripped in available_actions
            or any(a.lower() == lower for a in available_actions)
            or any(a.lower() in lower for a in available_actions)
        )
        if not matched:
            PARSE_MISS_COUNT += 1
        parsed = _original(response, available_actions)
        if LOG_COMPLETIONS:
            _LOG.info("move=%d game=%s opp=%s parsed=%s raw=%r",
                      PARSE_TOTAL_COUNT, _CTX.get("game"), _CTX.get("opponent"),
                      parsed, response)
        if _JSONL_FH is not None:
            _JSONL_FH.write(json.dumps({
                "ts": time.time(), "move": PARSE_TOTAL_COUNT,
                "game": _CTX.get("game"), "opponent": _CTX.get("opponent"),
                "raw": response, "parsed": parsed,
                "available_actions": list(available_actions),
                "parse_miss": not matched,
            }, ensure_ascii=False) + "\n")
            _JSONL_FH.flush()
        return parsed

    _agent_mod.parse_action = _wrapped


def _agent_fn_from_llm(agent: LLMAgent):
    def _fn(obs: GameObservation) -> GameAction:
        return agent(obs)
    return _fn


def make_2p_agent(generate_fn: Callable[[str], str]) -> Callable[[GameObservation], GameAction]:
    """Wrap a generate_fn into the 2P GameObservation -> GameAction interface."""
    return _agent_fn_from_llm(LLMAgent(generate_fn=generate_fn))


def play_episode_2p(env: KantEnvironment, agent_fn, *, game: str, **reset_kw):
    """Play one full episode of *game* on KantEnvironment. Returns (score, rounds)."""
    obs = env.reset(game=game, **reset_kw)
    while not obs.done:
        action = agent_fn(obs)
        obs = env.step(action)
    return obs.player_score, obs.current_round


def _build_nplayer_prompt(obs: NPlayerObservation) -> str:
    """Tight prompt for NPlayerObservation. Lists allowed action tokens
    inline so the model sees them in the instruction sentence as well as
    the action menu, and asks for one-word output."""
    actions_inline = " | ".join(obs.available_actions)
    actions_list = "\n".join(f"- {a}" for a in obs.available_actions)
    sections = [
        f"[Game]\n{obs.game_name}\n{obs.game_description}",
        f"[Players] {obs.num_players}; you are P{obs.player_index}",
        "[Scores] " + ", ".join(
            f"P{i}={s:g}" for i, s in enumerate(obs.scores)
        ),
        f"[Round] {obs.current_round} of {obs.total_rounds}",
        f"[Available Actions]\n{actions_list}",
        f"[Instruction] Reply with exactly one token from this list and "
        f"nothing else: {actions_inline}",
    ]
    return "\n\n".join(sections)


def make_nplayer_agent(generate_fn: Callable[[str], str]) -> Callable[[NPlayerObservation], NPlayerAction]:
    """Wrap a generate_fn into the N-player Observation -> Action interface.

    Looks up ``train.agent.parse_action`` via the module each call so the
    counter installed by ``install_parse_action_counter`` is honoured even
    though that installer patches the module attribute, not our import.
    """
    def _fn(obs: NPlayerObservation) -> NPlayerAction:
        prompt = _build_nplayer_prompt(obs)
        completion = generate_fn(prompt)
        action = _train_agent.parse_action(completion, obs.available_actions)
        return NPlayerAction(action=action)
    return _fn


def play_episode_nplayer(
    env: NPlayerEnvironment,
    agent_fn: Callable[[NPlayerObservation], NPlayerAction],
    *,
    game: str,
    opponent_strategies: Optional[list[str]] = None,
    opponent_fns: Optional[list[Optional[Callable]]] = None,
):
    """Play one full episode of *game* on NPlayerEnvironment.

    Returns (player_score, rounds_played) where player_score is the
    cumulative payoff for player zero (the agent_fn's role).
    """
    obs = env.reset(
        game=game,
        opponent_strategies=opponent_strategies,
        opponent_fns=opponent_fns,
    )
    while not obs.done:
        action = agent_fn(obs)
        obs = env.step(action)
    score = obs.scores[0] if obs.scores else 0.0
    return score, obs.current_round


_LLM_COALITION_OPPONENT_NAME = "_llm_opponent_for_coalition"


def register_llm_coalition_strategy(generate_fn) -> str:
    """Mutate the coalition strategy registry so the env can resolve our LLM by name.

    generate_fn is the raw prompt -> completion callable; we build the
    n-player agent here so the strategy reuses the same parse-action
    miss counter.
    """
    from env.nplayer.coalition.strategies import COALITION_STRATEGIES
    from _coalition_negotiate import make_coalition_strategy  # type: ignore[import-not-found]
    COALITION_STRATEGIES[_LLM_COALITION_OPPONENT_NAME] = make_coalition_strategy(
        generate_fn, make_nplayer_agent(generate_fn),
    )
    return _LLM_COALITION_OPPONENT_NAME


def play_episode_coalition(
    env: CoalitionEnvironment,
    agent_fn: Callable[[NPlayerObservation], NPlayerAction],
    generate_fn: Callable[[str], str],
    *,
    game: str,
    coalition_strategies: Optional[list[str]] = None,
):
    """Play one full episode of *game* on CoalitionEnvironment.

    Negotiate phase: prompt the LLM with pending proposals via
    _coalition_negotiate.llm_negotiate and submit the parsed responses.
    Action phase: route obs.base through *agent_fn*. Returns
    (player_score, rounds_played) for player zero with coalition
    payoff adjustments folded in via obs.base.scores.
    """
    from _coalition_negotiate import llm_negotiate  # type: ignore[import-not-found]
    obs = env.reset(game=game, coalition_strategies=coalition_strategies)
    while not obs.base.done:
        if obs.phase == COALITION_PHASE_NEGOTIATE:
            obs = env.negotiate_step(llm_negotiate(generate_fn, obs))
        if obs.phase == COALITION_PHASE_ACTION:
            obs = env.action_step(agent_fn(obs.base))
    score = obs.base.scores[0] if obs.base.scores else 0.0
    return score, obs.base.current_round


def _accumulate(call_one, episodes):
    ssum, rsum = 0.0, 0
    for _ in range(episodes):
        ps, rounds = call_one()
        ssum += ps
        rsum += rounds
    return ssum, rsum


def play_rows(env_kind, env, key, strategies, episodes, mode, agent_fn, opp_fn,
              opp_label, generate_fn=None, opp_generate_fn=None):
    """Dispatch one (game, opponent-set) row collection to the right env helper.

    generate_fn / opp_generate_fn are required for env_kind == "coalition";
    coalition negotiation prompts the LLM with the raw prompt->completion
    callable, not the prebuilt agent_fn closure.
    """
    rows = []
    if env_kind == "2p":
        if mode == "hardcoded":
            for s in strategies:
                ssum, rsum = _accumulate(
                    lambda s=s: play_episode_2p(env, agent_fn, game=key, strategy=s),
                    episodes,
                )
                rows.append((key, s, ssum, rsum))
        else:
            ssum, rsum = _accumulate(
                lambda: play_episode_2p(env, agent_fn, game=key, opponent_fn=opp_fn),
                episodes,
            )
            rows.append((key, opp_label, ssum, rsum))
    elif env_kind == "nplayer":
        if mode == "hardcoded":
            for s in strategies:
                ssum, rsum = _accumulate(
                    lambda s=s: play_episode_nplayer(
                        env, agent_fn, game=key, opponent_strategies=[s],
                    ),
                    episodes,
                )
                rows.append((key, s, ssum, rsum))
        else:
            ssum, rsum = _accumulate(
                lambda: play_episode_nplayer(
                    env, agent_fn, game=key, opponent_fns=[opp_fn],
                ),
                episodes,
            )
            rows.append((key, opp_label, ssum, rsum))
    elif env_kind == "coalition":
        if mode == "hardcoded":
            for s in strategies:
                ssum, rsum = _accumulate(
                    lambda s=s: play_episode_coalition(
                        env, agent_fn, generate_fn,
                        game=key, coalition_strategies=[s],
                    ),
                    episodes,
                )
                rows.append((key, s, ssum, rsum))
        else:
            llm_name = register_llm_coalition_strategy(opp_generate_fn or generate_fn)
            ssum, rsum = _accumulate(
                lambda: play_episode_coalition(
                    env, agent_fn, generate_fn,
                    game=key, coalition_strategies=[llm_name],
                ),
                episodes,
            )
            rows.append((key, opp_label, ssum, rsum))
    return rows
