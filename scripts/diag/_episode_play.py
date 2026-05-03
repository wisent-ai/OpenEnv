"""Per-env episode-play helpers for run_llama_kantbench.py.

Three env paths share a (game_key, score, rounds) row contract so the
caller can aggregate uniformly. Each helper takes the player's generate_fn
(prompt -> completion string) and the env-specific opponent spec.

Extracted from the runner because the runner sits at the per-file
300-line cap and N-player support would push it over.
"""

from __future__ import annotations

from typing import Callable, Optional

import train.agent as _train_agent
from env.environment import KantEnvironment
from env.models import GameAction, GameObservation, RoundResult
from env.nplayer.environment import NPlayerEnvironment
from env.nplayer.models import NPlayerAction, NPlayerObservation
from train.agent import LLMAgent, PromptBuilder

PARSE_MISS_COUNT = 0
PARSE_TOTAL_COUNT = 0


def install_parse_action_counter():
    """Wrap train.agent.parse_action so the runner can report the miss rate."""
    import train.agent as _agent_mod
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
        return _original(response, available_actions)

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
    """Compact prompt for NPlayerObservation. Mirrors the 2P PromptBuilder
    schema (game name, description, scores, history, available actions,
    instruction) but exposes the full scores list and the player's index."""
    sections = [
        f"[Game]\n{obs.game_name}\n{obs.game_description}",
        f"[Players] {obs.num_players}, you are P{obs.player_index}",
        "[Scores] " + ", ".join(
            f"P{i}={s:g}" for i, s in enumerate(obs.scores)
        ),
        f"[Round] {obs.current_round} of {obs.total_rounds}",
        "[Available Actions]\n" + "\n".join(f"- {a}" for a in obs.available_actions),
        "[Instruction] Reply with exactly one of the listed actions.",
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
