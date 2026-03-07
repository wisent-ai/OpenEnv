"""KantBench: a game theory RL environment for OpenEnv.

Each episode is one repeated game (e.g. Prisoner's Dilemma) against a
fixed strategy opponent.  The agent chooses a move each round; the
environment computes payoffs and returns a structured observation.

Supported games: Prisoner's Dilemma, Stag Hunt, Hawk-Dove,
                 Battle of Sexes, Chicken, Matching Pennies,
                 Rock-Paper-Scissors.
Opponent strategies: random, always_first, always_last, tit_for_tat,
                     grim_trigger, pavlov.
"""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import KantBenchAction, KantBenchObservation

# ---------------------------------------------------------------------------
# Game definitions (self-contained payoff matrices)
# ---------------------------------------------------------------------------

def _matrix(m: dict[tuple[str, str], tuple[float, float]]):
    """Return a payoff function from a matrix dict."""
    def fn(a: str, b: str) -> tuple[float, float]:
        return m[(a, b)]
    return fn


GAMES: dict[str, dict[str, Any]] = {
    "prisoners_dilemma": {
        "name": "Prisoner's Dilemma",
        "description": (
            "Two players choose to cooperate or defect simultaneously. "
            "Mutual cooperation is best collectively; defection is individually tempting."
        ),
        "actions": ["cooperate", "defect"],
        "rounds": 10,
        "payoff_fn": _matrix({
            ("cooperate", "cooperate"): (3.0, 3.0),
            ("cooperate", "defect"):    (0.0, 5.0),
            ("defect",    "cooperate"): (5.0, 0.0),
            ("defect",    "defect"):    (1.0, 1.0),
        }),
    },
    "stag_hunt": {
        "name": "Stag Hunt",
        "description": (
            "Two hunters choose to hunt stag (requires coordination) or hare "
            "(safe alone). Mutual cooperation yields the best outcome."
        ),
        "actions": ["stag", "hare"],
        "rounds": 10,
        "payoff_fn": _matrix({
            ("stag", "stag"): (4.0, 4.0),
            ("stag", "hare"): (0.0, 2.0),
            ("hare", "stag"): (2.0, 0.0),
            ("hare", "hare"): (2.0, 2.0),
        }),
    },
    "hawk_dove": {
        "name": "Hawk-Dove",
        "description": (
            "Two players compete over a resource. Hawk is aggressive; Dove is passive. "
            "Two hawks fight and both lose; two doves share."
        ),
        "actions": ["hawk", "dove"],
        "rounds": 10,
        "payoff_fn": _matrix({
            ("hawk", "hawk"): (-1.0, -1.0),
            ("hawk", "dove"): (4.0,   0.0),
            ("dove", "hawk"): (0.0,   4.0),
            ("dove", "dove"): (2.0,   2.0),
        }),
    },
    "battle_of_sexes": {
        "name": "Battle of the Sexes",
        "description": (
            "Two players want to coordinate but prefer different options. "
            "Player 1 prefers opera; Player 2 prefers football. "
            "Both prefer to be together over going alone."
        ),
        "actions": ["opera", "football"],
        "rounds": 10,
        "payoff_fn": _matrix({
            ("opera",    "opera"):    (3.0, 1.0),
            ("opera",    "football"): (0.0, 0.0),
            ("football", "opera"):    (0.0, 0.0),
            ("football", "football"): (1.0, 3.0),
        }),
    },
    "chicken": {
        "name": "Chicken (Snowdrift)",
        "description": (
            "Two drivers head toward each other. Swerving is safe but cowardly; "
            "going straight is bold but catastrophic if both do it."
        ),
        "actions": ["straight", "swerve"],
        "rounds": 10,
        "payoff_fn": _matrix({
            ("straight", "straight"): (-10.0, -10.0),
            ("straight", "swerve"):   (5.0,   -1.0),
            ("swerve",   "straight"): (-1.0,   5.0),
            ("swerve",   "swerve"):   (0.0,    0.0),
        }),
    },
    "matching_pennies": {
        "name": "Matching Pennies",
        "description": (
            "Player 1 wins if both show the same side; Player 2 wins if they differ. "
            "Pure zero-sum game with no stable pure-strategy Nash equilibrium."
        ),
        "actions": ["heads", "tails"],
        "rounds": 20,
        "payoff_fn": _matrix({
            ("heads", "heads"): (1.0,  -1.0),
            ("heads", "tails"): (-1.0,  1.0),
            ("tails", "heads"): (-1.0,  1.0),
            ("tails", "tails"): (1.0,  -1.0),
        }),
    },
    "rock_paper_scissors": {
        "name": "Rock-Paper-Scissors",
        "description": (
            "Classic zero-sum game. Rock beats Scissors, Scissors beats Paper, "
            "Paper beats Rock. Ties yield 0."
        ),
        "actions": ["rock", "paper", "scissors"],
        "rounds": 20,
        "payoff_fn": _matrix({
            ("rock",     "rock"):     (0.0,  0.0),
            ("rock",     "paper"):    (-1.0, 1.0),
            ("rock",     "scissors"): (1.0, -1.0),
            ("paper",    "rock"):     (1.0, -1.0),
            ("paper",    "paper"):    (0.0,  0.0),
            ("paper",    "scissors"): (-1.0, 1.0),
            ("scissors", "rock"):     (-1.0, 1.0),
            ("scissors", "paper"):    (1.0, -1.0),
            ("scissors", "scissors"): (0.0,  0.0),
        }),
    },
}

STRATEGIES = ["random", "always_first", "always_last", "tit_for_tat", "grim_trigger", "pavlov"]


def _opponent_move(strategy: str, actions: list[str], history: list[dict]) -> str:
    """Compute opponent's move given strategy and history."""
    if strategy == "random":
        return random.choice(actions)
    if strategy == "always_first":
        return actions[0]
    if strategy == "always_last":
        return actions[-1]
    if not history:
        return actions[0]  # default opening for reactive strategies
    last_agent_move = history[-1]["your_move"]
    last_opp_move   = history[-1]["opponent_move"]
    if strategy == "tit_for_tat":
        return last_agent_move if last_agent_move in actions else actions[0]
    if strategy == "grim_trigger":
        # Defect forever once agent defects; cooperate otherwise
        ever_defected = any(r["your_move"] == actions[-1] for r in history)
        return actions[-1] if ever_defected else actions[0]
    if strategy == "pavlov":
        # Repeat own last move if it paid well (i.e. opponent cooperated), else switch
        if last_opp_move == actions[0]:
            return last_opp_move  # mirror cooperation
        return actions[0] if last_opp_move != actions[0] else actions[-1]
    return random.choice(actions)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class KantbenchEnvironment(Environment):
    """Game theory environment for benchmarking LLM strategic reasoning.

    Each episode is a repeated 2-player game against one of six opponent
    strategies.  The agent submits a move each round and receives the payoff
    result as a structured observation.

    Example::

        env = KantBenchEnvironment()
        obs = env.reset()
        # obs.game_name == "Prisoner's Dilemma"
        # obs.available_moves == ["cooperate", "defect"]

        obs = env.step(KantBenchAction(move="cooperate"))
        # obs.your_payoff, obs.opponent_move, obs.cumulative_score, ...
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._game_key: str = "prisoners_dilemma"
        self._strategy: str = "random"
        self._history: list[dict] = []
        self._cumulative_score: float = 0.0

    def reset(self, **kwargs) -> KantBenchObservation:
        """Start a new episode with a randomly chosen game and opponent strategy."""
        self._game_key = random.choice(list(GAMES.keys()))
        self._strategy = random.choice(STRATEGIES)
        self._history = []
        self._cumulative_score = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        game = GAMES[self._game_key]
        return KantBenchObservation(
            game_name=game["name"],
            game_description=game["description"],
            available_moves=game["actions"],
            your_move="",
            opponent_move="",
            your_payoff=0.0,
            opponent_payoff=0.0,
            cumulative_score=0.0,
            round_number=0,
            max_rounds=game["rounds"],
            opponent_strategy=self._strategy,
            history=[],
            done=False,
            reward=0.0,
            message=(
                f"New episode: {game['name']} vs {self._strategy}. "
                f"Choose one of: {game['actions']}"
            ),
        )

    def step(self, action: KantBenchAction, **kwargs) -> KantBenchObservation:  # type: ignore[override]
        """Play one round of the current game."""
        game = GAMES[self._game_key]
        actions = game["actions"]
        max_rounds = game["rounds"]

        # Validate move
        move = action.move.lower().strip()
        if move not in actions:
            closest = actions[0]
            move = closest

        opp_move = _opponent_move(self._strategy, actions, self._history)
        your_pay, opp_pay = game["payoff_fn"](move, opp_move)

        self._state.step_count += 1
        self._cumulative_score += your_pay

        round_record = {
            "round": self._state.step_count,
            "your_move": move,
            "opponent_move": opp_move,
            "your_payoff": your_pay,
            "opponent_payoff": opp_pay,
        }
        self._history.append(round_record)

        done = self._state.step_count >= max_rounds

        return KantBenchObservation(
            game_name=game["name"],
            game_description=game["description"],
            available_moves=actions,
            your_move=move,
            opponent_move=opp_move,
            your_payoff=your_pay,
            opponent_payoff=opp_pay,
            cumulative_score=self._cumulative_score,
            round_number=self._state.step_count,
            max_rounds=max_rounds,
            opponent_strategy=self._strategy,
            history=self._history,
            done=done,
            reward=your_pay,
            message="Game over — call reset() to start a new episode." if done else "",
        )

    @property
    def state(self) -> State:
        return self._state
