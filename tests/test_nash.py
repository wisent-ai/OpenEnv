"""Tests for bench/nash.py -- Nash-equilibrium distance metric.

Covers TV distance correctness, empirical action aggregation,
multi-equilibrium minimisation, the empty-equilibria error, and an
end-to-end check that a defect-only policy in Prisoner's Dilemma is at
distance zero from the Nash equilibrium.
"""

from __future__ import annotations

import sys
import types

if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw: object) -> None:
            super().__init_subclass__(**kw)
        def __class_getitem__(cls, params: object) -> type:
            return cls
        def __init__(self) -> None:
            pass

    _iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
    _openenv_stub.core = _core_stub  # type: ignore[attr-defined]
    _core_stub.env_server = _server_stub  # type: ignore[attr-defined]
    _server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

from bench.evaluation.tournament import (
    EpisodeResult,
    GameResults,
    StrategyResults,
    TournamentResults,
)
from bench.nash import (
    compute_nash_distances,
    empirical_action_distribution,
    nash_distance,
    total_variation,
)


def test_total_variation_identical_distributions_is_zero():
    p = {"cooperate": 0.5, "defect": 0.5}
    assert total_variation(p, p) == 0.0


def test_total_variation_disjoint_supports_is_one():
    p = {"cooperate": 1.0}
    q = {"defect": 1.0}
    assert total_variation(p, q) == 1.0


def test_total_variation_handles_missing_keys():
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 1.0}
    assert total_variation(p, q) == 0.5


def test_total_variation_empty_inputs():
    assert total_variation({}, {}) == 0.0


def test_empirical_action_distribution_aggregates_across_episodes():
    eps = [
        EpisodeResult(
            game="prisoners_dilemma", strategy="tit_for_tat",
            player_score=0, opponent_score=0, rounds_played=2,
            cooperation_rate=0,
            history=[
                {"player_action": "defect", "opponent_action": "cooperate",
                 "player_payoff": 5, "opponent_payoff": 0},
                {"player_action": "defect", "opponent_action": "defect",
                 "player_payoff": 1, "opponent_payoff": 1},
            ],
        ),
        EpisodeResult(
            game="prisoners_dilemma", strategy="tit_for_tat",
            player_score=0, opponent_score=0, rounds_played=2,
            cooperation_rate=0,
            history=[
                {"player_action": "cooperate", "opponent_action": "defect",
                 "player_payoff": 0, "opponent_payoff": 5},
                {"player_action": "defect", "opponent_action": "defect",
                 "player_payoff": 1, "opponent_payoff": 1},
            ],
        ),
    ]
    dist = empirical_action_distribution(eps)
    assert dist == {"defect": 0.75, "cooperate": 0.25}


def test_empirical_action_distribution_empty_returns_empty_dict():
    assert empirical_action_distribution([]) == {}


def test_nash_distance_picks_nearest_equilibrium():
    # Stag Hunt has three NE: (1,0), (0,1), (2/3, 1/3).
    sh_equilibria = (
        {"stag": 1.0, "hare": 0.0},
        {"stag": 0.0, "hare": 1.0},
        {"stag": 2.0 / 3.0, "hare": 1.0 / 3.0},
    )
    # Empirical (0.65, 0.35) is closest to the mixed NE (2/3, 1/3),
    # at TV distance ~ 0.0167 < 0.35 (vs pure stag) < 0.65 (vs pure hare).
    empirical = {"stag": 0.65, "hare": 0.35}
    d = nash_distance(empirical, sh_equilibria)
    assert d < 0.05


def test_nash_distance_empty_equilibria_raises():
    try:
        nash_distance({"a": 1.0}, ())
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty equilibria")


def test_compute_nash_distances_zero_for_pure_defect_in_pd():
    """An agent that always defects sits exactly on the PD Nash equilibrium."""
    pd_episode = EpisodeResult(
        game="prisoners_dilemma", strategy="tit_for_tat",
        player_score=10, opponent_score=10, rounds_played=10,
        cooperation_rate=0.0,
        history=[
            {"player_action": "defect", "opponent_action": "defect",
             "player_payoff": 1, "opponent_payoff": 1}
            for _ in range(10)
        ],
    )
    strat_res = StrategyResults(
        strategy_name="tit_for_tat",
        episodes=[pd_episode],
        total_player_score=10,
        total_opponent_score=10,
        mean_cooperation_rate=0.0,
    )
    game_res = GameResults(
        game_name="Prisoner's Dilemma",
        strategy_results={"tit_for_tat": strat_res},
    )
    results = TournamentResults(
        games={"prisoners_dilemma": game_res},
        total_episodes=1,
        games_played=["prisoners_dilemma"],
        strategies_tested=["tit_for_tat"],
    )
    distances = compute_nash_distances(results)
    assert "prisoners_dilemma" in distances
    assert distances["prisoners_dilemma"]["tit_for_tat"] == 0.0


def test_compute_nash_distances_skips_unknown_games():
    """Games whose key isn't in the GAMES registry must not appear in output."""
    fake_episode = EpisodeResult(
        game="not_a_real_game_key_xyz", strategy="default",
        player_score=0, opponent_score=0, rounds_played=1,
        cooperation_rate=0.0,
        history=[
            {"player_action": "x", "opponent_action": "y",
             "player_payoff": 0, "opponent_payoff": 0},
        ],
    )
    strat_res = StrategyResults(
        strategy_name="default",
        episodes=[fake_episode],
    )
    game_res = GameResults(
        game_name="Fake",
        strategy_results={"default": strat_res},
    )
    results = TournamentResults(
        games={"not_a_real_game_key_xyz": game_res},
        total_episodes=1,
    )
    assert compute_nash_distances(results) == {}


def test_compute_nash_distances_includes_ultimatum_now():
    """Ultimatum gained an SPE entry (offer_0); confirm it scores."""
    offer_5_eps = [
        EpisodeResult(
            game="ultimatum", strategy="default",
            player_score=5, opponent_score=5, rounds_played=1,
            cooperation_rate=0.5,
            history=[
                {"player_action": "offer_5", "opponent_action": "accept",
                 "player_payoff": 5, "opponent_payoff": 5},
            ],
        ),
    ]
    strat_res = StrategyResults(
        strategy_name="default",
        episodes=offer_5_eps,
    )
    results = TournamentResults(
        games={"ultimatum": GameResults(
            game_name="Ultimatum",
            strategy_results={"default": strat_res},
        )},
        total_episodes=1,
    )
    distances = compute_nash_distances(results)
    # Empirical {offer_5: 1.0} vs SPE {offer_0: 1.0} -> TV distance 1.0.
    assert distances["ultimatum"]["default"] == 1.0
