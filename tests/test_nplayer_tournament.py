"""Tests for N-player and coalition tournament runners."""
import sys
import types

sys.path.insert(
    int(),
    "/Users/lukaszbartoszcze/Documents/OpenEnv/kant",
)

_openenv_stub = types.ModuleType("openenv")
_core_stub = types.ModuleType("openenv.core")
_server_stub = types.ModuleType("openenv.core.env_server")
_iface_stub = types.ModuleType("openenv.core.env_server.interfaces")
class _EnvironmentStub:
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
    def __class_getitem__(cls, params: object) -> type:
        return cls
    def __init__(self) -> None:
        pass
_iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
_openenv_stub.core = _core_stub  # type: ignore[attr-defined]
_core_stub.env_server = _server_stub  # type: ignore[attr-defined]
_server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
for _name, _mod in [
    ("openenv", _openenv_stub),
    ("openenv.core", _core_stub),
    ("openenv.core.env_server", _server_stub),
    ("openenv.core.env_server.interfaces", _iface_stub),
]:
    sys.modules[_name] = _mod

# Ensure coalition games are registered
from common.games_meta import coalition_config as _  # noqa: F401

from env.nplayer.models import NPlayerAction, NPlayerObservation
from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionObservation, CoalitionResponse,
)
from bench.evaluation.nplayer.nplayer_tournament import (
    NPlayerTournamentRunner, NPlayerTournamentResults,
)
from bench.evaluation.nplayer.coalition_tournament import (
    CoalitionTournamentRunner, CoalitionTournamentResults,
)
from constant_definitions.nplayer.nplayer_constants import NPLAYER_DEFAULT_ROUNDS

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE


def _always_cooperate_np(obs: NPlayerObservation) -> NPlayerAction:
    return NPlayerAction(action=obs.available_actions[_ZERO])


# -- NPlayerTournamentRunner tests --


class TestNPlayerTournament:

    def test_runs_single_game_single_strategy(self) -> None:
        runner = NPlayerTournamentRunner(agent_fn=_always_cooperate_np)
        results = runner.run_tournament(
            games=["coalition_cartel"],
            strategies=["random"],
            num_episodes=_ONE,
        )
        assert isinstance(results, NPlayerTournamentResults)
        assert results.total_episodes == _ONE
        assert "coalition_cartel" in results.games

    def test_episode_result_has_all_scores(self) -> None:
        runner = NPlayerTournamentRunner(agent_fn=_always_cooperate_np)
        results = runner.run_tournament(
            games=["coalition_cartel"],
            strategies=["always_cooperate"],
            num_episodes=_ONE,
        )
        ep = results.games["coalition_cartel"].strategy_results[
            "always_cooperate"
        ].episodes[_ZERO]
        assert len(ep.all_scores) > _ONE
        assert ep.rounds_played == NPLAYER_DEFAULT_ROUNDS

    def test_multiple_strategies(self) -> None:
        runner = NPlayerTournamentRunner(agent_fn=_always_cooperate_np)
        strats = ["always_cooperate", "always_defect"]
        results = runner.run_tournament(
            games=["coalition_cartel"],
            strategies=strats,
            num_episodes=_ONE,
        )
        assert results.total_episodes == _TWO
        game_res = results.games["coalition_cartel"]
        for s in strats:
            assert s in game_res.strategy_results

    def test_cooperation_rate_computed(self) -> None:
        runner = NPlayerTournamentRunner(agent_fn=_always_cooperate_np)
        results = runner.run_tournament(
            games=["coalition_alliance"],
            strategies=["always_cooperate"],
            num_episodes=_ONE,
        )
        ep = results.games["coalition_alliance"].strategy_results[
            "always_cooperate"
        ].episodes[_ZERO]
        assert ep.cooperation_rate >= float()


# -- CoalitionTournamentRunner tests --


class _SimpleCoalitionAgent:
    """Accepts all proposals, picks first action."""

    def negotiate(self, obs: CoalitionObservation) -> CoalitionAction:
        responses = [
            CoalitionResponse(
                responder=_ZERO, proposal_index=idx, accepted=True,
            )
            for idx in range(len(obs.pending_proposals))
        ]
        return CoalitionAction(responses=responses)

    def act(self, obs: CoalitionObservation) -> NPlayerAction:
        return NPlayerAction(action=obs.base.available_actions[_ZERO])


class TestCoalitionTournament:

    def test_runs_single_game(self) -> None:
        agent = _SimpleCoalitionAgent()
        runner = CoalitionTournamentRunner(agent=agent)
        results = runner.run_tournament(
            games=["coalition_cartel"],
            strategies=["coalition_random"],
            num_episodes=_ONE,
        )
        assert isinstance(results, CoalitionTournamentResults)
        assert results.total_episodes == _ONE
        assert "coalition_cartel" in results.games

    def test_episode_metrics(self) -> None:
        agent = _SimpleCoalitionAgent()
        runner = CoalitionTournamentRunner(agent=agent)
        results = runner.run_tournament(
            games=["coalition_alliance"],
            strategies=["coalition_loyal"],
            num_episodes=_ONE,
        )
        ep = results.games["coalition_alliance"][
            "coalition_loyal"
        ].episodes[_ZERO]
        assert ep.rounds_played > _ZERO
        assert ep.coalition_formation_rate >= float()
        assert ep.defection_rate >= float()

    def test_multiple_strategies(self) -> None:
        agent = _SimpleCoalitionAgent()
        runner = CoalitionTournamentRunner(agent=agent)
        strats = ["coalition_random", "coalition_loyal"]
        results = runner.run_tournament(
            games=["coalition_cartel"],
            strategies=strats,
            num_episodes=_ONE,
        )
        assert results.total_episodes == _TWO
        for s in strats:
            assert s in results.games["coalition_cartel"]
