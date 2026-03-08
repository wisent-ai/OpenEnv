"""Tournament runner for coalition formation and governance evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from common.games_meta.coalition_config import COALITION_GAMES
from env.nplayer.coalition.environment import CoalitionEnvironment
from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionObservation, CoalitionResponse,
)
from env.nplayer.coalition.strategies import COALITION_STRATEGIES
from env.nplayer.models import NPlayerAction
from constant_definitions.game_constants import (
    COALITION_EVAL_DEFAULT_EPISODES,
    EVAL_ONE, EVAL_ZERO, EVAL_ZERO_FLOAT,
)

_ZERO = int()


class CoalitionAgentProtocol(Protocol):
    """Protocol for agents compatible with CoalitionTournamentRunner."""

    def negotiate(
        self, obs: CoalitionObservation,
    ) -> CoalitionAction: ...

    def act(
        self, obs: CoalitionObservation,
    ) -> NPlayerAction: ...


@dataclass
class CoalitionEpisodeResult:
    """Outcome of a single coalition episode."""
    game: str
    strategy: str
    player_score: float
    adjusted_scores: List[float]
    rounds_played: int
    coalition_formation_rate: float
    defection_rate: float
    governance_proposals_count: int
    governance_adopted_count: int
    governance_rejected_count: int


@dataclass
class CoalitionStrategyResults:
    """Aggregated results for one coalition strategy across episodes."""
    strategy_name: str
    episodes: List[CoalitionEpisodeResult] = field(default_factory=list)
    total_player_score: float = EVAL_ZERO_FLOAT
    mean_coalition_rate: float = EVAL_ZERO_FLOAT
    mean_defection_rate: float = EVAL_ZERO_FLOAT


@dataclass
class CoalitionTournamentResults:
    """Full coalition tournament output container."""
    games: Dict[str, Dict[str, CoalitionStrategyResults]] = field(
        default_factory=dict,
    )
    total_episodes: int = EVAL_ZERO
    games_played: List[str] = field(default_factory=list)
    strategies_tested: List[str] = field(default_factory=list)


def _default_negotiate(obs: CoalitionObservation) -> CoalitionAction:
    """Accept all pending proposals, make no new ones."""
    responses = [
        CoalitionResponse(
            responder=_ZERO, proposal_index=idx, accepted=True,
        )
        for idx in range(len(obs.pending_proposals))
    ]
    return CoalitionAction(responses=responses)


def _default_act(obs: CoalitionObservation) -> NPlayerAction:
    """Pick the first available action."""
    return NPlayerAction(action=obs.base.available_actions[_ZERO])


class _DefaultCoalitionAgent:
    """Simple agent that accepts all proposals and cooperates."""

    def negotiate(self, obs: CoalitionObservation) -> CoalitionAction:
        return _default_negotiate(obs)

    def act(self, obs: CoalitionObservation) -> NPlayerAction:
        return _default_act(obs)


class CoalitionTournamentRunner:
    """Orchestrates coalition tournaments across games and strategies."""

    def __init__(
        self,
        env: Optional[CoalitionEnvironment] = None,
        agent: Optional[CoalitionAgentProtocol] = None,
    ) -> None:
        self._env = env if env is not None else CoalitionEnvironment()
        self._agent: CoalitionAgentProtocol = (
            agent if agent is not None else _DefaultCoalitionAgent()
        )

    def run_tournament(
        self,
        games: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        num_episodes: int = COALITION_EVAL_DEFAULT_EPISODES,
        tags: Optional[Sequence[str]] = None,
    ) -> CoalitionTournamentResults:
        """Execute the full coalition tournament."""
        if tags is not None:
            from common.games_meta.game_tags import get_games_by_tags
            tagged = set(get_games_by_tags(*tags))
            game_keys = sorted(tagged & set(COALITION_GAMES.keys()))
        elif games is not None:
            game_keys = list(games)
        else:
            game_keys = list(COALITION_GAMES.keys())
        strat_keys = (
            list(strategies) if strategies is not None
            else list(COALITION_STRATEGIES.keys())
        )
        results = CoalitionTournamentResults(
            games_played=list(game_keys),
            strategies_tested=list(strat_keys),
        )
        episode_counter = EVAL_ZERO
        for g_key in game_keys:
            game_strats: Dict[str, CoalitionStrategyResults] = {}
            for s_key in strat_keys:
                strat_res = CoalitionStrategyResults(strategy_name=s_key)
                for _ep in range(num_episodes):
                    ep_result = self._run_episode(g_key, s_key)
                    strat_res.episodes.append(ep_result)
                    strat_res.total_player_score += ep_result.player_score
                    episode_counter += EVAL_ONE
                ep_count = len(strat_res.episodes)
                if ep_count > EVAL_ZERO:
                    strat_res.mean_coalition_rate = sum(
                        e.coalition_formation_rate
                        for e in strat_res.episodes
                    ) / ep_count
                    strat_res.mean_defection_rate = sum(
                        e.defection_rate for e in strat_res.episodes
                    ) / ep_count
                game_strats[s_key] = strat_res
            results.games[g_key] = game_strats
        results.total_episodes = episode_counter
        return results

    def _run_episode(
        self, game_key: str, strategy_key: str,
    ) -> CoalitionEpisodeResult:
        """Play a single coalition episode."""
        cfg = COALITION_GAMES[game_key]
        num_opp = cfg.num_players - EVAL_ONE
        opp_strats = [strategy_key] * num_opp
        obs = self._env.reset(
            game=game_key, coalition_strategies=opp_strats,
        )
        rounds_with_coalitions = EVAL_ZERO
        rounds_with_defections = EVAL_ZERO
        total_rounds = EVAL_ZERO
        gov_proposals = EVAL_ZERO
        gov_adopted = EVAL_ZERO
        gov_rejected = EVAL_ZERO
        while not obs.base.done:
            neg_action = self._agent.negotiate(obs)
            obs = self._env.negotiate_step(neg_action)
            game_action = self._agent.act(obs)
            obs = self._env.action_step(game_action)
            total_rounds += EVAL_ONE
            if obs.coalition_history:
                last_round = obs.coalition_history[-EVAL_ONE]
                if last_round.active_coalitions:
                    rounds_with_coalitions += EVAL_ONE
                if last_round.defectors:
                    rounds_with_defections += EVAL_ONE
            if obs.governance_history:
                last_gov = obs.governance_history[-EVAL_ONE]
                gov_proposals += len(last_gov.proposals)
                gov_adopted += len(last_gov.adopted)
                gov_rejected += len(last_gov.rejected)
        coal_rate = (
            rounds_with_coalitions / total_rounds
            if total_rounds > EVAL_ZERO else EVAL_ZERO_FLOAT
        )
        defect_rate = (
            rounds_with_defections / total_rounds
            if total_rounds > EVAL_ZERO else EVAL_ZERO_FLOAT
        )
        return CoalitionEpisodeResult(
            game=game_key, strategy=strategy_key,
            player_score=obs.adjusted_scores[_ZERO],
            adjusted_scores=list(obs.adjusted_scores),
            rounds_played=total_rounds,
            coalition_formation_rate=coal_rate,
            defection_rate=defect_rate,
            governance_proposals_count=gov_proposals,
            governance_adopted_count=gov_adopted,
            governance_rejected_count=gov_rejected,
        )
