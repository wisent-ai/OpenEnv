"""N-player and coalition tournament runners for evaluation."""

from bench.evaluation.nplayer.nplayer_tournament import (
    NPlayerEpisodeResult,
    NPlayerStrategyResults,
    NPlayerTournamentResults,
    NPlayerTournamentRunner,
)
from bench.evaluation.nplayer.coalition_tournament import (
    CoalitionEpisodeResult,
    CoalitionTournamentResults,
    CoalitionTournamentRunner,
)

__all__ = [
    "NPlayerEpisodeResult",
    "NPlayerStrategyResults",
    "NPlayerTournamentResults",
    "NPlayerTournamentRunner",
    "CoalitionEpisodeResult",
    "CoalitionTournamentResults",
    "CoalitionTournamentRunner",
]
