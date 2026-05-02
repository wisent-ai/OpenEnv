"""Extension-loading side effects for ``common.games``.

Pure relocation from ``common/games.py``: imports every games_ext /
games_info / games_market / games_coop / meta / games_adaptive submodule
so their ``@register_game`` (or equivalent) decorators run and populate
``GAME_FACTORIES``. Lives in its own module so ``common/games.py`` can
stay under the per-file line cap.
"""

from __future__ import annotations


_EXTENSION_MODULES = (
    "common.games_ext.matrix_games",
    "common.games_ext.sequential",
    "common.games_ext.auction",
    "common.games_ext.nplayer",
    "common.games_ext.generated",
    "common.games_info.signaling",
    "common.games_info.contracts",
    "common.games_info.communication",
    "common.games_info.bayesian",
    "common.games_info.network",
    "common.games_market.oligopoly",
    "common.games_market.contests",
    "common.games_market.classic",
    "common.games_market.generated_v2",
    "common.games_market.advanced",
    "common.games_coop.cooperative",
    "common.games_coop.dynamic",
    "common.games_coop.pd_variants",
    "common.games_coop.infinite",
    "common.games_coop.stochastic",
    "common.meta.meta_games",
    "common.games_adaptive.factories",
)


def load_extensions() -> None:
    """Import every extension module, swallowing ImportError on missing optional deps."""
    import importlib

    for mod in _EXTENSION_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
