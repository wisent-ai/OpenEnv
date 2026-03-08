"""Game registry, strategies, and filters for the Kant Gradio app."""
from __future__ import annotations
import sys, os, random as _rand

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(int(), _REPO_ROOT)

_ZERO = int()
_ONE = int(bool(True))
_TWO = _ONE + _ONE
_THREE = _TWO + _ONE
_FOUR = _THREE + _ONE
_FIVE = _FOUR + _ONE
_NEG_ONE = -_ONE
_TEN = _FIVE + _FIVE
_ALL_FILTER = "All"

try:
    from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
except ImportError:
    DEFAULT_NUM_ROUNDS = _TEN

# -- Full game registry + tag system --
_HAS_REGISTRY = False
_CATEGORY_DIMS: dict = {}
try:
    from common.games import GAMES
    from common.games_meta.game_tags import GAME_TAGS, get_games_by_tag, list_categories
    _CATEGORY_DIMS = list_categories()
    _HAS_REGISTRY = True
except ImportError:
    GAMES = None
    GAME_TAGS = {}
    get_games_by_tag = lambda tag: []
    list_categories = lambda: {}

# -- N-player and coalition --
_HAS_NPLAYER = False
_NPLAYER_GAMES: dict = {}
try:
    from common.games_meta.nplayer_config import NPLAYER_GAMES as _NP_GAMES
    from common.games_meta.nplayer_games import _BUILTIN_NPLAYER_GAMES  # noqa: F401
    from common.games_meta.coalition_config import COALITION_GAMES  # noqa: F401
    _NPLAYER_GAMES = dict(_NP_GAMES)
    _HAS_NPLAYER = True
except ImportError:
    pass

# -- Variant system --
_HAS_VARIANTS = False
_VARIANT_NAMES: list[str] = []
_VARIANT_REGISTRY: dict = {}
compose_game = None
try:
    from common.variants import _VARIANT_REGISTRY, compose_game
    _VARIANT_NAMES = sorted(_VARIANT_REGISTRY.keys())
    _HAS_VARIANTS = True
except ImportError:
    pass

# -- N-player environment + strategies --
_HAS_NPLAYER_ENV = False
NPlayerEnvironment = None
NPlayerAction = None
NPLAYER_STRATEGIES: dict = {}
try:
    from env.nplayer.environment import NPlayerEnvironment
    from env.nplayer.models import NPlayerAction
    from env.nplayer.strategies import NPLAYER_STRATEGIES
    _HAS_NPLAYER_ENV = True
except ImportError:
    pass

# -- Build unified game info --
_GAME_INFO: dict[str, dict] = {}
_KEY_TO_NAME: dict[str, str] = {}

if _HAS_REGISTRY:
    for _key in sorted(GAMES.keys()):
        _cfg = GAMES[_key]
        _GAME_INFO[_cfg.name] = {
            "actions": _cfg.actions, "description": _cfg.description,
            "payoff_fn": _cfg.payoff_fn, "default_rounds": _cfg.default_rounds,
            "key": _key, "num_players": _cfg.num_players,
            "game_type": _cfg.game_type,
            "opponent_actions": _cfg.opponent_actions,
        }
        _KEY_TO_NAME[_key] = _cfg.name

if _HAS_NPLAYER:
    for _key, _cfg in _NPLAYER_GAMES.items():
        if _key not in _KEY_TO_NAME:
            _GAME_INFO[_cfg.name] = {
                "actions": _cfg.actions, "description": _cfg.description,
                "payoff_fn": _cfg.payoff_fn, "default_rounds": _cfg.default_rounds,
                "key": _key, "num_players": _cfg.num_players,
                "game_type": _cfg.game_type,
                "opponent_actions": getattr(_cfg, "opponent_actions", None),
            }
            _KEY_TO_NAME[_key] = _cfg.name

# -- Category filter --
def _filter_game_names(category_tag):
    if not _HAS_REGISTRY or category_tag == _ALL_FILTER:
        return sorted(_GAME_INFO.keys())
    matching_keys = get_games_by_tag(category_tag)
    return sorted(_KEY_TO_NAME[k] for k in matching_keys if k in _KEY_TO_NAME)

# -- Two-player strategies --
_HAS_FULL_STRATEGIES = False
try:
    from common.strategies import STRATEGIES as _STRAT_REGISTRY
    STRATEGIES_2P = _STRAT_REGISTRY
    _HAS_FULL_STRATEGIES = True
except ImportError:
    def _strat_random(actions, _h):
        return _rand.choice(actions)
    def _strat_first(actions, _h):
        return actions[_ZERO]
    def _strat_last(actions, _h):
        return actions[min(_ONE, len(actions) - _ONE)]
    def _strat_tft(actions, h):
        if not h:
            return actions[_ZERO]
        prev = h[_NEG_ONE]["player_action"]
        return prev if prev in actions else actions[_ZERO]
    STRATEGIES_2P = {"random": _strat_random, "always_cooperate": _strat_first,
                     "always_defect": _strat_last, "tit_for_tat": _strat_tft}

_NPLAYER_STRAT_NAMES = list(NPLAYER_STRATEGIES.keys()) if _HAS_NPLAYER_ENV else ["random"]

_GENERIC_STRATEGIES = [
    "random", "always_cooperate", "always_defect", "tit_for_tat",
    "tit_for_two_tats", "grudger", "pavlov", "suspicious_tit_for_tat",
    "generous_tit_for_tat", "adaptive", "mixed",
]
_GAME_TYPE_STRATEGIES: dict[str, list[str]] = {
    "ultimatum": ["ultimatum_fair", "ultimatum_low"],
    "trust": ["trust_fair", "trust_generous"],
    "public_goods": ["public_goods_fair", "public_goods_free_rider"],
    "threshold_public_goods": ["public_goods_fair", "public_goods_free_rider"],
}

def _strategies_for_game(gname: str) -> list[str]:
    info = _GAME_INFO.get(gname, {})
    game_type = info.get("game_type", "matrix")
    available = list(_GENERIC_STRATEGIES)
    available.extend(_GAME_TYPE_STRATEGIES.get(game_type, []))
    return [s for s in available if s in STRATEGIES_2P]

# -- Multiplayer filter --
_MP_FILTER_ALL = "All Games"
_MP_FILTER_TWO = "Two-Player"
_MP_FILTER_NP = "Multiplayer (N)"
_MP_FILTERS = [_MP_FILTER_ALL, _MP_FILTER_TWO, _MP_FILTER_NP]

def _is_nplayer(gname):
    return _GAME_INFO.get(gname, {}).get("num_players", _TWO) > _TWO

def _filter_by_mp(mp_filter, names):
    if mp_filter == _MP_FILTER_TWO:
        return [n for n in names if not _is_nplayer(n)]
    if mp_filter == _MP_FILTER_NP:
        return [n for n in names if _is_nplayer(n)]
    return names

# -- Variant filter --
_2P_ONLY_VARIANTS = {"noisy_actions", "noisy_payoffs", "self_play", "cross_model"}
_HUMAN_VARIANTS = [v for v in _VARIANT_NAMES if v not in ("self_play", "cross_model")]

# -- LLM opponent support --
_HAS_LLM_AGENT = False
try:
    from train.agent import PromptBuilder, parse_action
    from env.models import GameObservation, GameAction, RoundResult
    _HAS_LLM_AGENT = True
except ImportError:
    PromptBuilder = None
    parse_action = None
    GameObservation = None
    GameAction = None
    RoundResult = None

try:
    from constant_definitions.train.models.anthropic_constants import (
        CLAUDE_OPUS, CLAUDE_SONNET, CLAUDE_HAIKU,
    )
except ImportError:
    CLAUDE_OPUS = "claude-opus-four-six"
    CLAUDE_SONNET = "claude-sonnet-four-six"
    CLAUDE_HAIKU = "claude-haiku-four-five"

try:
    from constant_definitions.train.models.openai_constants import (
        GPT_4O_MINI, GPT_4O, GPT_5_4, O3_MINI, O3, O4_MINI,
    )
except ImportError:
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_5_4 = "gpt-5.4"
    O3_MINI = "o3-mini"
    O3 = "o3"
    O4_MINI = "o4-mini"

try:
    from constant_definitions.train.agent_constants import SYSTEM_PROMPT as _SYS_PROMPT
except ImportError:
    _SYS_PROMPT = (
        "You are playing a game-theory game. Analyse the situation and choose "
        "the best action. Respond with ONLY the action name, nothing else."
    )

_LLM_PROVIDERS = ["Anthropic", "OpenAI"]
_LLM_MODELS = {
    "Anthropic": [CLAUDE_HAIKU, CLAUDE_SONNET, CLAUDE_OPUS],
    "OpenAI": [GPT_4O_MINI, GPT_4O, GPT_5_4, O3_MINI, O3, O4_MINI],
}
_LLM_OPPONENT_LABEL = "LLM"

# -- OAuth token support --
try:
    from train.self_play.oauth import (
        get_anthropic_access_token as _get_ant_token,
        get_openai_credentials as _get_oai_creds,
    )
    from constant_definitions.var.meta.self_play_constants import (
        ANTHROPIC_OAUTH_BETA_HEADER,
    )
    _HAS_OAUTH = True
except ImportError:
    _HAS_OAUTH = False
    ANTHROPIC_OAUTH_BETA_HEADER = ""

import os as _os
_ENV_API_KEYS = {
    "Anthropic": _os.environ.get("ANTHROPIC_API_KEY", ""),
    "OpenAI": _os.environ.get("OPENAI_API_KEY", ""),
}


def get_env_api_key(provider: str) -> str | None:
    """Get an OAuth token (preferred) or env var API key."""
    if _HAS_OAUTH:
        try:
            if provider == "Anthropic":
                return _get_ant_token()
            if provider == "OpenAI":
                tok, _ = _get_oai_creds()
                return tok
        except Exception:
            pass
    key = _ENV_API_KEYS.get(provider, "")
    return key if key else None
