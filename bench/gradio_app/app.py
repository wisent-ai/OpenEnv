"""Kant Gradio Demo -- self-contained HuggingFace Spaces app."""
from __future__ import annotations
import sys, os, random as _rand
import gradio as gr

# ---------------------------------------------------------------------------
# sys.path: allow importing shared constants when running inside the repo
# ---------------------------------------------------------------------------
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
_NONE_VARIANT = "None"

try:
    from constant_definitions.game_constants import DEFAULT_NUM_ROUNDS
except ImportError:
    DEFAULT_NUM_ROUNDS = _TEN

# ---------------------------------------------------------------------------
# Full game registry + tag system
# ---------------------------------------------------------------------------
try:
    from common.games import GAMES
    from common.games_meta.game_tags import (
        GAME_TAGS, get_games_by_tag, list_categories,
    )
    _CATEGORY_DIMS = list_categories()
    _HAS_REGISTRY = True
except ImportError:
    GAMES = None
    _CATEGORY_DIMS = {}
    _HAS_REGISTRY = False

# ---------------------------------------------------------------------------
# N-player and coalition game registries
# ---------------------------------------------------------------------------
_HAS_NPLAYER = False
_NPLAYER_GAMES: dict = {}
try:
    from common.games_meta.nplayer_config import NPLAYER_GAMES as _NP_GAMES
    from common.games_meta.nplayer_games import _BUILTIN_NPLAYER_GAMES  # trigger registration
    from common.games_meta.coalition_config import COALITION_GAMES  # trigger registration
    _NPLAYER_GAMES = dict(_NP_GAMES)
    _HAS_NPLAYER = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Variant system
# ---------------------------------------------------------------------------
_HAS_VARIANTS = False
_VARIANT_NAMES: list[str] = []
try:
    from common.variants import _VARIANT_REGISTRY, compose_game
    _VARIANT_NAMES = sorted(_VARIANT_REGISTRY.keys())
    _HAS_VARIANTS = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# N-player environment + strategies
# ---------------------------------------------------------------------------
_HAS_NPLAYER_ENV = False
try:
    from env.nplayer.environment import NPlayerEnvironment
    from env.nplayer.models import NPlayerAction
    from env.nplayer.strategies import NPLAYER_STRATEGIES
    _HAS_NPLAYER_ENV = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Build unified game info from all registries
# ---------------------------------------------------------------------------
# Maps display_name -> {actions, description, payoff_fn, default_rounds, key, num_players, game_type}
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

# Add N-player games not already in the registry
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

# ---------------------------------------------------------------------------
# Category filter helpers
# ---------------------------------------------------------------------------
def _filter_game_names(category_tag):
    if not _HAS_REGISTRY or category_tag == _ALL_FILTER:
        return sorted(_GAME_INFO.keys())
    matching_keys = get_games_by_tag(category_tag)
    return sorted(_KEY_TO_NAME[k] for k in matching_keys if k in _KEY_TO_NAME)

# ---------------------------------------------------------------------------
# 2-player strategies (from the real strategy registry)
# ---------------------------------------------------------------------------
try:
    from common.strategies import STRATEGIES as _STRAT_REGISTRY
    STRATEGIES_2P = _STRAT_REGISTRY
    _HAS_FULL_STRATEGIES = True
except ImportError:
    # Minimal fallback
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
    _HAS_FULL_STRATEGIES = False

# N-player strategy names
_NPLAYER_STRAT_NAMES = list(NPLAYER_STRATEGIES.keys()) if _HAS_NPLAYER_ENV else ["random"]

# ---------------------------------------------------------------------------
# Multiplayer type filter
# ---------------------------------------------------------------------------
_MP_FILTER_ALL = "All Games"
_MP_FILTER_2P = "2-Player"
_MP_FILTER_NP = "Multiplayer (3+)"
_MP_FILTERS = [_MP_FILTER_ALL, _MP_FILTER_2P, _MP_FILTER_NP]

def _is_nplayer(gname):
    info = _GAME_INFO.get(gname, {})
    return info.get("num_players", _TWO) > _TWO

def _filter_by_mp(mp_filter, names):
    if mp_filter == _MP_FILTER_2P:
        return [n for n in names if not _is_nplayer(n)]
    if mp_filter == _MP_FILTER_NP:
        return [n for n in names if _is_nplayer(n)]
    return names

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _blank(gname, sname):
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    return {"game": gname, "strategy": sname, "history": [],
            "p_score": _ZERO, "o_score": _ZERO, "round": _ZERO,
            "max_rounds": info.get("default_rounds", DEFAULT_NUM_ROUNDS),
            "done": False, "num_players": np,
            "scores": [_ZERO] * np,
            "nplayer_env": None, "variant": None}

def _render(st):
    np = st.get("num_players", _TWO)
    is_mp = np > _TWO
    variant = st.get("variant")

    game_label = st["game"]
    if variant and variant != _NONE_VARIANT:
        game_label += f" + {variant}"

    lines = [f"**Game:** {game_label}  |  **Players:** {np}  |  **Opponent:** {st['strategy']}",
             f"**Round:** {st['round']} / {st['max_rounds']}"]

    if is_mp:
        scores = st.get("scores", [])
        score_parts = [f"P{i}: {s:.1f}" for i, s in enumerate(scores)]
        lines.append(f"**Scores:** {' | '.join(score_parts)}")
    else:
        lines.append(f"**Your score:** {st['p_score']}  |  **Opponent score:** {st['o_score']}")

    if st["done"]:
        lines.append("\n### Game Over")

    if is_mp:
        # N-player history table
        header_cols = ["Round"] + [f"P{i}" for i in range(np)] + [f"Pay{i}" for i in range(np)]
        lines.append("\n| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["-------"] * len(header_cols)) + "|")
        for r in st["history"]:
            actions = r.get("actions", [])
            payoffs = r.get("payoffs", [])
            row = [str(r["round"])]
            row.extend(str(a) for a in actions)
            row.extend(f"{p:.1f}" for p in payoffs)
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("\n| Round | You | Opponent | Your Pay | Opp Pay |")
        lines.append("|-------|-----|----------|----------|---------|")
        for r in st["history"]:
            lines.append(f"| {r['round']} | {r['player_action']} | "
                         f"{r['opponent_action']} | {r['p_pay']} | {r['o_pay']} |")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def play_round(action_str, state):
    if state is None or state["done"]:
        return state, "Reset the game to play again.", gr.update(), gr.update()

    # Use resolved info stored in state (handles variants correctly)
    info = state.get("resolved_info") or _GAME_INFO[state["game"]]
    np = state.get("num_players", _TWO)
    is_mp = np > _TWO

    if is_mp and _HAS_NPLAYER_ENV:
        # N-player game via NPlayerEnvironment
        nenv = state.get("nplayer_env")
        if nenv is None:
            return state, "Error: N-player environment not initialized.", gr.update(), gr.update()
        obs = nenv.step(NPlayerAction(action=action_str))
        last = obs.last_round
        state["round"] += _ONE
        state["scores"] = list(obs.scores)
        state["history"].append({
            "round": state["round"],
            "actions": list(last.actions),
            "payoffs": list(last.payoffs),
        })
        if obs.done:
            state["done"] = True
        acts = info["actions"]
        return (state, _render(state), info["description"],
                gr.update(choices=acts, value=acts[_ZERO]))
    else:
        # 2-player game
        opp_actions = info.get("opponent_actions")
        if opp_actions:
            opp_act_list = list(opp_actions)
        else:
            opp_act_list = info["actions"]
        strat = STRATEGIES_2P[state["strategy"]]
        game_type = info.get("game_type", "matrix")
        if _HAS_FULL_STRATEGIES:
            opp = strat.choose_action(game_type, opp_act_list, state["history"])
        else:
            opp = strat(opp_act_list, state["history"])
        p_pay, o_pay = info["payoff_fn"](action_str, opp)
        state["round"] += _ONE
        state["p_score"] += p_pay
        state["o_score"] += o_pay
        state["history"].append({"round": state["round"], "player_action": action_str,
                                 "opponent_action": opp, "p_pay": p_pay, "o_pay": o_pay})
        if state["round"] >= state["max_rounds"]:
            state["done"] = True
        acts = info["actions"]
        return (state, _render(state), info["description"],
                gr.update(choices=acts, value=acts[_ZERO]))

def _resolve_game_info(gname, variant):
    """Return the game info dict, possibly with variant applied."""
    info = _GAME_INFO.get(gname, {})
    if not variant or variant == _NONE_VARIANT or not _HAS_VARIANTS:
        return info
    key = info.get("key", "")
    if not key:
        return info
    try:
        composed = compose_game(key, variant)
        return {
            "actions": composed.actions, "description": composed.description,
            "payoff_fn": composed.payoff_fn, "default_rounds": composed.default_rounds,
            "key": key, "num_players": composed.num_players,
            "game_type": composed.game_type,
            "opponent_actions": composed.opponent_actions,
        }
    except (KeyError, ValueError):
        return info

def reset_game(gname, sname, variant):
    info_resolved = _resolve_game_info(gname, variant)
    base_info = _GAME_INFO.get(gname, {})
    np = info_resolved.get("num_players", _TWO)
    is_mp = np > _TWO

    st = {
        "game": gname, "strategy": sname, "history": [],
        "p_score": _ZERO, "o_score": _ZERO, "round": _ZERO,
        "max_rounds": info_resolved.get("default_rounds", DEFAULT_NUM_ROUNDS),
        "done": False, "num_players": np,
        "scores": [_ZERO] * np,
        "nplayer_env": None,
        "variant": variant if variant != _NONE_VARIANT else None,
        "resolved_info": info_resolved,
    }

    if is_mp and _HAS_NPLAYER_ENV:
        nenv = NPlayerEnvironment()
        game_key = base_info.get("key", "")
        strat_list = [sname] * (np - _ONE)
        nenv.reset(game_key, opponent_strategies=strat_list)
        st["nplayer_env"] = nenv

    acts = info_resolved["actions"]
    return (st, _render(st), info_resolved["description"],
            gr.update(choices=acts, value=acts[_ZERO]))

def on_game_change(gname, sname, variant):
    return reset_game(gname, sname, variant)

def on_category_change(tag, mp_filter):
    names = _filter_game_names(tag)
    names = _filter_by_mp(mp_filter, names)
    if not names:
        names = sorted(_GAME_INFO.keys())
    return gr.update(choices=names, value=names[_ZERO])

def on_mp_filter_change(mp_filter, tag):
    return on_category_change(tag, mp_filter)

def on_game_select(gname):
    """Update UI elements when a game is selected."""
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    is_mp = np > _TWO
    if is_mp and _HAS_NPLAYER_ENV:
        strat_names = _NPLAYER_STRAT_NAMES
    else:
        strat_names = list(STRATEGIES_2P.keys())
    player_label = f"Players: {np}" if is_mp else "2-Player"
    return (gr.update(choices=strat_names, value=strat_names[_ZERO]),
            gr.update(value=player_label))

# ---------------------------------------------------------------------------
# Variant filter: some variants only work for 2-player games
# ---------------------------------------------------------------------------
_2P_ONLY_VARIANTS = {"noisy_actions", "noisy_payoffs", "self_play", "cross_model"}

def on_game_select_variant(gname):
    """Update available variants when game changes."""
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    is_mp = np > _TWO
    if is_mp or not _HAS_VARIANTS:
        available = [_NONE_VARIANT]
    else:
        available = [_NONE_VARIANT] + [v for v in _VARIANT_NAMES
                                        if v not in _2P_ONLY_VARIANTS or not is_mp]
    return gr.update(choices=available, value=_NONE_VARIANT)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_GAME_NAMES = sorted(_GAME_INFO.keys())
_STRAT_NAMES = list(STRATEGIES_2P.keys())
_INIT_GAME = _GAME_NAMES[_ZERO] if _GAME_NAMES else "Prisoner's Dilemma"
_INIT_STRAT = _STRAT_NAMES[_ZERO]
_INIT_ACTS = _GAME_INFO[_INIT_GAME]["actions"] if _INIT_GAME in _GAME_INFO else ["cooperate", "defect"]

_TAG_CHOICES = [_ALL_FILTER]
for _dn, _dt in sorted(_CATEGORY_DIMS.items()):
    _TAG_CHOICES.extend(_dt)

_VARIANT_CHOICES = [_NONE_VARIANT] + _VARIANT_NAMES if _HAS_VARIANTS else [_NONE_VARIANT]

_init_np = _GAME_INFO.get(_INIT_GAME, {}).get("num_players", _TWO)
_init_player_label = f"Players: {_init_np}" if _init_np > _TWO else "2-Player"

with gr.Blocks(title="Kant Demo") as demo:
    gr.Markdown("# Kant -- Interactive Game Theory Demo")
    with gr.Tabs():
        with gr.TabItem("Human Play"):
            with gr.Row():
                cat_dd = gr.Dropdown(_TAG_CHOICES, value=_ALL_FILTER, label="Filter by Category")
                mp_dd = gr.Dropdown(_MP_FILTERS, value=_MP_FILTER_ALL, label="Player Count")
                game_dd = gr.Dropdown(_GAME_NAMES, value=_INIT_GAME, label="Game")
            with gr.Row():
                strat_dd = gr.Dropdown(_STRAT_NAMES, value=_INIT_STRAT, label="Opponent Strategy")
                variant_dd = gr.Dropdown(_VARIANT_CHOICES, value=_NONE_VARIANT, label="Variant")
                player_info = gr.Textbox(value=_init_player_label, label="Mode", interactive=False)
                reset_btn = gr.Button("Reset / New Game")
            game_desc = gr.Markdown(value=_GAME_INFO[_INIT_GAME]["description"])
            with gr.Row():
                action_dd = gr.Dropdown(_INIT_ACTS, value=_INIT_ACTS[_ZERO], label="Your Action")
                play_btn = gr.Button("Play Round", variant="primary")
            state_var = gr.State(_blank(_INIT_GAME, _INIT_STRAT))
            history_md = gr.Markdown(value=_render(_blank(_INIT_GAME, _INIT_STRAT)))

            # Wiring
            cat_dd.change(on_category_change, inputs=[cat_dd, mp_dd], outputs=[game_dd])
            mp_dd.change(on_mp_filter_change, inputs=[mp_dd, cat_dd], outputs=[game_dd])
            play_btn.click(play_round, inputs=[action_dd, state_var],
                           outputs=[state_var, history_md, game_desc, action_dd])
            reset_btn.click(reset_game, inputs=[game_dd, strat_dd, variant_dd],
                            outputs=[state_var, history_md, game_desc, action_dd])
            game_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_dd],
                           outputs=[state_var, history_md, game_desc, action_dd])
            game_dd.change(on_game_select, inputs=[game_dd],
                           outputs=[strat_dd, player_info])
            game_dd.change(on_game_select_variant, inputs=[game_dd],
                           outputs=[variant_dd])
            strat_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_dd],
                            outputs=[state_var, history_md, game_desc, action_dd])
            variant_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_dd],
                              outputs=[state_var, history_md, game_desc, action_dd])
        with gr.TabItem("Game Theory Reference"):
            # Build a reference table of all games
            ref_lines = ["# Game Theory Reference\n"]
            ref_lines.append("| Game | Players | Type | Actions | Rounds |")
            ref_lines.append("|------|---------|------|---------|--------|")
            for _gn in sorted(_GAME_INFO.keys()):
                _gi = _GAME_INFO[_gn]
                _np = _gi.get("num_players", _TWO)
                _gt = _gi.get("game_type", "")
                _acts = _gi["actions"]
                _act_str = ", ".join(_acts[:_FIVE])
                if len(_acts) > _FIVE:
                    _act_str += f" ... ({len(_acts)} total)"
                _dr = _gi.get("default_rounds", DEFAULT_NUM_ROUNDS)
                ref_lines.append(f"| {_gn} | {_np} | {_gt} | {_act_str} | {_dr} |")
            ref_lines.append(f"\n**Total games: {len(_GAME_INFO)}**")
            if _HAS_VARIANTS:
                ref_lines.append(f"\n**Available variants:** {', '.join(_VARIANT_NAMES)}")
            gr.Markdown("\n".join(ref_lines))

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
