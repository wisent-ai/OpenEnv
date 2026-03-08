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
    from common.variants import _VARIANT_REGISTRY, compose_game
    _CATEGORY_DIMS = list_categories()
    _HAS_REGISTRY = True
    _HUMAN_VARIANTS = [
        v for v in _VARIANT_REGISTRY
        if v not in ("self_play", "cross_model")
    ]
except ImportError:
    from constant_definitions.game_constants import (
        PD_CC_PAYOFF, PD_CD_PAYOFF, PD_DC_PAYOFF, PD_DD_PAYOFF,
        SH_SS_PAYOFF, SH_SH_PAYOFF, SH_HS_PAYOFF, SH_HH_PAYOFF,
        HD_HH_PAYOFF, HD_HD_PAYOFF, HD_DH_PAYOFF, HD_DD_PAYOFF,
    )
    GAMES = None
    _CATEGORY_DIMS = {}
    _HAS_REGISTRY = False
    _HUMAN_VARIANTS = []
    _VARIANT_REGISTRY = {}
    compose_game = None

# Maps display_name -> {actions, description, payoff_fn, default_rounds, key}
_GAME_INFO: dict[str, dict] = {}
_KEY_TO_NAME: dict[str, str] = {}

if _HAS_REGISTRY:
    for _key in sorted(GAMES.keys()):
        _cfg = GAMES[_key]
        _GAME_INFO[_cfg.name] = {
            "actions": _cfg.actions, "description": _cfg.description,
            "payoff_fn": _cfg.payoff_fn, "default_rounds": _cfg.default_rounds,
            "key": _key,
        }
        _KEY_TO_NAME[_key] = _cfg.name
else:
    _FB = {
        "Prisoner's Dilemma": (
            ["cooperate", "defect"], "prisoners_dilemma",
            "Two players simultaneously choose to cooperate or defect.",
            {("cooperate", "cooperate"): (PD_CC_PAYOFF, PD_CC_PAYOFF),
             ("cooperate", "defect"): (PD_CD_PAYOFF, PD_DC_PAYOFF),
             ("defect", "cooperate"): (PD_DC_PAYOFF, PD_CD_PAYOFF),
             ("defect", "defect"): (PD_DD_PAYOFF, PD_DD_PAYOFF)},
        ),
        "Stag Hunt": (
            ["stag", "hare"], "stag_hunt",
            "Two players choose between hunting stag or hare.",
            {("stag", "stag"): (SH_SS_PAYOFF, SH_SS_PAYOFF),
             ("stag", "hare"): (SH_SH_PAYOFF, SH_HS_PAYOFF),
             ("hare", "stag"): (SH_HS_PAYOFF, SH_SH_PAYOFF),
             ("hare", "hare"): (SH_HH_PAYOFF, SH_HH_PAYOFF)},
        ),
        "Hawk-Dove": (
            ["hawk", "dove"], "hawk_dove",
            "Two players choose between aggressive (hawk) and passive (dove).",
            {("hawk", "hawk"): (HD_HH_PAYOFF, HD_HH_PAYOFF),
             ("hawk", "dove"): (HD_HD_PAYOFF, HD_DH_PAYOFF),
             ("dove", "hawk"): (HD_DH_PAYOFF, HD_HD_PAYOFF),
             ("dove", "dove"): (HD_DD_PAYOFF, HD_DD_PAYOFF)},
        ),
    }
    for _gn, (_acts, _gk, _desc, _mat) in _FB.items():
        def _mk(m):
            return lambda a, b: m[(a, b)]
        _GAME_INFO[_gn] = {
            "actions": _acts, "description": _desc,
            "payoff_fn": _mk(_mat), "default_rounds": DEFAULT_NUM_ROUNDS,
            "key": _gk,
        }
        _KEY_TO_NAME[_gk] = _gn

# ---------------------------------------------------------------------------
# Category filter helpers
# ---------------------------------------------------------------------------
def _filter_game_names(category_tag):
    if not _HAS_REGISTRY or category_tag == _ALL_FILTER:
        return sorted(_GAME_INFO.keys())
    matching_keys = get_games_by_tag(category_tag)
    return sorted(_KEY_TO_NAME[k] for k in matching_keys if k in _KEY_TO_NAME)

# ---------------------------------------------------------------------------
# Inline strategies
# ---------------------------------------------------------------------------
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

STRATEGIES = {"random": _strat_random, "always_cooperate": _strat_first,
              "always_defect": _strat_last, "tit_for_tat": _strat_tft}

# ---------------------------------------------------------------------------
# Variant-aware game info
# ---------------------------------------------------------------------------
def _get_game_info(gname, variants=None):
    """Return game info dict, applying selected variants if any."""
    base_info = _GAME_INFO.get(gname)
    if not base_info or not variants or not _HAS_REGISTRY:
        return base_info
    base_key = base_info["key"]
    try:
        cfg = compose_game(base_key, *variants)
        return {"actions": cfg.actions, "description": cfg.description,
                "payoff_fn": cfg.payoff_fn, "default_rounds": cfg.default_rounds,
                "key": base_key}
    except (KeyError, ValueError):
        return base_info

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _blank(gname, sname, variants=None):
    info = _get_game_info(gname, variants)
    if info is None:
        info = {}
    return {"game": gname, "strategy": sname, "history": [],
            "p_score": _ZERO, "o_score": _ZERO, "round": _ZERO,
            "max_rounds": info.get("default_rounds", DEFAULT_NUM_ROUNDS),
            "done": False, "variants": list(variants or [])}

def _render(st):
    vlist = st.get("variants", [])
    vtag = f"  |  **Variants:** {', '.join(vlist)}" if vlist else ""
    lines = [f"**Game:** {st['game']}  |  **Opponent:** {st['strategy']}{vtag}",
             f"**Round:** {st['round']} / {st['max_rounds']}",
             f"**Your score:** {st['p_score']}  |  **Opponent score:** {st['o_score']}"]
    if st["done"]:
        lines.append("\n### Game Over")
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
    info = _get_game_info(state["game"], state.get("variants"))
    opp = STRATEGIES[state["strategy"]](info["actions"], state["history"])
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

def reset_game(gname, sname, variants=None):
    vlist = list(variants or [])
    st = _blank(gname, sname, vlist)
    info = _get_game_info(gname, vlist)
    acts = info["actions"]
    return (st, _render(st), info["description"],
            gr.update(choices=acts, value=acts[_ZERO]))

def on_game_change(gname, sname, variants=None):
    return reset_game(gname, sname, variants)

def on_category_change(tag):
    names = _filter_game_names(tag)
    if not names:
        names = sorted(_GAME_INFO.keys())
    return gr.update(choices=names, value=names[_ZERO])

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_GAME_NAMES = sorted(_GAME_INFO.keys())
_STRAT_NAMES = list(STRATEGIES.keys())
_INIT_GAME = _GAME_NAMES[_ZERO]
_INIT_STRAT = _STRAT_NAMES[_ZERO]
_INIT_ACTS = _GAME_INFO[_INIT_GAME]["actions"]

_TAG_CHOICES = [_ALL_FILTER]
for _dn, _dt in sorted(_CATEGORY_DIMS.items()):
    _TAG_CHOICES.extend(_dt)

with gr.Blocks(title="Kant Demo") as demo:
    gr.Markdown("# Kant -- Interactive Game Theory Demo")
    with gr.Tabs():
        with gr.TabItem("Human Play"):
            with gr.Row():
                cat_dd = gr.Dropdown(_TAG_CHOICES, value=_ALL_FILTER, label="Filter by Category")
                game_dd = gr.Dropdown(_GAME_NAMES, value=_INIT_GAME, label="Game")
                strat_dd = gr.Dropdown(_STRAT_NAMES, value=_INIT_STRAT, label="Opponent Strategy")
                reset_btn = gr.Button("Reset / New Game")
            if _HUMAN_VARIANTS:
                variant_cb = gr.CheckboxGroup(
                    _HUMAN_VARIANTS, value=[], label="Variants",
                    info="Apply transforms: communication, uncertainty, commitment, etc.",
                )
            else:
                variant_cb = gr.CheckboxGroup([], value=[], label="Variants", visible=False)
            game_desc = gr.Markdown(value=_GAME_INFO[_INIT_GAME]["description"])
            with gr.Row():
                action_dd = gr.Dropdown(_INIT_ACTS, value=_INIT_ACTS[_ZERO], label="Your Action")
                play_btn = gr.Button("Play Round", variant="primary")
            state_var = gr.State(_blank(_INIT_GAME, _INIT_STRAT))
            history_md = gr.Markdown(value=_render(_blank(_INIT_GAME, _INIT_STRAT)))
            _reset_out = [state_var, history_md, game_desc, action_dd]
            cat_dd.change(on_category_change, inputs=[cat_dd], outputs=[game_dd])
            play_btn.click(play_round, inputs=[action_dd, state_var],
                           outputs=_reset_out)
            reset_btn.click(reset_game, inputs=[game_dd, strat_dd, variant_cb],
                            outputs=_reset_out)
            game_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                           outputs=_reset_out)
            strat_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                            outputs=_reset_out)
            variant_cb.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                              outputs=_reset_out)
        with gr.TabItem("Game Theory Reference"):
            gr.Markdown("# Game Theory Reference\n\nUse the Human Play tab to "
                        "explore all games. Filter by category to find games by "
                        "communication level, information structure, payoff type, "
                        "domain, and more.")

demo.launch()
