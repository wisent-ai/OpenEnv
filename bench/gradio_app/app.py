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
# LLM opponent support (prompt builder + action parser)
# ---------------------------------------------------------------------------
_HAS_LLM_AGENT = False
try:
    from train.agent import PromptBuilder, parse_action
    from env.models import GameObservation, GameAction, RoundResult
    _HAS_LLM_AGENT = True
except ImportError:
    pass

try:
    from constant_definitions.train.models.anthropic_constants import (
        CLAUDE_OPUS, CLAUDE_SONNET, CLAUDE_HAIKU,
    )
except ImportError:
    CLAUDE_OPUS = "claude-opus-4-6"
    CLAUDE_SONNET = "claude-sonnet-4-6"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"

try:
    from constant_definitions.train.models.openai_constants import GPT_5_4
except ImportError:
    GPT_5_4 = "gpt-5.4"

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
    "OpenAI": [GPT_5_4, "gpt-4.1-mini", "gpt-4.1-nano"],
}
_LLM_OPPONENT_LABEL = "LLM"

# ---------------------------------------------------------------------------
# Build unified game info from all registries
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Strategy compatibility
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Multiplayer type filter
# ---------------------------------------------------------------------------
_MP_FILTER_ALL = "All Games"
_MP_FILTER_2P = "2-Player"
_MP_FILTER_NP = "Multiplayer (3+)"
_MP_FILTERS = [_MP_FILTER_ALL, _MP_FILTER_2P, _MP_FILTER_NP]

def _is_nplayer(gname):
    return _GAME_INFO.get(gname, {}).get("num_players", _TWO) > _TWO

def _filter_by_mp(mp_filter, names):
    if mp_filter == _MP_FILTER_2P:
        return [n for n in names if not _is_nplayer(n)]
    if mp_filter == _MP_FILTER_NP:
        return [n for n in names if _is_nplayer(n)]
    return names

# ---------------------------------------------------------------------------
# Variant-aware game info
# ---------------------------------------------------------------------------
def _get_game_info(gname, variants=None):
    base_info = _GAME_INFO.get(gname)
    if not base_info or not variants or not _HAS_VARIANTS:
        return base_info
    base_key = base_info["key"]
    try:
        cfg = compose_game(base_key, *variants)
        return {
            "actions": cfg.actions, "description": cfg.description,
            "payoff_fn": cfg.payoff_fn, "default_rounds": cfg.default_rounds,
            "key": base_key, "num_players": cfg.num_players,
            "game_type": cfg.game_type,
            "opponent_actions": cfg.opponent_actions,
        }
    except (KeyError, ValueError):
        return base_info

# ---------------------------------------------------------------------------
# LLM opponent: build observation from state and call API
# ---------------------------------------------------------------------------
def _state_to_observation(state, info) -> GameObservation | None:
    """Convert Gradio state dict to a GameObservation for the LLM opponent."""
    if not _HAS_LLM_AGENT:
        return None
    history = []
    for r in state.get("history", []):
        history.append(RoundResult(
            round_number=r["round"],
            player_action=r["opponent_action"],  # flipped: LLM is the opponent
            opponent_action=r["player_action"],
            player_payoff=r.get("o_pay", 0.0),
            opponent_payoff=r.get("p_pay", 0.0),
        ))
    opp_actions = info.get("opponent_actions")
    actions = list(opp_actions) if opp_actions else info["actions"]
    return GameObservation(
        game_name=info.get("key", state["game"]),
        game_description=info.get("description", ""),
        available_actions=actions,
        current_round=state["round"],
        total_rounds=state["max_rounds"],
        history=history,
        player_score=state["o_score"],
        opponent_score=state["p_score"],
        opponent_strategy="human",
    )


def _call_anthropic(api_key: str, model: str, prompt: str) -> str:
    """Call Anthropic Messages API. Returns raw text response."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=_TEN + _TEN,
        system=_SYS_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[_ZERO].text


def _call_openai(api_key: str, model: str, prompt: str) -> str:
    """Call OpenAI Chat Completions API. Returns raw text response."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=_TEN + _TEN,
        messages=[
            {"role": "system", "content": _SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[_ZERO].message.content


def _llm_choose_action(state, info, provider, model, api_key):
    """Have the LLM choose an action. Returns (action_str, llm_response)."""
    if not _HAS_LLM_AGENT:
        return _rand.choice(info["actions"]), "(LLM agent not available)"
    obs = _state_to_observation(state, info)
    prompt = PromptBuilder.build(obs)
    try:
        if provider == "Anthropic":
            raw = _call_anthropic(api_key, model, prompt)
        elif provider == "OpenAI":
            raw = _call_openai(api_key, model, prompt)
        else:
            return _rand.choice(info["actions"]), f"Unknown provider: {provider}"
    except Exception as exc:
        return _rand.choice(info["actions"]), f"API error: {exc}"
    opp_actions = info.get("opponent_actions")
    act_list = list(opp_actions) if opp_actions else info["actions"]
    action = parse_action(raw, act_list)
    return action, raw.strip()

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _blank(gname, sname, variants=None):
    info = _get_game_info(gname, variants) or {}
    np = info.get("num_players", _TWO)
    return {"game": gname, "strategy": sname, "history": [],
            "p_score": _ZERO, "o_score": _ZERO, "round": _ZERO,
            "max_rounds": info.get("default_rounds", DEFAULT_NUM_ROUNDS),
            "done": False, "num_players": np,
            "scores": [_ZERO] * np,
            "nplayer_env": None,
            "variants": list(variants or []),
            "llm_log": []}

def _render(st):
    np = st.get("num_players", _TWO)
    is_mp = np > _TWO
    vlist = st.get("variants", [])
    vtag = f"  |  **Variants:** {', '.join(vlist)}" if vlist else ""

    lines = [f"**Game:** {st['game']}  |  **Players:** {np}  |  **Opponent:** {st['strategy']}{vtag}",
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

    # LLM reasoning log
    llm_log = st.get("llm_log", [])
    if llm_log:
        lines.append("\n### LLM Opponent Responses")
        for entry in llm_log:
            lines.append(f"- **Round {entry['round']}**: `{entry['raw']}`")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def play_round(action_str, state, provider, model, api_key):
    if state is None or state["done"]:
        return state, "Reset the game to play again.", gr.update(), gr.update()

    info = _get_game_info(state["game"], state.get("variants"))
    np = state.get("num_players", _TWO)
    is_mp = np > _TWO
    is_llm = state.get("strategy") == _LLM_OPPONENT_LABEL

    if is_mp and _HAS_NPLAYER_ENV:
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
    elif is_llm:
        # LLM opponent
        if not api_key or not api_key.strip():
            return state, "Enter your API key to play against an LLM.", gr.update(), gr.update()
        opp, raw_response = _llm_choose_action(state, info, provider, model, api_key.strip())
        p_pay, o_pay = info["payoff_fn"](action_str, opp)
        state["round"] += _ONE
        state["p_score"] += p_pay
        state["o_score"] += o_pay
        state["history"].append({"round": state["round"], "player_action": action_str,
                                 "opponent_action": opp, "p_pay": p_pay, "o_pay": o_pay})
        state.setdefault("llm_log", []).append({"round": state["round"], "raw": raw_response})
        if state["round"] >= state["max_rounds"]:
            state["done"] = True
        acts = info["actions"]
        return (state, _render(state), info["description"],
                gr.update(choices=acts, value=acts[_ZERO]))
    else:
        opp_actions = info.get("opponent_actions")
        opp_act_list = list(opp_actions) if opp_actions else info["actions"]
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

def reset_game(gname, sname, variants=None):
    vlist = list(variants or [])
    info = _get_game_info(gname, vlist)
    np = info.get("num_players", _TWO)
    is_mp = np > _TWO

    st = _blank(gname, sname, vlist)

    if is_mp and _HAS_NPLAYER_ENV:
        nenv = NPlayerEnvironment()
        game_key = _GAME_INFO.get(gname, {}).get("key", "")
        strat_list = [sname] * (np - _ONE)
        nenv.reset(game_key, opponent_strategies=strat_list)
        st["nplayer_env"] = nenv

    acts = info["actions"]
    return (st, _render(st), info["description"],
            gr.update(choices=acts, value=acts[_ZERO]))

def on_game_change(gname, sname, variants=None):
    return reset_game(gname, sname, variants)

def on_category_change(tag, mp_filter):
    names = _filter_game_names(tag)
    names = _filter_by_mp(mp_filter, names)
    if not names:
        names = sorted(_GAME_INFO.keys())
    return gr.update(choices=names, value=names[_ZERO])

def on_mp_filter_change(mp_filter, tag):
    return on_category_change(tag, mp_filter)

def on_game_select(gname):
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    is_mp = np > _TWO
    if is_mp and _HAS_NPLAYER_ENV:
        strat_names = _NPLAYER_STRAT_NAMES
    else:
        strat_names = _strategies_for_game(gname) + [_LLM_OPPONENT_LABEL]
    player_label = f"Players: {np}" if is_mp else "2-Player"
    return (gr.update(choices=strat_names, value=strat_names[_ZERO]),
            gr.update(value=player_label))

def on_strategy_change(sname):
    """Show/hide LLM config based on strategy selection."""
    is_llm = sname == _LLM_OPPONENT_LABEL
    return (gr.update(visible=is_llm),  # llm_config_row
            gr.update(visible=is_llm))  # api_key_row

def on_provider_change(provider):
    """Update model choices when provider changes."""
    models = _LLM_MODELS.get(provider, [])
    return gr.update(choices=models, value=models[_ZERO] if models else "")

# ---------------------------------------------------------------------------
# Variant filter
# ---------------------------------------------------------------------------
_2P_ONLY_VARIANTS = {"noisy_actions", "noisy_payoffs", "self_play", "cross_model"}
_HUMAN_VARIANTS = [v for v in _VARIANT_NAMES if v not in ("self_play", "cross_model")]

def on_game_select_variant(gname):
    info = _GAME_INFO.get(gname, {})
    np = info.get("num_players", _TWO)
    is_mp = np > _TWO
    if is_mp or not _HAS_VARIANTS:
        available = []
    else:
        available = [v for v in _HUMAN_VARIANTS
                     if v not in _2P_ONLY_VARIANTS or not is_mp]
    return gr.update(choices=available, value=[])

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
_GAME_NAMES = sorted(_GAME_INFO.keys())
_INIT_STRAT_NAMES = (_strategies_for_game(_GAME_NAMES[_ZERO]) + [_LLM_OPPONENT_LABEL]) if _GAME_NAMES else ["random"]
_INIT_GAME = _GAME_NAMES[_ZERO] if _GAME_NAMES else "Prisoner's Dilemma"
_INIT_STRAT = _INIT_STRAT_NAMES[_ZERO]
_INIT_ACTS = _GAME_INFO[_INIT_GAME]["actions"] if _INIT_GAME in _GAME_INFO else ["cooperate", "defect"]

_TAG_CHOICES = [_ALL_FILTER]
for _dn, _dt in sorted(_CATEGORY_DIMS.items()):
    _TAG_CHOICES.extend(_dt)

_init_np = _GAME_INFO.get(_INIT_GAME, {}).get("num_players", _TWO)
_init_player_label = f"Players: {_init_np}" if _init_np > _TWO else "2-Player"


def _build_reference_md():
    if not _HAS_REGISTRY:
        return "# Game Theory Reference\n\nFull registry not available."
    sections = []
    for dim_name, tags in sorted(_CATEGORY_DIMS.items()):
        sec = [f"## {dim_name.replace('_', ' ').title()}"]
        for tag in tags:
            keys = get_games_by_tag(tag)
            names = sorted(_KEY_TO_NAME[k] for k in keys if k in _KEY_TO_NAME)
            if names:
                sec.append(f"**{tag}** ({len(names)}): {', '.join(names)}")
        sections.append("\n\n".join(sec))
    np_games = [(gn, gi) for gn, gi in _GAME_INFO.items() if gi.get("num_players", _TWO) > _TWO]
    if np_games:
        np_lines = ["## Multiplayer Games"]
        np_lines.append("| Game | Players | Actions | Rounds |")
        np_lines.append("|------|---------|---------|--------|")
        for gn, gi in sorted(np_games):
            acts = gi["actions"]
            act_str = ", ".join(acts[:_FOUR])
            if len(acts) > _FOUR:
                act_str += f" ... ({len(acts)} total)"
            np_lines.append(f"| {gn} | {gi['num_players']} | {act_str} | {gi['default_rounds']} |")
        sections.append("\n".join(np_lines))
    if _HUMAN_VARIANTS:
        vlines = ["## Composable Variants"]
        for vname in _HUMAN_VARIANTS:
            vlines.append(f"- **{vname}**")
        sections.append("\n".join(vlines))
    slines = ["## Opponent Strategies"]
    slines.append(f"**Generic** ({len(_GENERIC_STRATEGIES)}): {', '.join(_GENERIC_STRATEGIES)}")
    for gt, strats in sorted(_GAME_TYPE_STRATEGIES.items()):
        slines.append(f"**{gt}**: {', '.join(strats)}")
    if _HAS_NPLAYER_ENV:
        slines.append(f"**N-player**: {', '.join(_NPLAYER_STRAT_NAMES)}")
    slines.append(f"\n**LLM Opponents**: Select '{_LLM_OPPONENT_LABEL}' as strategy, "
                  "provide your Anthropic or OpenAI API key, and play against Claude or GPT models.")
    sections.append("\n\n".join(slines))
    total = len(_GAME_INFO)
    np_count = len(np_games)
    return (f"# Game Theory Reference\n\n**{total} games** ({total - np_count} two-player, "
            f"{np_count} multiplayer)\n\n" + "\n\n---\n\n".join(sections))


with gr.Blocks(title="Kant Demo") as demo:
    gr.Markdown("# Kant -- Interactive Game Theory Demo")
    with gr.Tabs():
        with gr.TabItem("Human Play"):
            with gr.Row():
                cat_dd = gr.Dropdown(_TAG_CHOICES, value=_ALL_FILTER, label="Filter by Category")
                mp_dd = gr.Dropdown(_MP_FILTERS, value=_MP_FILTER_ALL, label="Player Count")
                game_dd = gr.Dropdown(_GAME_NAMES, value=_INIT_GAME, label="Game")
            with gr.Row():
                strat_dd = gr.Dropdown(_INIT_STRAT_NAMES, value=_INIT_STRAT, label="Opponent Strategy")
                player_info = gr.Textbox(value=_init_player_label, label="Mode", interactive=False)
                reset_btn = gr.Button("Reset / New Game")

            # LLM config (hidden by default, shown when strategy = LLM)
            with gr.Row(visible=False) as llm_config_row:
                llm_provider = gr.Dropdown(
                    _LLM_PROVIDERS, value=_LLM_PROVIDERS[_ZERO],
                    label="LLM Provider",
                )
                llm_model = gr.Dropdown(
                    _LLM_MODELS[_LLM_PROVIDERS[_ZERO]],
                    value=_LLM_MODELS[_LLM_PROVIDERS[_ZERO]][_ZERO],
                    label="Model",
                )
            with gr.Row(visible=False) as api_key_row:
                api_key_input = gr.Textbox(
                    label="API Key", type="password",
                    placeholder="Enter your Anthropic or OpenAI API key",
                )

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

            # Wiring
            _reset_out = [state_var, history_md, game_desc, action_dd]
            cat_dd.change(on_category_change, inputs=[cat_dd, mp_dd], outputs=[game_dd])
            mp_dd.change(on_mp_filter_change, inputs=[mp_dd, cat_dd], outputs=[game_dd])
            play_btn.click(play_round,
                           inputs=[action_dd, state_var, llm_provider, llm_model, api_key_input],
                           outputs=_reset_out)
            reset_btn.click(reset_game, inputs=[game_dd, strat_dd, variant_cb],
                            outputs=_reset_out)
            game_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                           outputs=_reset_out)
            game_dd.change(on_game_select, inputs=[game_dd],
                           outputs=[strat_dd, player_info])
            game_dd.change(on_game_select_variant, inputs=[game_dd],
                           outputs=[variant_cb])
            strat_dd.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                            outputs=_reset_out)
            strat_dd.change(on_strategy_change, inputs=[strat_dd],
                            outputs=[llm_config_row, api_key_row])
            llm_provider.change(on_provider_change, inputs=[llm_provider],
                                outputs=[llm_model])
            variant_cb.change(on_game_change, inputs=[game_dd, strat_dd, variant_cb],
                              outputs=_reset_out)
        with gr.TabItem("Game Theory Reference"):
            gr.Markdown(value=_build_reference_md())

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
