"""Kant Gradio Demo -- self-contained HuggingFace Spaces app."""
from __future__ import annotations
import gradio as gr

from registry import (
    _ZERO, _ONE, _TWO, _FIVE, _TEN,
    _GAME_INFO, _CATEGORY_DIMS, _ALL_FILTER,
    _HUMAN_VARIANTS, _HAS_VARIANTS,
    _strategies_for_game,
    _MP_FILTERS, _MP_FILTER_ALL,
    _HAS_LLM_AGENT,
    _LLM_PROVIDERS, _LLM_MODELS, _LLM_OPPONENT_LABEL,
)
from llm_arena import run_tournament, render_tournament
from callbacks import (
    _get_game_info, _blank, _render,
    play_round, reset_game, on_game_change,
    on_category_change, on_mp_filter_change,
    on_game_select, on_game_select_variant,
    on_strategy_change, on_provider_change,
    _build_reference_md,
)

# -- UI constants --
_GAME_NAMES = sorted(_GAME_INFO.keys())
_INIT_STRAT_NAMES = (_strategies_for_game(_GAME_NAMES[_ZERO]) + [_LLM_OPPONENT_LABEL]) if _GAME_NAMES else ["random"]
_INIT_GAME = _GAME_NAMES[_ZERO] if _GAME_NAMES else "Prisoner's Dilemma"
_INIT_STRAT = _INIT_STRAT_NAMES[_ZERO]
_INIT_ACTS = _GAME_INFO[_INIT_GAME]["actions"] if _INIT_GAME in _GAME_INFO else ["cooperate", "defect"]

_TAG_CHOICES = [_ALL_FILTER]
for _dn, _dt in sorted(_CATEGORY_DIMS.items()):
    _TAG_CHOICES.extend(_dt)

_init_np = _GAME_INFO.get(_INIT_GAME, {}).get("num_players", _TWO)
_init_player_label = f"Players: {_init_np}" if _init_np > _TWO else "Two-Player"

# -- Infinite mode preset --
_INF_GAME = "Discounted Prisoner's Dilemma"
_INF_VARIANTS = ["constitutional", "exit", "noisy_payoffs", "noisy_actions"]
_INF_ROUNDS = _TEN * _TEN * _TEN
_ALL_LLM_MODELS = []
for _mods in _LLM_MODELS.values():
    _ALL_LLM_MODELS.extend(_mods)


# -- Gradio app --
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

        if _INF_GAME in _GAME_INFO and _HAS_VARIANTS and _ALL_LLM_MODELS:
            with gr.TabItem("Infinite Mode"):
                gr.Markdown(
                    "**LLM Tournament: Constitutional Discounted PD.** "
                    "Select models, provide API keys, and watch them compete "
                    "in a round-robin. Each match uses constitutional rule "
                    "negotiation, exit option, payoff noise, and action trembles."
                )
                with gr.Row():
                    arena_anthro_key = gr.Textbox(
                        label="Anthropic API Key", type="password",
                        placeholder="sk-ant-...")
                    arena_openai_key = gr.Textbox(
                        label="OpenAI API Key", type="password",
                        placeholder="sk-...")
                arena_models = gr.CheckboxGroup(
                    _ALL_LLM_MODELS, value=_ALL_LLM_MODELS[:_TWO],
                    label="Select Models for Tournament")
                with gr.Row():
                    arena_rounds = gr.Number(
                        value=_INF_ROUNDS, precision=_ZERO,
                        label="Rounds per Match (no upper limit)")
                    arena_run = gr.Button("Run Tournament", variant="primary")
                arena_results = gr.State([])
                arena_md = gr.Markdown("Select models and click Run Tournament.")

                def _run_arena(models, anthro_key, openai_key, num_rounds):
                    results, err = run_tournament(
                        _INF_GAME, _INF_VARIANTS, int(num_rounds),
                        models, anthro_key, openai_key)
                    if err:
                        return [], err
                    return results, render_tournament(results)

                arena_run.click(
                    _run_arena,
                    inputs=[arena_models, arena_anthro_key,
                            arena_openai_key, arena_rounds],
                    outputs=[arena_results, arena_md])

        with gr.TabItem("Game Theory Reference"):
            gr.Markdown(value=_build_reference_md())

demo.launch(server_name="0.0.0.0", server_port=7860)
