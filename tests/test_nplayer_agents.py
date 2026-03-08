"""Tests for train/nplayer -- N-player and coalition LLM agents."""
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

from env.nplayer.models import NPlayerAction, NPlayerObservation, NPlayerRoundResult
from env.nplayer.coalition.models import (
    CoalitionAction, CoalitionObservation, CoalitionProposal,
)
from env.nplayer.governance.models import RuntimeRules
from train.nplayer.nplayer_agent import NPlayerLLMAgent, NPlayerPromptBuilder
from train.nplayer.coalition_agent import CoalitionLLMAgent, CoalitionPromptBuilder
from constant_definitions.train.agent_constants import (
    NPLAYER_PROMPT_SECTION_PLAYERS,
    NPLAYER_PROMPT_SECTION_ALL_SCORES,
    COALITION_PROMPT_SECTION_PHASE,
    COALITION_PROMPT_SECTION_PROPOSALS,
    PROMPT_SECTION_ACTIONS,
    PROMPT_SECTION_GAME,
    PROMPT_SECTION_HISTORY,
)
from constant_definitions.game_constants import EVAL_ZERO, EVAL_ONE, EVAL_TWO

_ZERO = int()
_ONE = int(bool(True))
_THREE = EVAL_TWO + EVAL_ONE
_FOUR = _THREE + _ONE
_PLAYER_ZERO_LABEL = "Player " + str(_ZERO)
_ROUND_ONE_LABEL = "Round " + str(_ONE)
_SIX = _THREE + _THREE


def _make_nplayer_obs(
    game_name: str = "public_goods",
    available_actions: list | None = None,
    history: list | None = None,
    num_players: int = _FOUR,
) -> NPlayerObservation:
    if available_actions is None:
        available_actions = ["cooperate", "defect"]
    return NPlayerObservation(
        game_name=game_name,
        game_description="An N-player social dilemma.",
        available_actions=available_actions,
        current_round=_ONE,
        total_rounds=_SIX,
        scores=[float()] * num_players,
        num_players=num_players,
        player_index=_ZERO,
        history=history or [],
    )


def _make_coalition_obs(
    phase: str = "negotiate",
    pending: list | None = None,
) -> CoalitionObservation:
    base = _make_nplayer_obs()
    return CoalitionObservation(
        base=base, phase=phase,
        pending_proposals=pending or [],
        enforcement="cheap_talk",
        adjusted_scores=[float()] * _FOUR,
        active_players=list(range(_FOUR)),
        current_rules=RuntimeRules(),
    )


# -- NPlayerPromptBuilder tests --

def test_nplayer_prompt_contains_game():
    obs = _make_nplayer_obs()
    prompt = NPlayerPromptBuilder.build(obs)
    assert PROMPT_SECTION_GAME in prompt
    assert "public_goods" in prompt


def test_nplayer_prompt_contains_players():
    obs = _make_nplayer_obs()
    prompt = NPlayerPromptBuilder.build(obs)
    assert NPLAYER_PROMPT_SECTION_PLAYERS in prompt
    assert _PLAYER_ZERO_LABEL in prompt


def test_nplayer_prompt_contains_scores():
    obs = _make_nplayer_obs()
    prompt = NPlayerPromptBuilder.build(obs)
    assert NPLAYER_PROMPT_SECTION_ALL_SCORES in prompt


def test_nplayer_prompt_contains_actions():
    obs = _make_nplayer_obs(available_actions=["cooperate", "defect"])
    prompt = NPlayerPromptBuilder.build(obs)
    assert PROMPT_SECTION_ACTIONS in prompt
    assert "cooperate" in prompt
    assert "defect" in prompt


def test_nplayer_prompt_includes_history():
    rnd = NPlayerRoundResult(
        round_number=_ONE,
        actions=["cooperate", "defect", "cooperate", "defect"],
        payoffs=[float(_THREE), float(_FOUR), float(_THREE), float(_FOUR)],
    )
    obs = _make_nplayer_obs(history=[rnd])
    prompt = NPlayerPromptBuilder.build(obs)
    assert PROMPT_SECTION_HISTORY in prompt
    assert _ROUND_ONE_LABEL in prompt


# -- NPlayerLLMAgent tests --

def test_nplayer_agent_callable():
    def mock_gen(prompt: str) -> str:
        return "cooperate"
    agent = NPlayerLLMAgent(generate_fn=mock_gen)
    obs = _make_nplayer_obs()
    action = agent(obs)
    assert isinstance(action, NPlayerAction)
    assert action.action == "cooperate"


def test_nplayer_agent_stores_last_prompt():
    def mock_gen(prompt: str) -> str:
        return "defect"
    agent = NPlayerLLMAgent(generate_fn=mock_gen)
    obs = _make_nplayer_obs()
    agent(obs)
    assert len(agent.last_prompt) > EVAL_ZERO
    assert agent.last_completion == "defect"


# -- CoalitionPromptBuilder tests --

def test_coalition_negotiate_prompt():
    prop = CoalitionProposal(
        proposer=_ONE, members=[_ZERO, _ONE],
        agreed_action="cooperate",
    )
    obs = _make_coalition_obs(pending=[prop])
    prompt = CoalitionPromptBuilder.build_negotiate(obs)
    assert COALITION_PROMPT_SECTION_PHASE in prompt
    assert COALITION_PROMPT_SECTION_PROPOSALS in prompt
    assert "cooperate" in prompt


def test_coalition_action_prompt():
    obs = _make_coalition_obs(phase="action")
    prompt = CoalitionPromptBuilder.build_action(obs)
    assert COALITION_PROMPT_SECTION_PHASE in prompt
    assert PROMPT_SECTION_ACTIONS in prompt


# -- CoalitionLLMAgent tests --

def test_coalition_negotiate_with_json():
    resp_json = '{"proposals": [], "responses": {"' + str(_ZERO) + '": true}}'
    def mock_gen(prompt: str) -> str:
        return resp_json
    prop = CoalitionProposal(
        proposer=_ONE, members=[_ZERO, _ONE],
        agreed_action="cooperate",
    )
    obs = _make_coalition_obs(pending=[prop])
    agent = CoalitionLLMAgent(generate_fn=mock_gen)
    result = agent.negotiate(obs)
    assert isinstance(result, CoalitionAction)
    assert len(result.responses) == _ONE
    assert result.responses[_ZERO].accepted is True


def test_coalition_negotiate_accepts_all_on_bad_json():
    def mock_gen(prompt: str) -> str:
        return "not valid json"
    prop = CoalitionProposal(
        proposer=_ONE, members=[_ZERO, _ONE],
        agreed_action="cooperate",
    )
    obs = _make_coalition_obs(pending=[prop])
    agent = CoalitionLLMAgent(generate_fn=mock_gen)
    result = agent.negotiate(obs)
    assert len(result.responses) == _ONE
    assert result.responses[_ZERO].accepted is True


def test_coalition_act():
    def mock_gen(prompt: str) -> str:
        return "defect"
    obs = _make_coalition_obs(phase="action")
    agent = CoalitionLLMAgent(generate_fn=mock_gen)
    action = agent.act(obs)
    assert isinstance(action, NPlayerAction)
    assert action.action == "defect"
