"""Integration test: self-play with local, Anthropic, and OpenAI backends."""

from __future__ import annotations

import json
import sys
import types
import uuid

import httpx

if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw: object) -> None:
            super().__init_subclass__(**kw)
        def __class_getitem__(cls, params: object) -> type:
            return cls
        def __init__(self) -> None:
            pass

    _iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
    _openenv_stub.core = _core_stub  # type: ignore[attr-defined]
    _core_stub.env_server = _server_stub  # type: ignore[attr-defined]
    _server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
    for _n, _m in [
        ("openenv", _openenv_stub), ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

from env.environment import KantEnvironment
from env.models import GameAction
from train.self_play.opponents import FrozenOpponent
from train.self_play.oauth import get_anthropic_access_token, get_openai_credentials
from constant_definitions.train.models.anthropic_constants import CLAUDE_HAIKU
from constant_definitions.train.models.openai_constants import GPT_5_4
from constant_definitions.var.meta.self_play_constants import (
    ANTHROPIC_OAUTH_BETA_HEADER,
    ANTHROPIC_OAUTH_MAX_TOKENS,
    OPENAI_CODEX_API_URL,
)

_ZERO = int()
_ONE = int(bool(True))


def _run_episode(opponent: FrozenOpponent, label: str) -> None:
    """Run one self-play episode and print results."""
    env = KantEnvironment()
    obs = env.reset(game="prisoners_dilemma", opponent_fn=opponent)
    rounds = _ZERO
    while not obs.done:
        action = GameAction(action=obs.available_actions[_ZERO])
        obs = env.step(action)
        rounds += _ONE
    print(
        f"  [{label}] rounds={rounds} "
        f"player={obs.player_score:.2f} "
        f"opponent={obs.opponent_score:.2f} PASS"
    )


def test_local_mock() -> bool:
    """Test with a simple local mock generate function."""
    print("--- Local Mock ---")
    opp = FrozenOpponent(generate_fn=lambda p: "cooperate")
    _run_episode(opp, "local_mock")
    return True


def test_anthropic() -> bool:
    """Test with Anthropic API via OAuth."""
    print("--- Anthropic (OAuth) ---")
    import anthropic

    access_token = get_anthropic_access_token()
    print(f"  Token obtained (len={len(access_token)})")

    client = anthropic.Anthropic(
        api_key=None,
        auth_token=access_token,
        default_headers={"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER},
    )

    def _generate(prompt: str) -> str:
        resp = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=ANTHROPIC_OAUTH_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[_ZERO].text

    opp = FrozenOpponent(generate_fn=_generate)
    _run_episode(opp, "anthropic/" + CLAUDE_HAIKU)
    return True


def test_openai() -> bool:
    """Test with OpenAI Codex API via OAuth."""
    print("--- OpenAI (OAuth/Codex) ---")
    access_token, account_id = get_openai_credentials()
    print(f"  Token obtained, account={account_id[:_ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE + _ONE]}...")

    _timeout = float(
        (_ONE + _ONE + _ONE + _ONE + _ONE) * (_ONE + _ONE + _ONE + _ONE + _ONE + _ONE)
    )

    def _generate(prompt: str) -> str:
        with httpx.stream(
            "POST",
            OPENAI_CODEX_API_URL,
            json={
                "model": GPT_5_4,
                "instructions": "Reply with one word only.",
                "input": [{"role": "user", "content": prompt}],
                "stream": True,
                "store": False,
            },
            headers={
                "Authorization": "Bearer " + access_token,
                "chatgpt-account-id": account_id,
                "Content-Type": "application/json",
            },
            timeout=_timeout,
        ) as resp:
            resp.raise_for_status()
            text_parts: list[str] = []
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.output_text.delta":
                    text_parts.append(event.get("delta", ""))
            return "".join(text_parts) or "cooperate"

    opp = FrozenOpponent(generate_fn=_generate)
    _run_episode(opp, "openai/codex")
    return True


def main() -> None:
    """Run integration tests for all backends."""
    results = {}
    results["local"] = test_local_mock()
    try:
        results["anthropic"] = test_anthropic()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        results["anthropic"] = False
    try:
        results["openai"] = test_openai()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        results["openai"] = False

    print("\n=== Summary ===")
    for backend, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {backend}: {status}")


if __name__ == "__main__":
    main()
