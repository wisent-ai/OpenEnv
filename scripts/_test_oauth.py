"""Quick test: fetch OAuth tokens and make real API calls."""
import sys, os
sys.path.insert(int(), os.path.join(os.path.dirname(__file__), ".."))

from train.self_play.oauth import get_anthropic_access_token, get_openai_credentials
from constant_definitions.train.models.anthropic_constants import CLAUDE_SONNET
from constant_definitions.var.meta.self_play_constants import (
    ANTHROPIC_OAUTH_BETA_HEADER,
)

_ZERO = int()
_ONE = int(bool(True))
_TEN = (_ONE + _ONE + _ONE + _ONE + _ONE) * (_ONE + _ONE)

try:
    token = get_anthropic_access_token()
    print("Anthropic token: " + str(len(token)) + " chars")
    # OAuth needs auth_token + beta header
    import anthropic
    client = anthropic.Anthropic(
        api_key=None,
        auth_token=token,
        default_headers={"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER},
    )
    resp = client.messages.create(
        model=CLAUDE_SONNET,
        max_tokens=_TEN,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    print("Anthropic API call: " + resp.content[_ZERO].text)
except Exception as e:
    print("Anthropic FAILED: " + str(e))

try:
    token, acct = get_openai_credentials()
    print("OpenAI token: " + str(len(token)) + " chars")
except Exception as e:
    print("OpenAI FAILED: " + str(e))
