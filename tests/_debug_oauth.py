"""Debug: check Anthropic OAuth credentials in Supabase."""
import sys
import os

sys.path.insert(int(), os.path.join(os.path.dirname(__file__), ".."))

import httpx
from train.self_play.oauth import (
    _read_env_file,
    _supabase_headers,
    save_refresh_token,
)
from constant_definitions.var.meta.self_play_constants import (
    ANTHROPIC_OAUTH_TOKEN_URL,
    ANTHROPIC_OAUTH_CLIENT_ID,
    SUPABASE_OAUTH_TABLE,
    SUPABASE_PROVIDER_ANTHROPIC,
)

_ZERO = int()
_ONE = int(bool(True))
_TWENTY = (_ONE + _ONE + _ONE + _ONE + _ONE) * (_ONE + _ONE + _ONE + _ONE)

env = _read_env_file()
sb_url = env["NEXT_PUBLIC_SUPABASE_URL"]
sb_key = env["SUPABASE_SERVICE_ROLE_KEY"]

resp = httpx.get(
    sb_url + "/rest/v" + str(_ONE) + "/" + SUPABASE_OAUTH_TABLE,
    params={"provider": "eq." + SUPABASE_PROVIDER_ANTHROPIC, "select": "*"},
    headers=_supabase_headers(sb_key),
)
rows = resp.json()

# Try just the first one with verbose error output
row = rows[_ZERO]
rt = row["refresh_token"]
print(f"Exchanging {row['id']}...")
resp = httpx.post(
    ANTHROPIC_OAUTH_TOKEN_URL,
    data={
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
print(f"Status: {resp.status_code}")
print(f"Body: {resp.text}")
