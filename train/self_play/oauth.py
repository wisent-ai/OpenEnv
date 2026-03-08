"""OAuth token management for Anthropic and OpenAI self-play integration."""

from __future__ import annotations

import base64
import json
import os
from typing import Optional, Tuple

import httpx

from constant_definitions.var.meta.self_play_constants import (
    ANTHROPIC_OAUTH_TOKEN_URL,
    ANTHROPIC_OAUTH_CLIENT_ID,
    OPENAI_OAUTH_TOKEN_URL,
    OPENAI_OAUTH_CLIENT_ID,
    SUPABASE_OAUTH_TABLE,
    SUPABASE_PROVIDER_ANTHROPIC,
    SUPABASE_PROVIDER_OPENAI,
)

_ZERO = int()
_ONE = int(bool(True))
_CONTENT_TYPE_FORM = "application/x-www-form-urlencoded"


def _read_env_file() -> dict[str, str]:
    """Read Supabase credentials from env vars or content-platform .env.local."""
    # Check environment variables first (for HF Spaces / Docker)
    sb_url = os.environ.get("SUPABASE_URL", "") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if sb_url and sb_key:
        return {"NEXT_PUBLIC_SUPABASE_URL": sb_url, "SUPABASE_SERVICE_ROLE_KEY": sb_key}
    # Fall back to local .env.local file
    env_path = os.path.join(
        os.path.expanduser("~"),
        "Documents", "CodingProjects", "Wisent",
        "content-platform", ".env.local",
    )
    env_vars: dict[str, str] = {}
    with open(env_path) as fh:
        for line in fh:
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", _ONE)
                env_vars[key] = (
                    val.strip().strip('"').replace("\\n", "").strip()
                )
    return env_vars


def _supabase_headers(service_key: str) -> dict[str, str]:
    """Return Supabase REST API headers."""
    return {
        "apikey": service_key,
        "Authorization": "Bearer " + service_key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def fetch_refresh_token(
    provider: str,
    supabase_url: str = "",
    service_key: str = "",
) -> Tuple[str, str]:
    """Fetch the first refresh token for *provider* from Supabase.

    Returns (credential_id, refresh_token).
    """
    if not supabase_url or not service_key:
        env = _read_env_file()
        supabase_url = supabase_url or env["NEXT_PUBLIC_SUPABASE_URL"]
        service_key = service_key or env["SUPABASE_SERVICE_ROLE_KEY"]
    resp = httpx.get(
        supabase_url + "/rest/v" + str(_ONE) + "/" + SUPABASE_OAUTH_TABLE,
        params={"provider": "eq." + provider, "select": "*"},
        headers=_supabase_headers(service_key),
    )
    rows = resp.json()
    if not rows:
        raise RuntimeError(f"No {provider} credentials in Supabase")
    row = rows[_ZERO]
    return row["id"], row["refresh_token"]


def save_refresh_token(
    credential_id: str,
    new_refresh_token: str,
    supabase_url: str = "",
    service_key: str = "",
) -> None:
    """Save a rotated refresh token back to Supabase."""
    if not supabase_url or not service_key:
        env = _read_env_file()
        supabase_url = supabase_url or env["NEXT_PUBLIC_SUPABASE_URL"]
        service_key = service_key or env["SUPABASE_SERVICE_ROLE_KEY"]
    body: dict[str, str] = {"refresh_token": new_refresh_token}
    httpx.patch(
        supabase_url + "/rest/v" + str(_ONE) + "/" + SUPABASE_OAUTH_TABLE,
        params={"id": "eq." + credential_id},
        json=body,
        headers=_supabase_headers(service_key),
    )


def exchange_anthropic(
    refresh_token: str,
) -> Tuple[str, str]:
    """Exchange Anthropic refresh token. Returns (access, new_refresh)."""
    resp = httpx.post(
        ANTHROPIC_OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        },
        headers={"Content-Type": _CONTENT_TYPE_FORM},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"], data.get("refresh_token", "")


def exchange_openai(
    refresh_token: str,
) -> Tuple[str, str, str]:
    """Exchange OpenAI refresh token. Returns (access, new_refresh, account_id)."""
    resp = httpx.post(
        OPENAI_OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_OAUTH_CLIENT_ID,
        },
        headers={"Content-Type": _CONTENT_TYPE_FORM},
    )
    resp.raise_for_status()
    data = resp.json()
    access = data["access_token"]
    new_rt = data.get("refresh_token", "")
    account_id = _extract_account_id(data.get("id_token", ""))
    return access, new_rt, account_id


def _extract_account_id(id_token: str) -> str:
    """Extract chatgpt_account_id from an OpenAI id_token JWT."""
    if not id_token:
        return ""
    parts = id_token.split(".")
    if len(parts) < _ONE + _ONE:
        return ""
    payload = parts[_ONE]
    # Pad base64
    padding = (_ONE + _ONE + _ONE + _ONE) - len(payload) % (
        _ONE + _ONE + _ONE + _ONE
    )
    if padding < (_ONE + _ONE + _ONE + _ONE):
        payload += "=" * padding
    decoded = json.loads(base64.urlsafe_b64decode(payload))
    claims = decoded.get("https://api.openai.com/auth", {})
    return claims.get("chatgpt_account_id", "")


def get_anthropic_access_token() -> str:
    """Full flow: try all Supabase credentials until one works."""
    env = _read_env_file()
    sb_url = env["NEXT_PUBLIC_SUPABASE_URL"]
    sb_key = env["SUPABASE_SERVICE_ROLE_KEY"]
    resp = httpx.get(
        sb_url + "/rest/v" + str(_ONE) + "/" + SUPABASE_OAUTH_TABLE,
        params={"provider": "eq." + SUPABASE_PROVIDER_ANTHROPIC, "select": "*"},
        headers=_supabase_headers(sb_key),
    )
    rows = resp.json()
    last_err: Exception = RuntimeError("No credentials found")
    for row in rows:
        cred_id, rt = row["id"], row["refresh_token"]
        try:
            access, new_rt = exchange_anthropic(rt)
            if new_rt:
                save_refresh_token(cred_id, new_rt, sb_url, sb_key)
            return access
        except Exception as exc:
            last_err = exc
    raise last_err


def get_openai_credentials() -> Tuple[str, str]:
    """Full flow: returns (access_token, account_id)."""
    cred_id, rt = fetch_refresh_token(SUPABASE_PROVIDER_OPENAI)
    access, new_rt, account_id = exchange_openai(rt)
    if new_rt:
        save_refresh_token(cred_id, new_rt)
    return access, account_id
