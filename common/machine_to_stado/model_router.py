"""Server-side OpenAI-compatible calls through the Stado model router."""
from __future__ import annotations

import ipaddress
import json
import os
import urllib.request
import urllib.parse
from typing import Any


def _validated_router_url(raw_url: str) -> str:
    parsed = urllib.parse.urlsplit(raw_url)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("STADO_MODEL_ROUTER_URL is invalid")
    loopback = parsed.hostname == "localhost" or parsed.hostname.endswith(
        ".localhost"
    )
    if not loopback:
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and loopback
    ):
        raise RuntimeError(
            "STADO_MODEL_ROUTER_URL must use HTTPS or HTTP on loopback"
        )
    return raw_url.rstrip("/")


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
) -> str:
    raw_url = os.environ.get("STADO_MODEL_ROUTER_URL", "").strip()
    token = os.environ.get("OPENENV_MODEL_ROUTER_TOKEN", "").strip()
    if not raw_url:
        raise RuntimeError("STADO_MODEL_ROUTER_URL is required for API models")
    base_url = _validated_router_url(raw_url)
    if not token:
        raise RuntimeError("OPENENV_MODEL_ROUTER_TOKEN is required for API models")
    body: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int("120")) as response:
        payload = json.load(response)
    try:
        content = payload["choices"][int("0")]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Stado model router returned no completion") from exc
    if not isinstance(content, str):
        raise RuntimeError("Stado model router returned a non-text completion")
    return content
