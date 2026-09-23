"""Parts of stado.py, split by the tama size splitter; stado.py imports every name back."""

from __future__ import annotations
import mimetypes
import posixpath
from pathlib import Path
from urllib.parse import urlsplit


_SOCKET_TIMEOUT_SECONDS = int("120")
_ERROR_DETAIL_LIMIT = int("500")
_HTTP_OK = int("200")
_HTTP_REDIRECT = int("300")
_HTTP_NOT_FOUND = int("404")
_HTTP_CONFLICT = int("409")
_CHUNK_BYTES = int("1048576")
_EXIT_OK = int("0")
_EXIT_MISSING = int("1")


class StadoError(RuntimeError):
    """A Stado object request failed."""


class StadoNotFound(StadoError):
    """The requested Stado object does not exist."""


class StadoConflict(StadoError):
    """A create-only Stado object already exists."""


def split_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "stado" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"invalid Stado object URI: {uri}")
    key = parsed.path.lstrip("/")
    if any(part in {".", ".."} for part in key.split("/")):
        raise ValueError(f"unsafe Stado object URI: {uri}")
    return parsed.netloc, key


def join_uri(base: str, *parts: str) -> str:
    namespace, key = split_uri(base)
    clean = [key.rstrip("/")] if key else []
    for part in parts:
        value = str(part).strip("/")
        if value:
            clean.append(value)
    joined = posixpath.join(*clean) if clean else ""
    if any(part in {".", ".."} for part in joined.split("/")):
        raise ValueError("unsafe Stado object key")
    return f"stado://{namespace}/{joined}" if joined else f"stado://{namespace}"


def _content_type(path: Path) -> str:
    guessed, encoding = mimetypes.guess_type(path.name)
    if encoding == "gzip":
        return "application/gzip"
    return guessed or "application/octet-stream"
