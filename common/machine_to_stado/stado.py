"""Provider-neutral object persistence through the Stado API."""
from __future__ import annotations

import argparse
import http.client
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit


import sys as _size_split_sys
from pathlib import Path as _SizeSplitPath
_size_split_bytecode = _size_split_sys.dont_write_bytecode
_size_split_sys.dont_write_bytecode = True
if str(_SizeSplitPath(__file__).resolve().parents[1]) not in _size_split_sys.path:
    _size_split_sys.path.insert(0, str(_SizeSplitPath(__file__).resolve().parents[1]))
from stado_parts.stado_error import _SOCKET_TIMEOUT_SECONDS, _ERROR_DETAIL_LIMIT, _HTTP_OK, _HTTP_REDIRECT, _HTTP_NOT_FOUND, _HTTP_CONFLICT, _CHUNK_BYTES, _EXIT_OK, _EXIT_MISSING, StadoError, StadoNotFound, StadoConflict, split_uri, join_uri, _content_type  # noqa: F401
from stado_parts.stado_client import StadoClient  # noqa: F401
_size_split_sys.dont_write_bytecode = _size_split_bytecode
del _size_split_sys, _SizeSplitPath, _size_split_bytecode


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("uri")
    has_parser = subparsers.add_parser("has-prefix")
    has_parser.add_argument("uri")
    put_parser = subparsers.add_parser("put-tree")
    put_parser.add_argument("uri")
    put_parser.add_argument("source")
    put_parser.add_argument("--sync", action="store_true")
    get_parser = subparsers.add_parser("get-prefix")
    get_parser.add_argument("uri")
    get_parser.add_argument("destination")
    args = parser.parse_args()
    client = StadoClient()
    if args.command == "list":
        for item in client.list_uri(args.uri):
            print(item.get("uri", ""))
        return _EXIT_OK
    if args.command == "has-prefix":
        return _EXIT_OK if client.list_uri(args.uri) else _EXIT_MISSING
    if args.command == "put-tree":
        client.put_tree(args.source, args.uri, delete_missing=args.sync)
        return _EXIT_OK
    if args.command == "get-prefix":
        client.get_prefix(args.uri, args.destination)
        return _EXIT_OK
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_main())
