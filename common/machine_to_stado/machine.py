"""Trusted submitter helpers for provider-neutral Stado machine jobs."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tarfile
import tempfile
import re
from pathlib import Path, PurePosixPath
from typing import Any


class MachineError(RuntimeError):
    """A Stado machine command failed."""


def _invoke(*args: str) -> dict[str, Any]:
    result = subprocess.run(
        [os.environ.get("STADO_BIN", "stado"), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MachineError(result.stderr.strip() or "invalid Stado machine response") from exc
    if result.returncode != int("0") or not payload.get("ok"):
        error = payload.get("error") or {}
        raise MachineError(str(error.get("message") or result.stderr or "Stado machine command failed"))
    return payload["result"]


def _job(result: dict[str, Any], operation: str) -> dict[str, Any]:
    job = result.get("job")
    if not isinstance(job, dict):
        raise MachineError(f"Stado machine {operation} returned no job")
    return job

_OPENENV_NAMESPACE = "openenv"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ALLOWED_JOB_SECRETS = {"OPENENV_MODEL_ROUTER_TOKEN"}


def _require_openenv_uri(uri: str) -> None:
    from common.machine_to_stado.stado import split_uri

    namespace, _ = split_uri(uri)
    if namespace != _OPENENV_NAMESPACE:
        raise MachineError(
            f"OpenEnv machine objects must use stado://{_OPENENV_NAMESPACE}/..."
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(int("1048576")):
            digest.update(chunk)
    return digest.hexdigest()




def upload_source_tree(root: str | Path, destination_uri: str) -> str:
    """Archive a source tree, publish it create-only, and return its SHA-256."""
    _require_openenv_uri(destination_uri)
    root = Path(root)
    excluded = {
        ".git",
        "__pycache__",
        "history",
        "notebooks",
        "paper",
        "results",
        "tests",
    }
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as bundle:
            bundle.add(
                root,
                arcname=".",
                filter=lambda item: None
                if any(part in excluded for part in Path(item.name).parts)
                else item,
            )
        digest = _file_sha256(archive.name)
        from common.machine_to_stado.stado import StadoClient
        StadoClient().put_file(
            destination_uri,
            archive.name,
            content_type="application/gzip",
            if_absent=True,
        )
    return digest


def submit(
    *,
    request_id: str,
    command: str,
    input_objects: dict[str, dict[str, str]],
    output_uri: str,
    vram_gb: int = int("40"),
    gpu_type: str = "",
    max_cost_per_hour: float = float("4"),
    secret_env: dict[str, dict[str, str]] | None = None,
) -> str:
    _require_openenv_uri(output_uri)
    if not input_objects:
        raise MachineError("OpenEnv machine jobs require immutable input objects")
    normalized_inputs: dict[str, dict[str, str]] = {}
    for name, spec in input_objects.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise MachineError("invalid OpenEnv machine input object")
        if set(spec) != {"stado_uri", "relative_path", "sha256"}:
            raise MachineError(
                f"input_objects.{name} requires stado_uri, relative_path, and sha256"
            )
        uri = str(spec["stado_uri"])
        relative = PurePosixPath(str(spec["relative_path"]))
        digest = str(spec["sha256"]).lower()
        _require_openenv_uri(uri)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise MachineError(f"input_objects.{name}.relative_path is unsafe")
        if not _SHA256_RE.fullmatch(digest):
            raise MachineError(f"input_objects.{name}.sha256 is invalid")
        normalized_inputs[name] = {
            "stado_uri": uri,
            "relative_path": relative.as_posix(),
            "sha256": digest,
        }
    if secret_env and set(secret_env) - _ALLOWED_JOB_SECRETS:
        disallowed = ", ".join(sorted(set(secret_env) - _ALLOWED_JOB_SECRETS))
        raise MachineError(f"provider credentials are forbidden in jobs: {disallowed}")
    request = {
        "client_request_id": request_id,
        "command": command,
        "vram_gb": vram_gb,
        "gpu_type": gpu_type,
        "max_cost_per_hour_usd": max_cost_per_hour,
        "output_uri": output_uri,
        "input_objects": normalized_inputs,
    }
    if secret_env:
        request["secret_env"] = secret_env
    fd, temporary = tempfile.mkstemp(prefix="stado-machine-", suffix=".json")
    os.close(fd)
    path = Path(temporary)
    try:
        path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")))
        result = _invoke("machine", "submit", "--request-file", str(path))
    finally:
        path.unlink(missing_ok=True)
    job_id = _job(result, "submit").get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise MachineError("Stado machine submit returned no job id")
    return job_id


def status(job_id: str) -> dict[str, Any]:
    return _job(_invoke("machine", "status", job_id), "status")


def cancel(job_id: str) -> None:
    _invoke("machine", "cancel", job_id)
