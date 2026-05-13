"""Deterministic stratified train/eval game split.

Side-effect: importing this module monkey-patches huggingface_hub.HfApi
and the module-level hf_hub_download / snapshot_download with the
wisent-compute fleet-wide GCS token bucket so they wait on a shared 1000
tokens / 5min budget before issuing the HTTP call. Required because the
wisent-compute agent fleet shares an outbound IP and saturates HF's
1000-req/5min API quota during concurrent extraction + training runs,
killing every AutoTokenizer.from_pretrained call inside
transformers' `_patch_mistral_regex` -> `hf_api.model_info` chain
with HTTP 429.

train.train imports this module before any AutoTokenizer.from_pretrained
call, so the patch is in place by the time the HF API call goes out.
On a host where `wisent_compute` is not installed (local dev outside
the agent fleet), the patch is silently skipped — no 429 problem to
solve there anyway.
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Set, Tuple


def _install_hf_rate_limit_token_bucket() -> None:
    """Wrap HfApi.{model_info,hf_hub_download,...} and module-level
    huggingface_hub.{hf_hub_download,snapshot_download} with
    wisent_compute's wait_for_hf_token so every HF API call goes
    through the shared GCS token bucket. Skips silently when
    wisent_compute is not installed (local dev)."""
    try:
        from huggingface_hub import HfApi
        from wisent_compute.providers.local.hf_rate import wait_for_hf_token
    except Exception:
        return
    if getattr(HfApi, "_wisent_rate_limit_installed", False):
        return
    methods_to_wrap = (
        "upload_file", "upload_folder", "list_repo_tree",
        "preupload_lfs_files", "create_commit",
        "model_info", "dataset_info", "repo_info",
        "list_repo_files", "hf_hub_download",
    )
    for _m in methods_to_wrap:
        _orig = getattr(HfApi, _m, None)
        if _orig is None:
            continue

        def _make(_o):
            def _w(self, *a, **k):
                wait_for_hf_token()
                return _o(self, *a, **k)
            return _w

        setattr(HfApi, _m, _make(_orig))
    HfApi._wisent_rate_limit_installed = True
    try:
        import huggingface_hub as _hh
        for _fn in ("hf_hub_download", "snapshot_download"):
            _orig = getattr(_hh, _fn, None)
            if _orig is None or getattr(_orig, "_wisent_rate_limit_installed", False):
                continue

            def _make_mod(_o):
                def _w(*a, **k):
                    wait_for_hf_token()
                    return _o(*a, **k)
                _w._wisent_rate_limit_installed = True
                return _w

            setattr(_hh, _fn, _make_mod(_orig))
    except Exception:
        pass


_install_hf_rate_limit_token_bucket()


def _gcs_sync(local_path: str, gcs_uri: str, log_fn=print) -> bool:
    """Sync a local file OR directory to GCS via gcloud storage cp.

    Used by the checkpoint resume path: when a Trainer save_steps fires
    on a wisent-compute agent VM that gets reaped before training
    completes (Llama-1B 5k run was reaped 3 times today), the local
    checkpoint dies with the VM. Pushing it to GCS each save means a
    fresh agent can pull the latest checkpoint on startup and resume
    from that step instead of from step 0.

    Returns True on success, False on any failure. Tries gcloud
    storage first (modern, doesn't depend on pyOpenSSL.crypto.sign),
    falls back to gsutil only when gcloud is not on PATH.
    """
    import shutil, subprocess
    cmd = None
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", "--recursive", local_path, gcs_uri]
    elif shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "cp", "-r", local_path, gcs_uri]
    if cmd is None:
        log_fn("[ckpt-sync] no gcloud/gsutil available")
        return False
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log_fn(f"[ckpt-sync] {' '.join(cmd)} failed rc={r.returncode} "
               f"stderr={(r.stderr or '')[:200]}")
        return False
    return True


def _gcs_pull(gcs_uri: str, local_dir: str, log_fn=print) -> bool:
    """Inverse of _gcs_sync: pull a checkpoint tree from GCS into a
    local directory before trainer.train(resume_from_checkpoint=...).
    """
    import os, shutil, subprocess
    os.makedirs(local_dir, exist_ok=True)
    cmd = None
    if shutil.which("gcloud"):
        cmd = ["gcloud", "storage", "cp", "--recursive", gcs_uri, local_dir]
    elif shutil.which("gsutil"):
        cmd = ["gsutil", "-m", "cp", "-r", gcs_uri, local_dir]
    if cmd is None:
        return False
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log_fn(f"[ckpt-pull] {' '.join(cmd)} rc={r.returncode} "
               f"stderr={(r.stderr or '')[:200]}")
        return False
    return True


from common.games_meta.game_tags import GAME_TAGS
from constant_definitions.batch4.tag_constants import CATEGORIES
from constant_definitions.game_constants import EVAL_ZERO, EVAL_ONE
from constant_definitions.train.split_constants import (
    MIN_EVAL_TAG_FRACTION_DENOMINATOR,
    MIN_EVAL_TAG_FRACTION_NUMERATOR,
    SPLIT_SEED,
    TRAIN_FRACTION_DENOMINATOR,
    TRAIN_FRACTION_NUMERATOR,
)

# Domain tags are used for stratification
_DOMAIN_TAGS: List[str] = CATEGORIES["domain"]


def get_train_eval_split(
    seed: int = SPLIT_SEED,
) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Return (train_games, eval_games) as frozen sets of game keys.

    The split is deterministic for a given seed and stratified so that
    every domain tag has at least ``MIN_EVAL_TAG_FRACTION`` representation
    in the eval set.
    """
    all_games = sorted(GAME_TAGS.keys())
    rng = random.Random(seed)

    # Build domain -> games index
    domain_to_games: Dict[str, List[str]] = {tag: [] for tag in _DOMAIN_TAGS}
    for game_key in all_games:
        tags = GAME_TAGS[game_key]
        for dtag in _DOMAIN_TAGS:
            if dtag in tags:
                domain_to_games[dtag].append(game_key)

    # Guarantee minimum eval representation per domain
    eval_set: Set[str] = set()
    for dtag in _DOMAIN_TAGS:
        games_with_tag = domain_to_games[dtag]
        if not games_with_tag:
            continue
        min_eval = _min_eval_count(len(games_with_tag))
        already_in_eval = [g for g in games_with_tag if g in eval_set]
        needed = min_eval - len(already_in_eval)
        if needed > EVAL_ZERO:
            candidates = [g for g in games_with_tag if g not in eval_set]
            rng.shuffle(candidates)
            for g in candidates[:needed]:
                eval_set.add(g)

    # Fill remaining eval slots up to target size
    total = len(all_games)
    target_train = (total * TRAIN_FRACTION_NUMERATOR) // TRAIN_FRACTION_DENOMINATOR
    target_eval = total - target_train
    remaining = [g for g in all_games if g not in eval_set]
    rng.shuffle(remaining)
    slots_to_fill = target_eval - len(eval_set)
    if slots_to_fill > EVAL_ZERO:
        for g in remaining[:slots_to_fill]:
            eval_set.add(g)

    train_set = frozenset(g for g in all_games if g not in eval_set)
    return train_set, frozenset(eval_set)


def _min_eval_count(tag_total: int) -> int:
    """Minimum number of games with a given tag that must be in eval."""
    _numer = tag_total * MIN_EVAL_TAG_FRACTION_NUMERATOR
    result = (_numer + MIN_EVAL_TAG_FRACTION_DENOMINATOR - EVAL_ONE) // MIN_EVAL_TAG_FRACTION_DENOMINATOR
    return max(result, EVAL_ONE)
