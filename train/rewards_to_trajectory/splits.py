"""Deterministic stratified train/eval split and local checkpoint helpers."""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Set, Tuple



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


def _ckpt_complete(ckpt_dir):
    """True iff ckpt_dir has every weight file it claims.

    Sharded: every shard named in model.safetensors.index.json's
    weight_map must exist on disk. Non-sharded: a single
    model.safetensors or pytorch_model.bin is present.
    """
    import json as _json
    import os as _os
    idx = _os.path.join(ckpt_dir, "model.safetensors.index.json")
    if _os.path.isfile(idx):
        try:
            with open(idx) as _f:
                wmap = _json.load(_f).get("weight_map", {})
        except (ValueError, OSError):
            return False
        shards = set(wmap.values())
        if not shards:
            return False
        return all(
            _os.path.isfile(_os.path.join(ckpt_dir, s)) for s in shards
        )
    return (
        _os.path.isfile(_os.path.join(ckpt_dir, "model.safetensors"))
        or _os.path.isfile(_os.path.join(ckpt_dir, "pytorch_model.bin"))
    )


def resolve_resume_checkpoint(requested, output_dir):
    """Map --resume-from-checkpoint to a concrete resume target.

    requested == "latest": return the path of the newest checkpoint-N
    directory that passes _ckpt_complete, skipping incomplete ones
    (an interrupted Stado pull leaves the index + trainer_state but not
    the multi-GB safetensors shards, which made HF Trainer crash in
    load_sharded_checkpoint and hard-FAIL the run — Qwen3 724084db at
    step 1000 on 2026-05-15). Return None if none are complete. Any
    other truthy value is returned unchanged; falsy -> None.
    """
    import glob as _glob
    import os as _os
    if requested != "latest":
        return requested or None

    def _step(p):
        try:
            return int(_os.path.basename(p).split("-")[-1])
        except ValueError:
            return -1

    dirs = sorted(
        _glob.glob(_os.path.join(output_dir, "checkpoint-*")),
        key=_step, reverse=True,
    )
    for d in dirs:
        if _ckpt_complete(d):
            print(f"Resuming from validated checkpoint: {d}")
            return d
        print(
            f"Skipping incomplete checkpoint {d} "
            f"(missing safetensors shard(s) — interrupted Stado pull)"
        )
    print("No complete checkpoint found, starting fresh.")
    return None
