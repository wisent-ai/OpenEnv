"""Deterministic stratified train/eval game split."""

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
