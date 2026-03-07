"""Tests for train/splits.py -- deterministic stratified split."""

from __future__ import annotations

from common.games_meta.game_tags import GAME_TAGS
from constant_definitions.batch4.tag_constants import CATEGORIES
from constant_definitions.game_constants import EVAL_ZERO
from constant_definitions.train.split_constants import (
    MIN_EVAL_TAG_FRACTION_DENOMINATOR,
    MIN_EVAL_TAG_FRACTION_NUMERATOR,
    SPLIT_SEED,
)
from train.splits import get_train_eval_split

_DOMAIN_TAGS = CATEGORIES["domain"]
_ONE = int(bool(True))
_DIFFERENT_SEED = SPLIT_SEED + _ONE


def test_split_sizes_sum_to_total():
    """Train + eval should cover all games with no gaps."""
    train, eval_ = get_train_eval_split()
    assert len(train) + len(eval_) == len(GAME_TAGS)


def test_no_overlap():
    """Train and eval sets must be disjoint."""
    train, eval_ = get_train_eval_split()
    assert len(train & eval_) == EVAL_ZERO


def test_determinism():
    """Same seed produces the same split."""
    train_a, eval_a = get_train_eval_split(seed=SPLIT_SEED)
    train_b, eval_b = get_train_eval_split(seed=SPLIT_SEED)
    assert train_a == train_b
    assert eval_a == eval_b


def test_different_seed_different_split():
    """Different seed produces a different split."""
    _train_a, eval_a = get_train_eval_split(seed=SPLIT_SEED)
    _train_b, eval_b = get_train_eval_split(seed=_DIFFERENT_SEED)
    assert eval_a != eval_b


def test_domain_tag_coverage():
    """Every domain tag should have minimum representation in eval."""
    _train, eval_ = get_train_eval_split()
    for dtag in _DOMAIN_TAGS:
        games_with_tag = [
            g for g, tags in GAME_TAGS.items() if dtag in tags
        ]
        if not games_with_tag:
            continue
        eval_with_tag = [g for g in games_with_tag if g in eval_]
        min_required = max(
            _ONE,
            (len(games_with_tag) * MIN_EVAL_TAG_FRACTION_NUMERATOR
             + MIN_EVAL_TAG_FRACTION_DENOMINATOR - _ONE)
            // MIN_EVAL_TAG_FRACTION_DENOMINATOR,
        )
        assert len(eval_with_tag) >= min_required, (
            f"Tag {dtag}: {len(eval_with_tag)} eval games "
            f"but need >= {min_required}"
        )


def test_all_games_are_known():
    """Every game in the split should exist in GAME_TAGS."""
    train, eval_ = get_train_eval_split()
    all_games = set(GAME_TAGS.keys())
    assert train <= all_games
    assert eval_ <= all_games


def test_eval_is_nonempty():
    """Eval set should contain at least one game."""
    _train, eval_ = get_train_eval_split()
    assert len(eval_) >= _ONE
