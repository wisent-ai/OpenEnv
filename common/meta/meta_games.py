"""Pre-registered meta-gaming rule-proposal games for KantBench.

Registers symmetric meta-games (rule_proposal, rule_signal) for the
three core matrix games.  Constitutional and proposer-responder are
composed per-episode via ``compose_game()`` because constitutional uses
mutable closure state that must be fresh per episode.
"""
from __future__ import annotations

from dataclasses import replace

from common.games import GAMES
from common.meta.variants_meta import apply_rule_proposal, apply_rule_signal

_BASE_KEYS = ("prisoners_dilemma", "stag_hunt", "hawk_dove")

_FRIENDLY_NAMES = {
    "prisoners_dilemma": "Prisoner's Dilemma",
    "stag_hunt": "Stag Hunt",
    "hawk_dove": "Hawk-Dove",
}

META_GAMES: dict = {}

for _key in _BASE_KEYS:
    _base = GAMES[_key]
    _fname = _FRIENDLY_NAMES[_key]

    _rp = apply_rule_proposal(_base, base_key=_key)
    _rp = replace(
        _rp,
        name=f"Rule Proposal {_fname}",
        description=(
            f"{_fname} with simultaneous binding rule proposals. "
            "Both players propose a rule and choose an action. "
            "If proposals match the agreed rule modifies payoffs."
        ),
    )
    META_GAMES[f"rule_proposal_{_key}"] = _rp

    _rs = apply_rule_signal(_base, base_key=_key)
    _rs = replace(
        _rs,
        name=f"Rule Signal {_fname}",
        description=(
            f"{_fname} with simultaneous non-binding rule signals. "
            "Both players signal a preferred rule and choose an action. "
            "Signals are visible but never enforced."
        ),
    )
    META_GAMES[f"rule_signal_{_key}"] = _rs

GAMES.update(META_GAMES)
