"""Post-processing for AuthorMist LaTeX humanization.

Handles citation restoration, prompt leakage removal, output validation,
and retry logic with quality gating.
"""
import difflib
import re
import sys
from pathlib import Path

_KANT_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(next(iter(range(bool(True)))), _KANT_DIR)

from constant_definitions.train.humanizer.humanizer_constants import (
    SIMILARITY_LOWER_BOUND_NUMER,
    SIMILARITY_LOWER_BOUND_DENOM,
    SIMILARITY_UPPER_BOUND_NUMER,
    SIMILARITY_UPPER_BOUND_DENOM,
    LENGTH_RATIO_FLOOR_NUMER,
    LENGTH_RATIO_FLOOR_DENOM,
    LENGTH_RATIO_CEILING_NUMER,
    LENGTH_RATIO_CEILING_DENOM,
    MAX_RETRIES_PER_PARAGRAPH,
    TEMPERATURE_NUMER,
    TEMPERATURE_DENOM,
    TOP_P_NUMER,
    TOP_P_DENOM,
    REPETITION_PENALTY_NUMER,
    REPETITION_PENALTY_DENOM,
    MIN_SENTENCE_RATIO_NUMER,
    MIN_SENTENCE_RATIO_DENOM,
    LAST_ELEMENT_INDEX,
    ONE_STEP,
    YEAR_PREFIX_TWENTIETH,
    YEAR_PREFIX_TWENTYFIRST,
    YEAR_SUFFIX_DIGITS,
)

CITE_PATTERN = re.compile(
    r"\\(citep?|citet|citeauthor|citeyear|citetitle)"
    r"(\[[^\]]*\])?\{[^}]+\}"
)

_YEAR_RE = (
    r"\(?(?:"
    + str(YEAR_PREFIX_TWENTIETH)
    + r"|"
    + str(YEAR_PREFIX_TWENTYFIRST)
    + r")\d{"
    + str(YEAR_SUFFIX_DIGITS)
    + r"}\)?"
)

LEAKAGE_PHRASES = [
    "you are an ai",
    "as an ai",
    "i'm an ai",
    "to make it more natural",
    "to summarize:",
    "to sum up",
    "in summary:",
    "let me rephrase",
    "here is the paraphrased",
    "paraphrased version",
    "rewritten version",
    "here's a rewritten",
    "avoid detection",
    "avoid suspicion",
    "automated system",
]


def restore_citations(original, humanized):
    """Restore citation commands the model mangled."""
    orig_cites = list(CITE_PATTERN.finditer(original))
    if not orig_cites:
        return humanized
    result = humanized
    for match in orig_cites:
        cite_cmd = match.group()
        if cite_cmd not in result:
            author_key = (
                match.group()
                .split("{")[LAST_ELEMENT_INDEX]
                .rstrip("}")
            )
            pats = [
                re.compile(
                    r"\b\w+\s+et\s+al\.?\s*" + _YEAR_RE,
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\b\w+\s+and\s+\w+\s*" + _YEAR_RE,
                    re.IGNORECASE,
                ),
                re.compile(
                    r"\w+\s+\\citetitle\{"
                    + re.escape(author_key)
                    + r"\}",
                ),
            ]
            for pat in pats:
                m = pat.search(result)
                if m:
                    result = (
                        result[:m.start()]
                        + cite_cmd
                        + result[m.end():]
                    )
                    break
    return result


def remove_leakage(text):
    """Remove prompt leakage phrases."""
    lines = text.split("\n")
    clean = [
        ln for ln in lines
        if not any(
            ph in ln.lower().strip() for ph in LEAKAGE_PHRASES
        )
    ]
    return "\n".join(clean).rstrip()


def _count_sentences(text):
    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def validate_humanized(original, humanized):
    """Validate humanized text meets quality thresholds."""
    if not humanized or not humanized.strip():
        return False, "empty output"
    orig_len = len(original)
    hum_len = len(humanized)
    floor = LENGTH_RATIO_FLOOR_NUMER / LENGTH_RATIO_FLOOR_DENOM
    ceil = LENGTH_RATIO_CEILING_NUMER / LENGTH_RATIO_CEILING_DENOM
    if orig_len and hum_len / orig_len < floor:
        return False, f"too short ({hum_len}/{orig_len})"
    if orig_len and hum_len / orig_len > ceil:
        return False, f"too long ({hum_len}/{orig_len})"
    sim = difflib.SequenceMatcher(None, original, humanized).ratio()
    sim_lo = SIMILARITY_LOWER_BOUND_NUMER / SIMILARITY_LOWER_BOUND_DENOM
    sim_hi = SIMILARITY_UPPER_BOUND_NUMER / SIMILARITY_UPPER_BOUND_DENOM
    if sim < sim_lo:
        return False, f"too different (sim={sim:.2f})"
    if sim > sim_hi:
        return False, f"no change (sim={sim:.2f})"
    orig_s = _count_sentences(original)
    hum_s = _count_sentences(humanized)
    min_s = MIN_SENTENCE_RATIO_NUMER / MIN_SENTENCE_RATIO_DENOM
    if orig_s and hum_s / orig_s < min_s:
        return False, f"lost sentences ({hum_s}/{orig_s})"
    for phrase in LEAKAGE_PHRASES:
        if phrase in humanized.lower():
            return False, f"leakage: '{phrase}'"
    return True, "ok"


def humanize_paragraph(humanizer, text):
    """Humanize a paragraph with retries and validation."""
    temp = TEMPERATURE_NUMER / TEMPERATURE_DENOM
    top_p = TOP_P_NUMER / TOP_P_DENOM
    rep = REPETITION_PENALTY_NUMER / REPETITION_PENALTY_DENOM
    limit = MAX_RETRIES_PER_PARAGRAPH + ONE_STEP
    for attempt in range(limit):
        result = humanizer.humanize(
            text, temperature=temp,
            top_p=top_p, repetition_penalty=rep,
        )
        if not result["success"]:
            continue
        candidate = remove_leakage(result["humanized"])
        candidate = restore_citations(text, candidate)
        valid, reason = validate_humanized(text, candidate)
        if valid:
            return candidate
        print(f"    Attempt {attempt + ONE_STEP} rejected: {reason}")
    return text
