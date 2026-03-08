#!/usr/bin/env python3
"""Batch humanize all LaTeX sections of the Kant paper.

Loads AuthorMist once, processes every prose .tex file with automated
post-processing, then overwrites originals after creating backups.

Usage:
    python humanize_paper.py
    python humanize_paper.py --dry-run
    python humanize_paper.py --file sections/background.tex
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

_KANT_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(next(iter(range(bool(True)))), _KANT_DIR)

HUMANIZER_DIR = (
    "/Users/lukaszbartoszcze/Documents/CodingProjects"
    "/Wisent/backends/research/research/humanizer"
)
sys.path.insert(next(iter(range(bool(True)))), HUMANIZER_DIR)

from constant_definitions.train.humanizer.humanizer_constants import (
    MIN_PARAGRAPH_CHARS,
    MIN_MODEL_INPUT_CHARS,
    MAX_RETRIES_PER_PARAGRAPH,
    MAX_MODEL_TOKENS,
    ONE_STEP,
)
from postprocess import humanize_paragraph

PAPER_DIR = Path(__file__).resolve().parent.parent
BACKUP_DIR = PAPER_DIR / "backups_pre_humanize"
SKIP_PATTERNS = {"figures", "main.tex", "slides.tex"}


def _find_section_files():
    """Find all .tex section files to humanize."""
    sections_dir = PAPER_DIR / "sections"
    if not sections_dir.exists():
        return []
    return [
        f for f in sorted(sections_dir.rglob("*.tex"))
        if not any(p in f.relative_to(PAPER_DIR).parts for p in SKIP_PATTERNS)
    ]


class EnhancedLaTeXHumanizer:
    """LaTeX humanizer with post-processing and validation."""

    def __init__(self, humanizer):
        self.humanizer = humanizer

    def humanize_latex(self, content):
        """Humanize prose in LaTeX with full automation."""
        math_map = {}
        protected = content

        disp = (
            r"(\$\$[\s\S]*?\$\$"
            r"|\\\[[\s\S]*?\\\]"
            r"|\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}"
            r"|\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}"
            r"|\\begin\{gather\*?\}[\s\S]*?\\end\{gather\*?\})"
        )
        for i, m in enumerate(re.finditer(disp, content)):
            ph = f"__MATHD_{i}__"
            math_map[ph] = m.group()
            protected = protected.replace(m.group(), ph, ONE_STEP)

        for i, m in enumerate(re.finditer(r"(\$[^$]+\$)", protected)):
            ph = f"__MATHI_{i}__"
            math_map[ph] = m.group()
            protected = protected.replace(m.group(), ph, ONE_STEP)

        cmd_map = {}
        cmd_re = (
            r"(\\(?:citep?|citet|citeauthor|citeyear|citetitle"
            r"|ref|label|includegraphics|textbf|textit|emph"
            r"|footnote|section|subsection|paragraph|chapter"
            r"|begin|end|item|url|href)\{[^}]*\})"
        )
        for i, m in enumerate(re.finditer(cmd_re, protected)):
            ph = f"__CMD_{i}__"
            cmd_map[ph] = m.group()
            protected = protected.replace(m.group(), ph, ONE_STEP)

        lines = protected.split("\n")
        result_lines = []
        prose_buffer = []
        thresh = MIN_PARAGRAPH_CHARS // MAX_RETRIES_PER_PARAGRAPH

        for line in lines:
            stripped = line.strip()
            is_prose = (
                stripped
                and not stripped.startswith("\\")
                and not stripped.startswith("%")
                and not any(
                    stripped.startswith(p)
                    for p in ("__MATHD", "__MATHI", "__CMD")
                )
                and len(stripped) > thresh
            )
            if is_prose:
                prose_buffer.append(line)
            else:
                if prose_buffer:
                    self._flush_prose(prose_buffer, result_lines)
                    prose_buffer = []
                result_lines.append(line)

        if prose_buffer:
            self._flush_prose(prose_buffer, result_lines)

        final = "\n".join(result_lines)
        for ph, orig in cmd_map.items():
            final = final.replace(ph, orig)
        for ph, orig in math_map.items():
            final = final.replace(ph, orig)
        return final

    def _flush_prose(self, buffer, result_lines):
        """Process accumulated prose buffer."""
        pt = "\n".join(buffer)
        if len(pt.strip()) > MIN_PARAGRAPH_CHARS:
            pre = pt[:MIN_MODEL_INPUT_CHARS].replace("\n", " ")
            print(f"  Humanizing: \"{pre}...\"")
            h = humanize_paragraph(self.humanizer, pt)
            result_lines.append(h)
        else:
            result_lines.append(pt)


def main():
    parser = argparse.ArgumentParser(
        description="Batch humanize Kant paper LaTeX sections",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file", type=str)
    parser.add_argument(
        "--device", type=str, choices=["cuda", "mps", "cpu"],
    )
    args = parser.parse_args()

    if args.file:
        target = PAPER_DIR / args.file
        if not target.exists():
            print(f"Not found: {target}")
            sys.exit(ONE_STEP)
        files = [target]
    else:
        files = _find_section_files()

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f.relative_to(PAPER_DIR)}")

    if args.dry_run:
        print("\n[DRY RUN] No changes.")
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    for f in files:
        rel = f.relative_to(PAPER_DIR)
        bk = BACKUP_DIR / rel
        bk.parent.mkdir(parents=True, exist_ok=True)
        if not bk.exists():
            shutil.copy(f, bk)
    print(f"\nBackups at {BACKUP_DIR}/")

    print("\nLoading AuthorMist model...")
    from humanizer import Humanizer
    hum = Humanizer(device=args.device, max_length=MAX_MODEL_TOKENS)
    lh = EnhancedLaTeXHumanizer(hum)

    total = len(files)
    for idx, tf in enumerate(files, ONE_STEP):
        rel = tf.relative_to(PAPER_DIR)
        print(f"\n[{idx}/{total}] {rel}")
        with open(tf, "r") as fh:
            orig = fh.read()
        result = lh.humanize_latex(orig)
        if result != orig:
            with open(tf, "w") as fh:
                fh.write(result)
            print(f"  Saved {rel}")
        else:
            print(f"  No changes for {rel}")

    print(f"\nDone. {total} files processed.")
    print(f"Backups at: {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
