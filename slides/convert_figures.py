"""Convert TikZ PDF figures to SVG for Slidev embedding."""
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
OUT_DIR = Path(__file__).resolve().parent / "figures"
FIGURES = [
    "payoff_matrices",
    "architecture",
    "governance_flow",
    "training_pipeline",
    "tournament_heatmap",
]

PREAMBLE = (
    r"\documentclass[tikz,border=2pt]{standalone}" "\n"
    r"\usepackage{amsmath,amssymb}" "\n"
    r"\usepackage{tikz}" "\n"
    r"\usetikzlibrary{positioning,arrows.meta,shapes.geometric,calc,backgrounds}" "\n"
    r"\usepackage{xcolor}" "\n"
    r"\begin{document}" "\n"
)

POSTAMBLE = "\n" r"\end{document}" "\n"


def build_standalone(fig_name: str, tmp_dir: Path) -> Path:
    src = PAPER_DIR / "figures" / f"{fig_name}.tex"
    tex_path = tmp_dir / f"{fig_name}.tex"
    tex_path.write_text(PREAMBLE + src.read_text() + POSTAMBLE)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tmp_dir,
        capture_output=True,
    )
    return tmp_dir / f"{fig_name}.pdf"


def pdf_to_svg(pdf_path: Path, svg_path: Path) -> None:
    doc = fitz.open(str(pdf_path))
    first_page = next(iter(doc))
    svg_text = first_page.get_svg_image()
    svg_path.write_text(svg_text)
    doc.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for fig in FIGURES:
            pdf = build_standalone(fig, tmp_dir)
            if not pdf.exists():
                print(f"FAIL (pdflatex): {fig}", file=sys.stderr)
                continue
            svg_out = OUT_DIR / f"{fig}.svg"
            pdf_to_svg(pdf, svg_out)
            print(f"OK: {fig} -> {svg_out}")


if __name__ == "__main__":
    main()
