"""Plot the per-step dynamics CSV produced by dynamics_over_training.py.

Reads the CSV (step, lie_rate, fooled_rate, coop_rate, mutual_coop_rate,
mutual_defect_rate, mean_payoff) and writes a single multi-panel PNG
showing each metric's trajectory across training. Headline figure for
the emergent-dynamics paper draft.

Usage:
    PYTHONPATH=. python3 scripts/free_chat/plot_dynamics.py \\
        --csv /tmp/dynamics_5k_s0.csv \\
        --out /tmp/dynamics_5k_s0.png \\
        --title "Llama-3.2-1B GRPO on free_chat_prisoners_dilemma (5000 steps)"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_csv(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    cols: dict = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            if k == "step":
                cols[k].append(int(v))
            else:
                try:
                    cols[k].append(float(v))
                except ValueError:
                    cols[k].append(float("nan"))
    return cols


def _plot(cols: dict, out: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("lie_rate", "lie_rate: message names one action; player does the other"),
        ("coop_rate", "coop_rate: fraction of rounds the player cooperated"),
        ("mutual_defect_rate", "mutual_defect_rate: both players defect"),
        ("mean_payoff", "mean_payoff: average per-round reward"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.0 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, panels):
        if col not in cols:
            ax.text(0.5, 0.5, f"missing column: {col}", ha="center")
            continue
        ax.plot(cols["step"], cols[col], linewidth=1.2)
        ax.set_ylabel(col)
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("training step")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/tmp/dynamics.csv",
                    help="Input CSV from dynamics_over_training.py")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="free_chat_prisoners_dilemma GRPO dynamics",
                    help="Figure title")
    args = p.parse_args()
    if not Path(args.csv).is_file():
        raise SystemExit(f"--csv not found: {args.csv}")
    cols = _load_csv(args.csv)
    _plot(cols, args.out, args.title)


if __name__ == "__main__":
    main()
