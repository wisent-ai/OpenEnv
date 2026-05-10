"""End-to-end sanity for the dynamics-over-training analyzer.

Generates a synthetic 3-step trajectory log where:
  step 1: player always says "cooperate" but always defects (100% lie rate)
  step 2: player message mentions both actions (ambiguous; not lying or honest)
  step 3: player says "defect" and defects (100% honest)

Then runs scripts/free_chat/dynamics_over_training.py and asserts the
output CSV reports lie_rate=1.0 at step 1, lie_rate=0.0 at step 3.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile

import common.games_info.communication  # noqa: F401  registers free_chat_*


def _make_step(step: int, player_msg: str, player_act: str) -> dict:
    return {
        "step": step,
        "completion_idx": 0,
        "game": "free_chat_prisoners_dilemma",
        "strategy": "always_defect",
        "rounds": 5,
        "player_score": 0.0,
        "trajectory": [
            {
                "round": r,
                "player_message": player_msg,
                "opponent_message": "",
                "player_action": player_act,
                "opponent_action": "defect",
                "player_payoff": 1.0 if player_act == "defect" else 0.0,
            }
            for r in range(1, 6)
        ],
    }


def main() -> None:
    log_path = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ).name
    csv_path = log_path.replace(".jsonl", ".csv")

    with open(log_path, "w") as fh:
        fh.write(json.dumps(_make_step(1, "I will cooperate, trust me.", "defect")) + "\n")
        fh.write(json.dumps(_make_step(2, "cooperate or defect, I do not know.", "defect")) + "\n")
        fh.write(json.dumps(_make_step(3, "I will defect.", "defect")) + "\n")

    # Run the analyzer
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(
        [
            "python3",
            "scripts/free_chat/dynamics_over_training.py",
            "--trajectory-log", log_path,
            "--game", "free_chat_prisoners_dilemma",
            "--out", csv_path,
        ],
        env=env, capture_output=True, text=True,
    )
    print("--- analyzer stdout ---")
    print(res.stdout)
    print("--- analyzer stderr ---")
    print(res.stderr)
    assert res.returncode == 0, "analyzer failed"

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 3, f"expected 3 steps, got {len(rows)}"
    by_step = {int(r["step"]): r for r in rows}

    # step 1: 5 rounds, all "say cooperate / do defect" -> lie_rate=1.0
    assert float(by_step[1]["lie_rate"]) == 1.0, by_step[1]
    # step 2: ambiguous message (mentions both actions) -> 0 honest, 0 lies -> lie_rate=0
    assert float(by_step[2]["lie_rate"]) == 0.0, by_step[2]
    # step 3: 5 rounds say defect, do defect -> lie_rate=0.0
    assert float(by_step[3]["lie_rate"]) == 0.0, by_step[3]
    # mean_payoff is 1.0 every step (always defect against always_defect)
    for s in (1, 2, 3):
        assert abs(float(by_step[s]["mean_payoff"]) - 1.0) < 1e-9, by_step[s]

    print()
    print("OK: dynamics analyzer reports correct per-step lie_rate / mean_payoff.")
    print(f"step 1 lie_rate={by_step[1]['lie_rate']}  step 3 lie_rate={by_step[3]['lie_rate']}")


if __name__ == "__main__":
    main()
