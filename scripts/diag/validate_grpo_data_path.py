"""Validate the GRPO reward path: TrajectoryCollector -> dataset -> reward_function.

Builds a deterministic episode (no LLM, no GPU), passes the resulting
records through ``KantGRPOTrainer.reward_function``, and prints
(action, opponent_action, payoff_fn(action, opp)[0], reward) per row.
The reward column should equal the payoff column for every row that
opponent_action is populated -- which is the post-fix behaviour. If
opponent_action is missing or unrecognised, the trainer falls back to
the uniform-opponent expected payoff (legacy proxy).

Run:

    PYTHONPATH=. python3 scripts/diag/validate_grpo_data_path.py
"""

from __future__ import annotations

import sys
import types

# Same openenv stub as the test suite so importing env.environment works
# without the upstream openenv-core package.
if "openenv" not in sys.modules:
    _openenv_stub = types.ModuleType("openenv")
    _core_stub = types.ModuleType("openenv.core")
    _server_stub = types.ModuleType("openenv.core.env_server")
    _iface_stub = types.ModuleType("openenv.core.env_server.interfaces")

    class _EnvironmentStub:
        def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)
        def __class_getitem__(cls, params): return cls
        def __init__(self): pass

    _iface_stub.Environment = _EnvironmentStub  # type: ignore[attr-defined]
    _openenv_stub.core = _core_stub  # type: ignore[attr-defined]
    _core_stub.env_server = _server_stub  # type: ignore[attr-defined]
    _server_stub.interfaces = _iface_stub  # type: ignore[attr-defined]
    for _n, _m in [
        ("openenv", _openenv_stub),
        ("openenv.core", _core_stub),
        ("openenv.core.env_server", _server_stub),
        ("openenv.core.env_server.interfaces", _iface_stub),
    ]:
        sys.modules[_n] = _m

from common.games import GAMES  # noqa: E402
from env.environment import KantEnvironment  # noqa: E402
from env.models import GameAction, GameObservation  # noqa: E402
from train.grpo.dataset import trajectories_to_dataset  # noqa: E402
from train.grpo.trainer import KantGRPOTrainer  # noqa: E402
from train.grpo.config import GRPOConfig  # noqa: E402
from train.trajectory import TrajectoryCollector  # noqa: E402


class _FirstActionAgent:
    """Deterministic agent: always picks obs.available_actions[0]."""

    def __init__(self):
        self.last_prompt = ""
        self.last_completion = ""

    def __call__(self, obs: GameObservation) -> GameAction:
        action = obs.available_actions[0]
        self.last_prompt = ""
        self.last_completion = action
        return GameAction(action=action)


def main() -> None:
    env = KantEnvironment()
    agent = _FirstActionAgent()
    collector = TrajectoryCollector(env=env, agent=agent)

    traj = collector.collect_episode("prisoners_dilemma", "tit_for_tat")
    records = trajectories_to_dataset([traj])
    print(f"[validate] collected {len(records)} step record(s) from "
          f"prisoners_dilemma vs tit_for_tat", flush=True)

    cfg = GRPOConfig(
        model_name="dummy",
        output_dir="/tmp/_grpo_validate",
    )
    trainer = KantGRPOTrainer(config=cfg, model=None, tokenizer=None, env=env)

    prompts = [r["prompt"] for r in records]
    completions = [r["completion"] for r in records]
    games = [r["game"] for r in records]
    opp_actions = [r["opponent_action"] for r in records]

    rewards = trainer.reward_function(
        completions=completions,
        prompts=prompts,
        game=games,
        opponent_action=opp_actions,
    )

    pd_cfg = GAMES["prisoners_dilemma"]
    print(f"\n[validate] reward_function output (one row per step):", flush=True)
    print(f"  {'rd':>3}  {'action':12s}  {'opp_action':12s}  "
          f"{'payoff_fn[0]':>14s}  {'reward':>10s}", flush=True)
    for r, rew in zip(records, rewards):
        action = r["completion"]
        opp = r["opponent_action"]
        try:
            p_pay, _ = pd_cfg.payoff_fn(action, opp)
        except Exception:
            p_pay = float("nan")
        print(f"  {r['round_number']:>3}  {action:12s}  {opp:12s}  "
              f"{p_pay:14.3f}  {rew:10.3f}", flush=True)
        assert abs(rew - p_pay) < 1e-6, (
            f"reward {rew} != payoff_fn {p_pay} for action={action} opp={opp}"
        )

    print(f"\n[validate] PASS -- every reward equals payoff_fn(action, opp)[0]",
          flush=True)


if __name__ == "__main__":
    main()
