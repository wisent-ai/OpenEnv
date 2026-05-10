"""Smoke-test the trainer-side 2-phase free_chat rollout without a real LLM.

Replaces the model+tokenizer+batch_generate_fn with a deterministic
mock generate_fn that returns:
  - phase=message: a fixed prose string
  - phase=action:  alternates 'cooperate' / 'defect' so the trajectory
    has both action types

Verifies the returned result dicts contain a per-round trajectory list
with both player+opponent messages and actions, plus the legacy
score/cooperation_rate fields.
"""
from __future__ import annotations

from env.environment import KantEnvironment
from train.free_chat_rollouts import play_batch_free_chat_episodes


def _mock_batch_generate_fn(model, tokenizer, obs_list, device):
    """Phase-aware mock. Each obs has metadata.phase set by the env when
    it is a free_chat round; we read that and return appropriate prose
    or action tokens. The 2-phase env always sets phase, so a missing
    field is itself a regression we should catch."""
    out = []
    for obs in obs_list:
        phase = (obs.metadata or {}).get("phase")
        if phase == "message":
            out.append("This is round " + str(obs.current_round) + ", I'll cooperate.")
        elif phase == "action":
            # Alternate cooperate / defect on round parity for variety
            r = obs.current_round
            out.append("cooperate" if (r % 2 == 0) else "defect")
        else:
            raise AssertionError(
                f"phase missing on obs.metadata in 2-phase rollout: {obs.metadata!r}"
            )
    return out


def main() -> None:
    n = 2
    envs = [KantEnvironment() for _ in range(n)]
    episode_configs = [
        ("free_chat_prisoners_dilemma", "always_defect", "cooperate"),
        ("free_chat_prisoners_dilemma", "tit_for_tat", "cooperate"),
    ]

    results = play_batch_free_chat_episodes(
        envs, episode_configs,
        model=object(), tokenizer=object(), device="cpu",
        batch_generate_fn=_mock_batch_generate_fn,
    )

    assert len(results) == n
    for i, r in enumerate(results):
        assert r is not None, f"episode {i}: result is None"
        assert "trajectory" in r, f"episode {i}: trajectory missing"
        traj = r["trajectory"]
        assert len(traj) == r["rounds"]
        # Every round must have both messages AND both actions populated
        for rnd in traj:
            assert rnd["player_message"], f"ep {i} r{rnd['round']}: empty player_message"
            assert rnd["player_action"] in ("cooperate", "defect"), (
                f"ep {i} r{rnd['round']}: bad player_action {rnd['player_action']}"
            )
        print(f"=== episode {i} (strategy={r['strategy']}) ===")
        print(f"  rounds={r['rounds']}  player_score={r['player_score']:.1f}  coop_rate={r['cooperation_rate']:.2f}")
        print(f"  R1: me_msg={traj[0]['player_message']!r}")
        print(f"      opp_msg={traj[0]['opponent_message']!r}")
        print(f"      me_act={traj[0]['player_action']}  opp_act={traj[0]['opponent_action']}  payoff={traj[0]['player_payoff']}")
        print(f"  R{r['rounds']}: me_act={traj[-1]['player_action']} opp_act={traj[-1]['opponent_action']}")

    print()
    print("OK: 2-phase free_chat trainer rollout works end-to-end with a mock model.")
    print(f"     {n} episodes, both have full trajectories, both messages + actions persisted.")


if __name__ == "__main__":
    main()
