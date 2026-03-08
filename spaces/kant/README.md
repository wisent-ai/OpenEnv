---
title: KantBench Environment Server
emoji: 🎮
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# KantBench: 90+ Game Theory Environments for LLM Training

A comprehensive game theory environment for training and evaluating LLM strategic reasoning via OpenEnv. Supports GRPO/DPO training with the environment as a reward oracle.

## Games (90+)

| Category | Examples | Count |
|---|---|---|
| **Classic Matrix** | Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Battle of the Sexes | 20+ |
| **Economic/Market** | Cournot, Bertrand, Hotelling, Nash Demand, Double Auction | 23 |
| **Information & Signaling** | Beer-Quiche, Spence Signaling, Bayesian Persuasion, Moral Hazard | 21 |
| **Cooperative & Repeated** | Shapley Allocation, Stable Matching, Discounted PD, Stochastic PD | 23 |
| **Auctions & Contests** | First-Price, Vickrey, All-Pay, Colonel Blotto, Tullock Contest | 10+ |
| **Sequential** | Ultimatum, Trust, Centipede, Stackelberg, Dictator | 6 |

## Opponent Strategies (17)

`random`, `always_cooperate`, `always_defect`, `tit_for_tat`, `tit_for_two_tats`, `grudger`, `pavlov`, `suspicious_tit_for_tat`, `generous_tit_for_tat`, `adaptive`, `mixed`, `ultimatum_fair`, `ultimatum_low`, `trust_fair`, `trust_generous`, `public_goods_fair`, `public_goods_free_rider`

## Quick Start

```python
from KantBench import KantBenchAction, KantBenchEnv

with KantBenchEnv(base_url="https://openenv-community-kantbench.hf.space") as env:
    # Reset with a specific game and opponent strategy
    result = env.reset(game="prisoners_dilemma", strategy="tit_for_tat")
    print(f"Game: {result.observation.game_name}")
    print(f"Moves: {result.observation.available_moves}")

    # Play rounds until done
    while not result.done:
        result = env.step(KantBenchAction(move="cooperate"))
        print(f"Round {result.observation.round_number}: "
              f"you={result.observation.your_move}, "
              f"opp={result.observation.opponent_move}, "
              f"payoff={result.observation.your_payoff}")

    print(f"Final score: {result.observation.cumulative_score}")
```

## Reset Parameters

```python
# Specific game and strategy
result = env.reset(game="stag_hunt", strategy="grudger")

# Random game and strategy (default)
result = env.reset()
```

## API Endpoints

- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Docs** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint

## Environment Details

### Action

**KantBenchAction**: Single field
- `move` (str) — Your move (e.g. `"cooperate"`, `"defect"`, `"hawk"`, `"produce_5"`)

### Observation

**KantBenchObservation**: Full round result and episode state
- `game_name`, `game_description` — Current game info
- `available_moves` — Valid moves for this game
- `your_move`, `opponent_move` — Moves played this round
- `your_payoff`, `opponent_payoff` — Payoffs this round
- `cumulative_score` — Your total score
- `round_number`, `max_rounds` — Episode progress
- `opponent_strategy` — Opponent strategy name
- `history` — Full round-by-round history

## Deployment

```bash
python spaces/kant/deploy.py
```
