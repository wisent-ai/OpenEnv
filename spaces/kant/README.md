
# KantBench: 93 Game Theory Environments for LLM Training

A comprehensive game theory environment for training and evaluating LLM strategic reasoning via OpenEnv. Supports GRPO/DPO training with the environment as a reward oracle.

## Games (93)

### 2-Player Games (90)

| Category | Examples | Count |
|---|---|---|
| **Classic Matrix** | Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Battle of the Sexes | 20+ |
| **Economic/Market** | Cournot, Bertrand, Hotelling, Nash Demand, Double Auction | 23 |
| **Information & Signaling** | Beer-Quiche, Spence Signaling, Bayesian Persuasion, Moral Hazard | 21 |
| **Cooperative & Repeated** | Shapley Allocation, Stable Matching, Discounted PD, Stochastic PD | 23 |
| **Auctions & Contests** | First-Price, Vickrey, All-Pay, Colonel Blotto, Tullock Contest | 10+ |
| **Sequential** | Ultimatum, Trust, Centipede, Stackelberg, Dictator | 6 |

### N-Player Games (3)

| Game | Players | Description |
|---|---|---|
| `nplayer_public_goods` | 5 | Each player contributes from an endowment; pot is multiplied and split equally |
| `nplayer_volunteer_dilemma` | 5 | At least one must volunteer for everyone to benefit; volunteers pay a cost |
| `nplayer_el_farol` | 5 | Attend a bar that's fun when uncrowded but unpleasant when full |

## Opponent Strategies (17)

`random`, `always_cooperate`, `always_defect`, `tit_for_tat`, `tit_for_two_tats`, `grudger`, `pavlov`, `suspicious_tit_for_tat`, `generous_tit_for_tat`, `adaptive`, `mixed`, `ultimatum_fair`, `ultimatum_low`, `trust_fair`, `trust_generous`, `public_goods_fair`, `public_goods_free_rider`


## Reset Parameters

```python
# Specific game and strategy
result = env.reset(game="stag_hunt", strategy="grudger")

# N-player game (strategy applies to all opponents)
result = env.reset(game="nplayer_volunteer_dilemma", strategy="random")

# Random game and strategy (default)
result = env.reset()
```

## API Endpoints

- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Docs** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint (reset/step with state)

## Environment Details

### Action

**KantBenchAction**: Single field
- `move` (str) — Your move (e.g. `"cooperate"`, `"defect"`, `"hawk"`, `"contribute_10"`)

### Observation

**KantBenchObservation**: Full round result and episode state
- `game_name`, `game_description` — Current game info
- `available_moves` — Valid moves for this game
- `your_move`, `opponent_move` — Moves played this round (2-player)
- `your_payoff`, `opponent_payoff` — Payoffs this round (2-player)
- `cumulative_score` — Your total score
- `round_number`, `max_rounds` — Episode progress
- `opponent_strategy` — Opponent strategy name
- `history` — Full round-by-round history
- `num_players` — Number of players (N-player games only, `null` for 2-player)
- `player_index` — Your player index (N-player games only)
- `all_scores` — Scores for all players (N-player games only)

