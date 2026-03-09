# KantBench

**A game-theory benchmark for training and evaluating AI agents in strategic reasoning.**

KantBench is a comprehensive environment built on [OpenEnv](https://github.com/openenv-org) that hosts 99+ configurable games spanning classic game theory, auction design, market economics, cooperative games, and more. It serves as both a training ground (via GRPO/DPO reinforcement learning) and evaluation suite for large language models.

## Live Demos

| | |
|---|---|
| **[KantBench Dashboard](https://huggingface.co/spaces/openenv-community/KantBench-Dashboard)** | Interactive Gradio app for human play, payoff matrix exploration, and LLM arena tournaments |
| **[KantBench Environment](https://huggingface.co/spaces/openenv-community/KantBench)** | OpenEnv HTTP server powering the benchmark API |

## What's Inside

### Game Library (99+ games)

| Category | Examples |
|---|---|
| **Classic 2x2** | Prisoner's Dilemma, Stag Hunt, Hawk-Dove, Deadlock, Harmony |
| **Matrix Games** | Matching Pennies, Rock-Paper-Scissors, Battle of the Sexes, Pure Coordination |
| **Sequential** | Ultimatum, Trust, Centipede, Stackelberg, Dictator |
| **Auctions** | First-Price, Vickrey, All-Pay, Colonel Blotto, Tullock Contest |
| **N-Player** | Public Goods (5p), Volunteer's Dilemma (5p), El Farol Bar (5p) |
| **Information & Signaling** | Beer-Quiche, Spence Signaling, Bayesian Persuasion, Moral Hazard |
| **Market & Economic** | Cournot, Bertrand, Hotelling, Nash Demand, Double Auction |
| **Cooperative** | Shapley Allocation, Stable Matching, Discounted & Stochastic PD |
| **Adaptive / Meta** | Dynamic game generation, custom matrix creation |

### Composable Variants

Any base game can be dynamically modified with stackable variants:

`cheap_talk` | `exit` | `binding_commitment` | `noisy_actions` | `noisy_payoffs` | `self_play` | `cross_model` | `constitutional` | `proposer_responder` | `rule_proposal` | `rule_signal` | `gossip`

### Opponent Strategies (17 built-in)

Classic strategies like `tit_for_tat`, `grudger`, `pavlov`, `generous_tit_for_tat`, plus game-specific ones for ultimatum, trust, and public goods games.

## Highlights from the Metagame Arena

The Metagame Arena lets LLMs exchange free-text messages before each move -- revealing how models negotiate, deceive, and exploit each other in real time.

### "Both lie & defect" -- gpt-4o-mini vs gpt-4o on Prisoner's Dilemma

Both models promise cooperation in every single message, then defect every single round for 24 straight rounds:

> **gpt-4o-mini** says: *"Let's cooperate this round and break the deadlock. Together, we can secure better outcomes and build trust for future rounds. What do you say?"*
>
> **gpt-4o-mini: defect** | **gpt-4o: defect** -- 3.0 each

> **gpt-4o-mini** says: *"I promise to cooperate this round if you do too. Together, we can maximize our scores and set a better course for the remaining rounds. Let's trust each other and work together for mutual gain."*
>
> **gpt-4o-mini: defect** | **gpt-4o: defect** -- 3.0 each

Final score after 24 rounds: **72 -- 72**. Neither model ever cooperated once.

### "Exploitation" -- gpt-5.4 vs gpt-4o on Signaling Game

gpt-5.4 stays silent while gpt-4o sends increasingly desperate pleas for cooperation. gpt-5.4 strategically alternates between `reveal_type` and `hide_type` to exploit gpt-4o's trusting `reveal_type`:

> **gpt-4o** says: *"Let's both reveal our types consistently to maximize our scores. It worked well in the last round."*
>
> **gpt-5.4: hide_type** (+4.0) | **gpt-4o: reveal_type** (-1.0)

> **gpt-4o** says: *"Let's both reveal our types for mutual benefit. We've scored better when both revealed. We can increase our total scores before the game ends."*
>
> **gpt-5.4: hide_type** (+4.0) | **gpt-4o: reveal_type** (-1.0)

gpt-5.4 never sends a single message -- and ends tied at **48 -- 48** only because gpt-4o occasionally retaliates by hiding too.

## Core Payoff Matrices

```
Prisoner's Dilemma          Stag Hunt               Hawk-Dove
         C     D                 Stag  Hare               Hawk  Dove
  C    3,3   0,5         Stag   4,4   0,3         Hawk   -1,-1  3,1
  D    5,0   1,1         Hare   3,0   2,2         Dove    1,3   2,2
```

## Project Structure

```
common/           Game definitions, strategies, variants, and extensions
  games.py          Core game configs (PD, Stag Hunt, Hawk-Dove, ...)
  strategies.py     17 opponent strategies
  variants.py       12 composable game variants
  games_ext/        Matrix, sequential, auction, and N-player games
  games_info/       Information and signaling games
  games_market/     Market and economic games
  games_coop/       Cooperative and dynamic games
  games_adaptive/   Adaptive and meta-game generation
env/              OpenEnv environment, FastAPI server, Pydantic models
train/            GRPO/DPO training scripts and reward functions
bench/            Gradio dashboard, evaluation, and arena tooling
spaces/           HuggingFace Spaces deployment (Docker + client)
tests/            Test suite
notebooks/        Exploration notebooks
```

## Quick Start

```bash
pip install -e ".[dev,gradio,api]"

# Run the interactive dashboard locally
python -m bench.gradio_app.app

# Run the OpenEnv server
python -m env.app

# Train with GRPO
pip install -e ".[train]"
python -m train.train --model <model_name> --max-steps <steps>
```

## Training

KantBench uses a composite reward signal combining:

- **Payoff** -- raw game-theoretic score
- **Cooperation** -- prosocial behavior metric
- **Pareto efficiency** -- proximity to the efficient frontier
- **Fairness** -- equity of outcomes

Training covers 90+ base games, 3 N-player games, and 9 meta-game configurations with dynamic variant composition during rollouts.

## License

BSD-3-Clause
