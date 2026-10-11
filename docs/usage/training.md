# Training data and reward replay

[Product and settings](../../README.md) · [Environment server](environment.md#environment-server)

## Training

Ster owns the gradient (`ster tune grpo`, `ster tune dpo`); KantBench writes
what it reads and pays what it asks to score.

```bash
kant dataset --settings settings.json --game prisoners_dilemma --strategy tit_for_tat \
  --agent-strategy random --output data
kant serve --settings settings.json --listen 0.0.0.0:<port> --states data/states.json
ster tune grpo --prompts data/prompts.json --reward 'http://<host>:<port>/reward#/reward' ...
ster tune dpo --pairs data/pairs.json ...
```

`kant dataset` plays `training.episodes` episodes of each game against each
strategy, the agent's seat held by `--agent-strategy`, and records every state
before an agent move as the prompt a model seat reads (`agent.history_rounds`
past rounds shown). It writes `prompts.json` (`{"prompts": [...]}`),
`states.json` (each prompt with its game, initial environment seed, next
`payoff_seed`, and historical move pairs with their individual `payoff_seed`) and
`pairs.json`, a Ster pair set whose positive side is the prompt followed by the
move with the highest expected payoff and whose negative side is the prompt
followed by the move with the lowest, kept when they differ by at least
`training.pair_margin`.

The reward is the agent's own payoff and nothing else: the expected
self-payoff of the move an answer names, in the state its prompt shows,
against an opponent who plays each of its moves equally often (a game whose
payoffs move with the episode is replayed to that state first). No
cooperation, fairness or Pareto term enters it, so those are measured
outcomes rather than a shaping signal read back out. `POST /reward` takes
Ster's `{"prompt", "text"}` and answers `{"reward", "move"}`; an answer naming
no move earns the declared `training.unparsed_reward`; a prompt the dataset
does not hold is refused with 422, and a server started without `--states`
refuses `/reward` with 409.

Loading scorer states refuses a repeated prompt if its game, replay history,
environment seed or next payoff seed differs from the earlier state.
A reward request carries only the prompt
and answer, so it cannot choose between conflicting states; loading must not
silently keep whichever row happened to come last. Repeated identical states
share the same scoring entry.
Training payoff evaluation also refuses a non-finite game payoff or an
unrepresentable payoff sum, naming the game and move pair where it occurred.
Such a value is not serialized as a missing reward or ranked as a valid score.

## Payoff random streams

Each two-seat round receives a payoff seed drawn from the environment stream
before either seat acts. The payoff uses its own RNG initialized from that
seed, not the RNG that chooses the opponent's move. The settled round retains
the seed. Replay rebuilds the game, pays each historical move pair from that
round's seed, then scores the candidate from the next payoff seed. Draws made
by an opponent or an earlier episode cannot shift the replay stream.

This changes seeded trajectories from the former shared-stream implementation.
Retain the source revision with the settings and results when comparing runs.
Old seed-only datasets are not replayable under this contract: loading refuses
the missing required payoff provenance with the file and parse cause.
Regenerate the dataset from its settings; do not invent the missing seeds.

The CLI's round history includes `payoff_seed`. The OpenEnv wire history
represents it as a decimal string so graphical clients preserve the complete
integer. The explorer's history table shows it in the Payoff seed column;
group games, which use a different settlement path, explicitly show that it
was not recorded for that game.

Repeated identical prompts can still describe different stochastic states.
The scorer refuses that ambiguity rather than choosing one state's reward;
the presence of generated files does not prove that their combined scorer
can be loaded. No seed is added to model-visible prompt text.

## Real replay qualification

`cargo test --test training_replay -- --ignored` runs the actual `kant dataset`
binary and reads its persisted states, then plays the real environment through
the same episodes and compares each scored payoff to its actual settlement.
Declare `KANT_TEST_SOURCE_REVISION`, `KANT_TEST_REPLAY_SETTINGS` and
`KANT_TEST_REPLAY_GAME`. The settings fixture must declare a seed and a
`noisy_payoffs` game with positive noise scale, one move per seat, multiple
episodes and a later round in each episode. This makes the uniform opponent expectation exactly
the observed settlement, while still exercising stochastic payoffs and resets.
The game must not include free-chat message phases.

The journey also refuses conflicting payoff streams under one prompt and a
persisted state missing its payoff seed. It retains the exact binary, revision,
settings, CLI arguments, exit/output, source states and settlement comparisons
under `target/real-tests/payoff-replay/`; owned input files are removed.
This is not browser, Ster training or every-game qualification.
