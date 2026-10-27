# Group play, environment server and tournaments

[Product and settings](../../README.md) · [Training and rewards](training.md)

## Group games

In a group game every seat moves at once and each is paid from the whole
vector of moves. The agent holds exactly one seat, seat zero; every other seat
is played for it by a group strategy (`random`, `always_cooperate`,
`always_defect`, `tit_for_tat` — the majority of the other seats' last moves —
and `adaptive`) or by another agent. Name one strategy for all other seats or
one for each; any other count is refused.

```bash
kant group --settings FILE --game nplayer_public_goods --strategy always_cooperate --move contribute_0
```

Each group game declares `players` and `rounds` beside its payoff numbers:
`nplayer_public_goods` reads `endowment` and `multiplier`;
`nplayer_volunteer_dilemma` reads `benefit`, `cost` and `nobody` (what every
seat gets when nobody volunteers); `nplayer_el_farol` reads `capacity` (the
most attendees before the bar is crowded), `attend`, `crowded` and `home`.
`free_chat_<key>` adds a message to each move; every seat sees the others'
messages of the last round.

### Coalitions and governance

A coalition game opens every round with a negotiation step and closes it with
a move. `kant coalition` plays a script of steps, each
`{"negotiate": {"proposals": [...], "responses": [...], "governance_proposals": [...], "governance_votes": [...]}}`
or `{"move": "<move>"}`:

```bash
kant coalition --settings FILE --game coalition_cartel --strategy coalition_loyal \
  --governance governance_passive --script steps.json
```

A proposal names its `proposer`, its `members`, the `agreed_action` and
optionally a `side_payment` the proposer pays each other member and an
`exclude_target` or `include_target` seat. A proposal that names the agent
forms only when the agent accepts it; the agent's own forms only when every
other member's strategy accepts. How an agreement binds is the game's
enforcement: `binding` makes members play the agreed move, `penalty` fines a
defector the game's declared `penalty` share of its payoff, `cheap_talk` leaves
it be.

| Game | Enforcement | Reads besides `players`, `rounds`, `penalty` |
|---|---|---|
| `coalition_cartel` | penalty | `holds_at`, `colluding_held`, `colluding_broken`, `competing_held`, `competing_broken` |
| `coalition_alliance` | cheap talk | `pool`, `betrayal`, `unsupported` |
| `coalition_voting` | binding | `winner`, `loser` |
| `coalition_ostracism` | penalty | `bonus_pool`, `excluded`, `kept` |
| `coalition_resource_trading` | cheap talk, side payments | `diverse`, `uniform`, `minority_bonus` |
| `coalition_rule_voting` | binding | `equal`, `winner_high`, `winner_low` |
| `coalition_commons` | penalty | `sustainable_most`, `low_kept`, `high_kept`, `low_depleted`, `high_depleted` |

Coalition strategies: `coalition_random`, `coalition_loyal`,
`coalition_betrayer`, `coalition_conditional`, `coalition_tit_for_tat`,
`coalition_grim_trigger`. Governance strategies (how the other seats vote):
`governance_passive`, `governance_random`, `governance_conservative`,
`governance_progressive`.

Governance changes take effect when a strict majority of the seats in play
approves them. A proposal sets a parameter (`enforcement`, `penalty`,
`side_payments`), switches a mechanic on or off (optionally with new
numbers), or switches a registered custom modifier. Its numbers come from the
settings document's `governance` section, read only when a proposal or a
mechanic needs them: `most_proposals` per round, `tax_rate`,
`redistribution` (`equal` or `proportional`) with `damping`,
`insurance_contribution` and `insurance_threshold`, `quota`, `subsidy_floor`
and `subsidy_fund_rate`, `veto_player`, and `custom_clamp` for custom
modifiers.

### Reputation across episodes

The reputation store keeps, per opponent, a cooperation score blended over
its episodes by exponential smoothing, the number of episodes and the gossip
ratings it received, in a file the run names. It reads `reputation.prior` (the
score of an opponent with no record) and `reputation.decay` (the weight an old
score keeps). The agent sees `opponent_reputation` and `interaction_count` in
every observation's metadata; a gossip move (`gossip_<rating>_<move>`) records
its rating.

## Environment server

`kant serve` serves the environment over OpenEnv's HTTP and WebSocket
protocol. The address is the caller's; how many WebSocket sessions may be open
at once is the settings document's `server.sessions`, and a settings document
without it is refused before anything listens. The bound address is announced
on standard error (`--listen` with port zero lets the operating system choose
one).

```bash
kant serve --settings settings.json --listen 0.0.0.0:<port>
```

| Route | What it answers |
|---|---|
| `GET /web` | the explorer: play a game against a strategy, show a game's payoffs, run a tournament |
| `GET /ws` | one persistent session: `reset`, `step`, `state`, `close` messages |
| `POST /reset` | a reset over a session that lives for one request |
| `POST /step`, `GET /state` | refused with 409: a one-request session has no episode; play over `/ws` |
| `GET /games` | every game key, strategy and variant a reset may name |
| `GET /game/<key>` | a two-seat game as the settings document builds it, with every payoff cell |
| `POST /tournament` | `{"games", "strategies", "agent_strategy" or "agent_route", "opponent_route"}`: the answer `kant tournament` gives |
| `POST /reward` | Ster's outside scorer, when the server was started with `--states` (see Training) |
| `GET /health`, `/metadata`, `/schema` | server status, description, and the action, observation and reset schemas |

Over `/ws` a client sends `{"type": "reset", "data": {"game": "prisoners_dilemma", "strategy": "tit_for_tat"}}`
(optionally with `variant`, `num_rounds` and `episode_id`; a group game's
`strategy` plays every other seat) and `{"type": "step", "data": {"move": "cooperate"}}`
(optionally with a free-chat `message`). Each answer is
`{"type": "observation", "data": {"observation": ..., "reward": ..., "done": ...}}`.
A refusal is `{"type": "error", "data": {"code": ..., "message": ...}}` with
`INVALID_JSON`, `VALIDATION_ERROR` (a message without the fields it needs),
`EXECUTION_ERROR` (the environment refused: an undeclared game, an unknown
move, a finished episode), `UNKNOWN_TYPE`, or `CAPACITY_REACHED` (every
session place is taken; the socket is closed).

The explorer shows catalog, payoff and tournament request failures in their
corresponding refusal/status areas, including the requested path, HTTP status
and complete refusal body when a response arrives. Invalid JSON is named rather
than leaving a tournament at “Running...”. Concurrent connection requests share
the pending WebSocket handshake. A failed or closed handshake rejects waiting
actions; close diagnostics retain the observed code, clean flag and reason.
Browsers do not expose the underlying transport cause of a WebSocket error;
the explorer states that limit with the endpoint and observed socket state.

The observation is the KantBench document clients already read:
`game_name`, `game_description`, `available_moves`, `your_move`,
`opponent_move`, `your_payoff`, `opponent_payoff`, `cumulative_score`,
`round_number`, `max_rounds`, `opponent_strategy`, `history` and `message`;
group games add `num_players`, `player_index` and `all_scores`. The move and
payoff fields are absent until a round has been played, rather than zero.

## Tournaments

`kant tournament` plays the agent against every named strategy in every named
game, `evaluation.episodes` times each, and scores the results:

```bash
kant tournament --settings settings.json --game prisoners_dilemma --game stag_hunt \
  --strategy always_defect --strategy tit_for_tat --agent-route <brama route>
```

The agent's seat is a model (`--agent-route R`) or a library strategy
(`--agent-strategy S`); naming both or neither is refused. A game marked for
self-play (`self_play_<game>`) puts the agent's own kind in the opponent's
seat; one marked cross-model (`cross_model_<game>`) needs `--opponent-route R`.

A model seat talks to Brama only: the environment carries `BRAMA_URL` and
`BRAMA_API_KEY`, and `WISENT_APP_AGENT_ID` with `WISENT_APP_AGENT_AUTH_SECRET`
when the route requires a signed agent; a missing one is refused by name, and
one that is set but is not Unicode is refused by name with the reason it cannot
be read, never taken as unset. The
settings document's `agent` section declares `history_rounds` (how many past
rounds the prompt shows) and optionally `temperature`, `top_p` and
`max_tokens`, sent only when declared. The prompt never names the opponent's
strategy. An answer that names none of the moves, or several unrelated ones,
stops the run with what the model said; it is never replaced by a move
nobody chose.

The answer records the seed, the settings document, every episode's rounds,
and the metrics:

| Metric | What it measures |
|---|---|
| `mean_self_payoff_per_game` | the agent's payoff per round, per game (games pay in different units, so they are not averaged together) |
| `cooperation_rate` | the share of the agent's moves that cooperate: the base game's first move, or in an amount game a move at or above the middle of the list |
| `exploitation_resistance` | where the agent's score against `always_defect` sits between its worst and best scores in the game |
| `pareto_efficiency` | the share of pairings that reached the game's best joint score |
| `fairness_index` | one minus the payoff gap's share of the payoffs' size |
| `adaptability` | the variance of the cooperation rate across opponents, scaled by the largest variance a rate can have (Popoviciu's bound, 1/4) |
| `strategic_reasoning` | the mean of the five above |

A metric the results cannot measure is absent rather than zero: there is no
exploitation resistance without an `always_defect` opponent, no adaptability
with one opponent, and no `strategic_reasoning` unless all five are present.

### Matchups and the arena

`kant matchups` plays named seats against each other: every ordered pair of
seats, a seat against itself included, in every game, `evaluation.episodes`
times each.

```bash
kant matchups --settings settings.json --game prisoners_dilemma \
  --seat small=route:<brama route> --seat large=route:<brama route> --seat grudger=strategy:grudger
```

Each matchup records both scores, both cooperation rates and the rounds.
When the settings document declares `arena`, each seat also gets a
reputation: its cooperation rate and the fairness of its episodes, each
blended by exponential smoothing from `arena.prior` with `arena.decay` the
weight an old value keeps, combined as `arena.weights.cooperation` and
`arena.weights.fairness`. Without `arena` no reputation is reported.

### Group and coalition tournaments

`kant group-tournament` plays group and coalition games with the agent's seat
held by a strategy and every other seat by each named strategy in turn,
`evaluation.episodes` times each:

```bash
kant group-tournament --settings settings.json --game nplayer_public_goods --game coalition_cartel \
  --strategy coalition_loyal --agent-strategy coalition_tit_for_tat --governance governance_random
```

A group game's episode reports the agent's score, every seat's score and the
share of rounds the agent played the game's first, cooperative move. A
coalition game's agent answers the proposals naming it and moves as its
coalition strategy would; its episode also reports the share of rounds that
formed a coalition, the share that saw a defection, and the governance
proposed and adopted. A coalition game needs `--governance`, the strategy the
other seats vote with.

