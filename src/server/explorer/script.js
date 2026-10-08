// The KantBench explorer: reset a game over the server's WebSocket session
// and play it move by move.
const byId = (id) => document.getElementById(id);
let socket = null;
let catalog = null;

function fill(select, names) {
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
}

function strategiesFor(game) {
  const select = byId("strategy");
  select.innerHTML = "";
  fill(select, catalog.group.includes(game) ? catalog.group_strategies : catalog.strategies);
}

async function load() {
  catalog = await (await fetch("/games")).json();
  fill(byId("game"), catalog.pair.concat(catalog.group));
  fill(byId("variant"), catalog.variants);
  fill(byId("agent-strategy"), catalog.strategies);
  for (const name of catalog.strategies) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = name;
    box.className = "opponent";
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + name));
    byId("opponents").appendChild(label);
  }
  strategiesFor(byId("game").value);
  byId("game").addEventListener("change", () => strategiesFor(byId("game").value));
}

async function showPayoffs() {
  const table = byId("payoffs");
  table.innerHTML = "";
  byId("payoff-refusal").textContent = "";
  const response = await fetch("/game/" + encodeURIComponent(byId("game").value));
  if (!response.ok) {
    byId("payoff-refusal").textContent = (await response.json()).detail;
    return;
  }
  const built = await response.json();
  const head = document.createElement("tr");
  cell(head, "you \\ opponent");
  for (const column of built.columns) cell(head, column);
  table.appendChild(head);
  built.rows.forEach((rowName, index) => {
    const row = document.createElement("tr");
    cell(row, rowName);
    for (const paid of built.cells[index]) {
      cell(row, Array.isArray(paid) ? paid.join(", ") : paid.refused);
    }
    table.appendChild(row);
  });
}

async function runTournament() {
  const strategies = Array.from(document.querySelectorAll(".opponent:checked")).map((box) => box.value);
  const request = { games: [byId("game").value], strategies };
  if (byId("agent-strategy").value) request.agent_strategy = byId("agent-strategy").value;
  if (byId("agent-route").value) request.agent_route = byId("agent-route").value;
  byId("tournament-status").textContent = "Running...";
  const table = byId("metrics");
  table.innerHTML = "";
  const response = await fetch("/tournament", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  const result = await response.json();
  if (!response.ok) {
    byId("tournament-status").textContent = result.detail;
    return;
  }
  byId("tournament-status").textContent = result.total_episodes + " episodes, seed " + result.seed;
  for (const [name, value] of Object.entries(result.metrics)) {
    const row = document.createElement("tr");
    cell(row, name);
    cell(row, value === null ? "not measurable from these results" : JSON.stringify(value));
    table.appendChild(row);
  }
}

function cell(row, text) {
  const td = document.createElement("td");
  td.textContent = text;
  row.appendChild(td);
}

function show(result) {
  const seen = result.observation;
  byId("refusal").textContent = "";
  byId("status").textContent = seen.game_name + ": round " + seen.round_number + " of " + seen.max_rounds + (result.done ? " (done)" : "");
  byId("game-text").textContent = seen.game_description;
  const moves = byId("moves");
  moves.innerHTML = "";
  if (!result.done) {
    for (const played of seen.available_moves) {
      const button = document.createElement("button");
      button.textContent = played;
      button.addEventListener("click", () => step(played));
      moves.appendChild(button);
    }
  }
  byId("scores").textContent = seen.all_scores ? "Scores: " + seen.all_scores.join(", ") : "Your score: " + seen.cumulative_score;
  const rows = byId("history");
  rows.innerHTML = "";
  for (const round of seen.history) {
    const row = document.createElement("tr");
    cell(row, round.round);
    cell(row, round.actions ? round.actions.join(", ") : round.your_move + " / " + round.opponent_move);
    cell(row, round.payoffs ? round.payoffs.join(", ") : round.your_payoff + " / " + round.opponent_payoff);
    rows.appendChild(row);
  }
}

function connect() {
  return new Promise((resolve) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      resolve(socket);
      return;
    }
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(scheme + "://" + location.host + "/ws");
    socket.addEventListener("open", () => resolve(socket));
    socket.addEventListener("message", (event) => {
      const reply = JSON.parse(event.data);
      if (reply.type === "observation") {
        show(reply.data);
      } else if (reply.type === "error") {
        byId("refusal").textContent = reply.data.code + ": " + reply.data.message;
      }
    });
    socket.addEventListener("close", () => {
      byId("status").textContent = "Session closed.";
    });
  });
}

async function reset() {
  const data = { game: byId("game").value, strategy: byId("strategy").value };
  if (byId("variant").value) {
    data.variant = byId("variant").value;
  }
  if (byId("rounds").value) {
    data.num_rounds = Number(byId("rounds").value);
  }
  (await connect()).send(JSON.stringify({ type: "reset", data }));
}

async function step(played) {
  const data = { move: played };
  if (byId("message").value) {
    data.message = byId("message").value;
  }
  (await connect()).send(JSON.stringify({ type: "step", data }));
}

byId("reset").addEventListener("click", reset);
byId("show-payoffs").addEventListener("click", showPayoffs);
byId("run-tournament").addEventListener("click", runTournament);
load();
