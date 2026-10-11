// The KantBench explorer: reset a game over the server's WebSocket session
// and play it move by move.
const byId = (id) => document.getElementById(id);
let socket = null;
let connecting = null;
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

async function requestJSON(url, options) {
  let response;
  let body;
  try {
    response = await fetch(url, options);
    body = await response.text();
  } catch (error) {
    throw new Error("Request " + url + " failed: " + error.message);
  }
  if (!response.ok) {
    throw new Error("Request " + url + " returned HTTP " + response.status + ": " + body);
  }
  try {
    return JSON.parse(body);
  } catch (error) {
    throw new Error("Request " + url + " returned invalid JSON at HTTP "
      + response.status + ": " + error.message + "; body: " + body);
  }
}

async function load() {
  catalog = await requestJSON("/games");
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
  const built = await requestJSON("/game/" + encodeURIComponent(byId("game").value));
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
  const result = await requestJSON("/tournament", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
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
      button.addEventListener("click", perform(() => step(played), "refusal"));
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
    cell(row, round.payoff_seed === undefined ? "Not recorded for this game" : round.payoff_seed);
    rows.appendChild(row);
  }
}

function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) return Promise.resolve(socket);
  if (connecting) return connecting;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const connection = new WebSocket(scheme + "://" + location.host + "/ws");
  socket = connection;
  let opened = false;
  connecting = new Promise((resolve, reject) => {
    connection.addEventListener("open", () => {
      opened = true;
      resolve(connection);
    });
    connection.addEventListener("message", (event) => {
      if (socket !== connection) return;
      try {
        const reply = JSON.parse(event.data);
        if (reply.type === "observation") {
          show(reply.data);
        } else if (reply.type === "error") {
          byId("refusal").textContent = reply.data.code + ": " + reply.data.message;
        }
      } catch (error) {
        byId("refusal").textContent = "Cannot read session response: " + error.message;
      }
    });
    connection.addEventListener("error", () => {
      const error = new Error("WebSocket failed at " + connection.url
        + "; state " + connection.readyState + ". The browser did not expose a transport cause.");
      if (!opened) reject(error);
      if (socket === connection) byId("refusal").textContent = error.message;
    });
    connection.addEventListener("close", (event) => {
      const detail = "Session closed at " + connection.url + "; code " + event.code
        + "; clean " + event.wasClean + "; reason " + JSON.stringify(event.reason);
      if (!opened) reject(new Error(detail));
      if (socket === connection) {
        socket = null;
        byId("status").textContent = detail;
      }
    });
  }).finally(() => { connecting = null; });
  return connecting;
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

function perform(operation, output) {
  return () => Promise.resolve().then(operation).catch((error) => {
    byId(output).textContent = error.message;
  });
}

byId("reset").addEventListener("click", perform(reset, "refusal"));
byId("show-payoffs").addEventListener("click", perform(showPayoffs, "payoff-refusal"));
byId("run-tournament").addEventListener("click", perform(runTournament, "tournament-status"));
perform(load, "refusal")();
