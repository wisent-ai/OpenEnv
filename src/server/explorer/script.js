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
  strategiesFor(byId("game").value);
  byId("game").addEventListener("change", () => strategiesFor(byId("game").value));
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
load();
