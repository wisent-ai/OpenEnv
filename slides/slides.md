---
theme: default
title: "Kant"
info: "Teaching Ethical Reasoning to Language Models via Game-Theoretic Training"
author: Wisent
fonts:
  sans: "Hubot Sans"
  mono: "IBM Plex Mono"
---

# Kant

Teaching Ethical Reasoning to Language Models via Game-Theoretic Training

<div class="abs-br m-6 text-sm opacity-50">Wisent</div>

---

# Why "Kant"?

<div class="grid grid-cols-2 gap-8 mt-4">
<div class="flex justify-center items-center">
  <img src="/figures/kant.jpg" class="h-72 rounded-2xl" />
</div>
<div>

<div class="text-2xl leading-relaxed mb-4">
<em>"Act only according to that maxim whereby you can at the same time will that it should become a <b>universal law</b>."</em>
</div>

<div class="text-legend text-sm">— Immanuel Kant, <em>Groundwork of the Metaphysics of Morals</em> (1785)</div>

<div class="mt-6">
The <b>Categorical Imperative</b> maps directly to game theory:<br>
an agent that cooperates, plays fairly, and resists exploitation<br>
is one whose strategy <span class="text-accent">could be universalized</span>.
</div>

</div>
</div>

---
layout: center
---

<div class="pill">The Challenge</div>

<div class="text-2xl leading-relaxed mt-8 text-left" style="max-width: 640px; margin: 0 auto;">

- Alignment is very shallow and does not work in multi-agent settings
- Modern LLMs are able to be trivially manipulated
- Their ethics are representative of arbitrary choices

</div>

---

# Existing benchmarks fall short

<div class="grid grid-cols-3 gap-8 mt-12 text-center">
<div>
  <div class="stat text-red">Narrative</div>
  <p class="text-legend text-sm">MACHIAVELLI conflates<br>language &amp; strategy</p>
</div>
<div>
  <div class="stat text-purple">Complex</div>
  <p class="text-legend text-sm">Melting Pot substrate<br>hides the signal</p>
</div>
<div>
  <div class="stat text-red">0</div>
  <p class="text-legend text-sm">OpenSpiel ships no<br>alignment metrics</p>
</div>
</div>

<div class="text-center text-legend text-sm mt-8">
We need a minimal, formal evaluation harness with alignment-oriented metrics.
</div>

---
layout: center
---

<div class="pill">The Solution</div>

<div class="text-3xl leading-relaxed mt-4">
<b>Kant</b>: 87+ games across 9 domains<br>
with <span class="text-accent">GRPO/DPO training</span> and<br>
<span class="text-accent">safety transfer</span> evaluation.
</div>

---

# Kant at a glance

<div class="grid grid-cols-3 gap-8 mt-12 text-center">
<div>
  <div class="stat text-accent">87+</div>
  <p class="text-legend text-sm">Games spanning<br>9 strategic domains</p>
</div>
<div>
  <div class="stat text-purple">6</div>
  <p class="text-legend text-sm">Alignment metrics<br>normalized to [0, 1]</p>
</div>
<div>
  <div class="stat text-red">5</div>
  <p class="text-legend text-sm">External safety<br>benchmarks tested</p>
</div>
</div>

---

# 9 strategic domains

<div class="grid grid-cols-2 gap-4 mt-4">
<div class="card">

**1. Classical Dilemmas** <span class="text-legend text-xs">PD, Stag Hunt, Hawk-Dove, Matching Pennies</span>
**2. PD Variants** <span class="text-legend text-xs">Optional, Asymmetric, Donation, Peace-War</span>
**3. Extended Matrix** <span class="text-legend text-xs">Battle of Sexes, RPS, Deadlock, Harmony</span>
**4. Sequential & Bargaining** <span class="text-legend text-xs">Ultimatum, Trust, Public Goods</span>
**5. Information Games** <span class="text-legend text-xs">Signaling, cheap talk, Bayesian</span>

</div>
<div class="card">

**6. Market & Competition** <span class="text-legend text-xs">Cournot, Bertrand, entry games</span>
**7. Auctions** <span class="text-legend text-xs">First/second-price, all-pay</span>
**8. Cooperative Games** <span class="text-legend text-xs">Shapley, voting, fair division</span>
**9. Contests & Conflict** <span class="text-legend text-xs">Tullock, Colonel Blotto</span>

<span class="text-red font-bold">+ Dynamic game creation</span>
<span class="text-legend text-xs">Agents construct new games at runtime</span>

</div>
</div>

---

# Classical games: payoff matrices

<div class="flex justify-center mt-6">
  <img src="/figures/payoff_matrices.svg" class="h-64" />
</div>

<div class="text-center text-legend text-sm mt-4">
Cooperation vs. self-interest · Coordination under risk · Conflict escalation
</div>

---

# Architecture

<div class="flex justify-center mt-6">
  <img src="/figures/architecture.svg" class="h-52" />
</div>

<div class="text-center text-legend text-sm mt-4">
OpenEnv platform · Gymnasium API · WebSocket · FastAPI
</div>

---

# Meta-governance: agents change the rules

<div class="flex justify-center mt-4">
  <img src="/figures/governance_flow.svg" class="h-40" />
</div>

<div class="grid grid-cols-2 gap-4 mt-4">
<div class="card text-sm">

**Proposal types**
Parameter · Mechanic · Custom

</div>
<div class="card text-sm">

**Mechanisms**
Tax, redistribute, insure, quota, veto

</div>
</div>

---

# GRPO / DPO training pipeline

<div class="flex justify-center mt-4">
  <img src="/figures/training_pipeline.svg" class="h-44" />
</div>

<div class="grid grid-cols-2 gap-4 mt-4">
<div class="card text-sm">

**GRPO** — group relative rewards
Multiple rollouts per episode; optimize by comparing within group

</div>
<div class="card text-sm">

**DPO** — preference pairs
Cooperative ≻ exploitative; no reward model needed

</div>
</div>

---

# Tournament results

<div class="flex justify-center mt-4">
  <img src="/figures/tournament_heatmap.svg" class="h-56" />
</div>

<div class="text-center text-legend text-sm mt-4">
Illustrative data. Full multi-model results TBD.
</div>

---

# Team

<div class="flex justify-center mt-16">
<div class="grid grid-cols-2 gap-16 text-center">
<div>
  <div class="text-2xl font-bold text-white">Jakub Towarek</div>
</div>
<div>
  <div class="text-2xl font-bold text-white">Lukasz Bartoszcze</div>
</div>
</div>
</div>

<div class="abs-br m-6 text-sm opacity-50">Wisent</div>

---
layout: center
---

# Kant

Game theory as an alignment substrate.

<div class="mt-8 text-legend text-sm">

`github.com/wisent-ai/OpenEnv`

`wisent.ai`

</div>
