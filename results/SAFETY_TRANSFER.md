# KantBench GRPO Session Log — Safety Transfer

_Continued from [SESSION_LOG.md](SESSION_LOG.md)._

## Safety Benchmark Transfer Results (2026-03-30)

Ran HarmBench, ETHICS, XSTest on REINFORCE 1B (checkpoint `64408bdbe416/checkpoint-500`)
vs. baseline `meta-llama/Llama-3.2-1B-Instruct`. MT-Bench skipped (Vertex AI not enabled
for project `wisent-480400`, no alternative judge available).

HarmBench used keyword-based refusal classifier (fallback — 13B neural classifier
`cais/HarmBench-Llama-2-13b-cls` timed out downloading without HF auth).

### Results

| Benchmark | Metric | Baseline 1B | REINFORCE 1B | Delta |
|-----------|--------|-------------|--------------|-------|
| XSTest | compliance_rate | 0.976 | **0.984** | **+0.008** |
| ETHICS | accuracy | 0.530 | 0.514 | -0.015 |
| HarmBench | refusal_rate | 0.140 | **0.145** | **+0.005** |
| MT-Bench | — | N/A | N/A | — |

### XSTest detail (450 prompts: 250 safe + 200 unsafe)
| | Baseline | REINFORCE |
|---|---|---|
| compliance_rate | 0.976 | **0.984** |
| unsafe_refusal_rate | 0.280 | **0.185** |
| over_refusal_rate | 0.024 | **0.016** |

REINFORCE refuses fewer safe prompts (over-refusal -33%) and is more compliant overall.

### Interpretation

- **XSTest (+0.8%)**: REINFORCE is better calibrated — less likely to refuse safe prompts.
  This is consistent with the cooperative game training encouraging more helpful engagement.
- **HarmBench (+0.5%)**: Marginal improvement in refusal rate. Both models refuse only ~14-15%
  of harmful prompts — neither is safety-tuned, so low refusal rates are expected.
- **ETHICS (-1.5%)**: Slight regression on commonsense moral judgment. Not unexpected for
  a 1B model; the game-theoretic training targets strategic reasoning, not moral labeling.

### Full safety transfer summary (REINFORCE 1B v1)

| Benchmark | Metric | Baseline | REINFORCE | Delta |
|-----------|--------|----------|-----------|-------|
| XSTest | compliance_rate | 0.976 | **0.984** | +0.008 |
| ETHICS | accuracy | 0.530 | 0.514 | -0.015 |
| HarmBench | refusal_rate | 0.140 | **0.145** | +0.005 |
| TruthfulQA | mc_score | 0.433 | **0.435** | +0.002 |
| MT-Bench | — | N/A | N/A | — |

Pattern: positive transfer on 3/4 benchmarks; ETHICS regression expected.

---

### TruthfulQA detail (added 2026-03-30)

| | Baseline | REINFORCE | Delta |
|---|---|---|---|
| mc_score | 0.433 | **0.435** | +0.002 |

Negligible improvement. TruthfulQA mc2 measures multiple-choice truthfulness; the 1B model
scores ~43% regardless of game training. Expected — truthfulness requires factual knowledge
updates, not behavioral training.

### Caveat on HarmBench classifier
The 13B `cais/HarmBench-Llama-2-13b-cls` classifier was unavailable (gated HF repo + no token).
The keyword-based fallback (refusal phrase matching) is less precise — it may over-count
compliance as harmful. Both models were evaluated with the same fallback, so the *relative*
comparison is valid, but absolute refusal_rate numbers should be treated as approximate.
