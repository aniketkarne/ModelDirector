# ModelDirector

![Hero](docs/hero.jpg)

**Open-source, stateless AI model selection engine.**

> **Select the cheapest model that can successfully complete the task.**

> **🆕 v0.2 — New selector backend: [`laya`](docs/laya.md)**
> Run the entire selector for **$0/call** with a single forward pass
> (~33 ms on GPU) via the [Laya decision engine](https://github.com/NandhaKishorM/laya).
> No per-token cost, no API key, no telemetry. Set
> `selector.backend: laya` (or pass `--selector-backend laya`) and you're
> done. The original `llm` backend (LiteLLM-routed LLM) is still the default
> and still the right choice when you want a nuanced per-axis judge.
> Install: `pip install "modeldirector[laya]"` · Full docs: [`docs/laya.md`](docs/laya.md)

ModelDirector doesn't execute prompts and isn't a gateway. It scores a
user prompt against a configurable set of candidate models and returns the
**best pick** — with full reasoning, a per-model cost estimate, and an
auditable decision trace.

```
$ echo "Translate 'hello' to Spanish." | modeldirector select -c config.yaml
{
  "selected_model": "gpt5mini",
  "policy": "cheapest_capable",
  "scores": {
    "gpt5mini": { "overall": 92, "reasoning": 80, "coding": 85, "context": 90, "explanation": "..." },
    "sonnet":   { "overall": 95, "reasoning": 90, "coding": 92, "context": 95, "explanation": "..." },
    "opus":     { "overall": 99, "reasoning": 99, "coding": 97, "context": 99, "explanation": "..." }
  },
  "estimated_cost_usd": {
    "gpt5mini": 0.0001,
    "sonnet":   0.0020,
    "opus":     0.0100
  },
  "input_tokens": 7,
  "reason": "'gpt5mini' is the first model exceeding the threshold of 80 (overall=92, priority=1, cost=$0.15/1M in, $0.60/1M out)."
}
```

---

## The product: a model selection engine

ModelDirector is **a model selection engine**. Not a router. Not a gateway.
Not a model. It does one thing:

1. You define your candidate models (id, real cost, capabilities, strengths, description).
2. ModelDirector scores a prompt against every candidate.
3. Your configured policy picks the winner.
4. The decision, scores, and per-model USD cost estimate are returned.

The engine is **100% independent** of how your agent actually calls the
LLM. It works with anything that talks LiteLLM — OpenAI, Anthropic,
OpenRouter, Ollama, vLLM, Bedrock, Azure. The same engine powers every
adapter:

```
                          ModelDirector Engine
                  ┌────────────────────────────────────┐
                  │  Config loader  ·  Selector        │
                  │  Policy engine  ·  Cost estimator  │
                  └────────────────────────────────────┘
                                    │
        ┌──────────────┬────────────┼────────────┬──────────────┐
        │              │            │            │              │
   Python SDK         CLI         REST         MCP      Your adapter
```

Pick the surface that fits. Same engine, same policies, same result.

---

## Why

Most agents ship with a single hardcoded model. Two problems:

1. **Cost** — using Opus for a typo-fix is wasteful.
2. **Lock-in** — switching providers means rewriting call sites.

ModelDirector sits *before* the model call. The same `director.select(prompt)`
call works for OpenAI, Anthropic, local Ollama, or a self-hosted mix.
The selection is auditable (`reason`), deterministic (configurable seed),
explainable (full per-model scores), and **transparent** — every output
includes a per-model USD cost estimate so the trade-off is visible to the
caller.

---

## Install

```bash
# Core (SDK + CLI)
pip install modeldirector

# With REST adapter
pip install "modeldirector[rest]"

# With MCP adapter (for Claude Desktop, Continue, etc.)
pip install "modeldirector[mcp]"

# With the Laya decision engine backend (cheap, self-hosted selector)
pip install "modeldirector[laya]"

# Everything
pip install "modeldirector[all]"
```

Or from this repo:

```bash
git clone https://github.com/aniketkarne-com/ModelDirector
cd ModelDirector
uv sync --all-extras
```

> **Two selector backends ship today:** `llm` (default — a configured LLM
> scores every candidate) and `laya` (the Laya decision engine — a
> non-autoregressive System-1 model that scores every candidate in a
> single forward pass, ~33 ms on GPU, $0/call). See
> [`docs/laya.md`](docs/laya.md) for the trade-offs and
> `examples/config-laya.yaml` for a working config.

---

## Quickstart

1. **Write a config** (`config.yaml`):

    ```yaml
    selector:
      provider: openrouter
      model: anthropic/claude-3.5-haiku  # small, fast, JSON-reliable
      temperature: 0.0

    policy:
      type: cheapest_capable
      threshold: 80

    models:
      - id: gpt5mini                 # model id = model name, no tier labels
        name: openai/gpt-4o-mini
        display_name: GPT-4o mini
        description: |
          OpenAI's small, fast, low-cost model. Good for classification,
          routing, short summarisation, and basic Q&A. Weaker at long-
          horizon reasoning and complex code generation.
        strengths:                    # structured tags the selector can rely on
          - classification
          - short_summarisation
          - simple_qa
        capabilities: { reasoning: 65, coding: 70, context: 70 }
        priority: 1
        cost:                          # USD per 1M tokens (input / output)
          input: 0.15
          output: 0.60

      - id: sonnet
        name: anthropic/claude-3.5-sonnet
        display_name: Claude 3.5 Sonnet
        description: |
          Anthropic's mid-tier model. Strong at coding, architecture,
          and long-context reasoning. 200k context window.
        strengths:
          - coding
          - architecture
          - refactoring
          - long_context
        capabilities: { reasoning: 88, coding: 92, context: 95 }
        priority: 2
        cost:
          input: 3.00
          output: 15.00

      - id: opus
        name: anthropic/claude-opus-4
        display_name: Claude Opus 4
        description: |
          Anthropic's frontier model. Best reasoning, complex multi-step
          planning, nuanced code generation.
        strengths:
          - hard_reasoning
          - complex_coding
          - architecture_design
        capabilities: { reasoning: 95, coding: 95, context: 99 }
        priority: 3
        cost:
          input: 15.00
          output: 75.00
    ```

2. **Select a model for any prompt:**

    ```bash
    # CLI
    modeldirector select -c config.yaml -p "Summarise this article."

    # From a file
    modeldirector select -c config.yaml -f prompt.txt

    # From stdin
    echo "Write a haiku" | modeldirector select -c config.yaml
    ```

3. **Or use the Python SDK:**

    ```python
    from modeldirector import ModelDirector, load_config

    director = ModelDirector(load_config("config.yaml"))
    result = director.select("Explain quantum entanglement in one paragraph.")
    print(result.selected_model, "-", result.reason)
    print("  cost:", result.estimated_cost_usd[result.selected_model], "USD")
    ```

4. **Or call the REST API:**

    ```bash
    MODELDIRECTOR_CONFIG=config.yaml uvicorn modeldirector.adapters.rest:app
    curl -X POST http://localhost:8000/select \
         -H 'content-type: application/json' \
         -d '{"prompt": "Fix the typo in this sentence."}'
    ```

5. **Or expose it as an MCP tool:**

    ```bash
    modeldirector-mcp config.yaml   # talks MCP stdio to Claude Desktop etc.
    ```

---

## How it works

![How it works](docs/how-it-works.jpg)

The selector is a small, fast LLM (e.g. `claude-3.5-haiku`, `gpt-4o-mini`,
`qwen3-32b`) that gets a structured JSON prompt with your task, every
candidate's profile (description, capabilities, **strengths**, and **real
cost per 1M tokens**), and a strict scoring rubric. It returns scores only —
the decision logic is yours.

**Three built-in policies:**

| Policy                | Picks                                                       |
| --------------------- | ----------------------------------------------------------- |
| `cheapest_capable`    | First model (by priority, then by input cost) whose `overall >= threshold` |
| `highest_confidence`  | The model with the highest overall score                    |
| `best_value`          | Best `overall / cost.input` ratio (confidence per dollar)  |

Custom policies subclass `Policy` and override `apply(scores, models)`.

---

## Configuration reference

### `selector`

The selector is whatever ModelDirector asks "which of these candidate
models should handle this prompt?" Two backends ship today:

* **`llm`** (default) — asks a configurable LLM via LiteLLM. Slower
  (~0.5-3 s), costs per-token, returns per-axis scores
  (`reasoning` / `coding` / `context` / `creativity`).
* **`laya`** — the [Laya decision engine](docs/laya.md), a
  non-autoregressive System-1 model. Single forward pass (~33 ms on
  GPU), $0 per call after the initial checkpoint download. Returns a
  single calibrated probability per candidate, which becomes the
  `overall` score.

| Field         | Required | Default                        | Notes |
| ------------- | -------- | ------------------------------ | ----- |
| `backend`     | no       | `"llm"`                        | `"llm"` or `"laya"`. See above. |
| `provider`    | llm      | `openrouter`                   | LiteLLM provider, e.g. `openrouter`, `openai`, `ollama`. Ignored when `backend: laya`. |
| `model`       | llm      | —                              | Model name (LiteLLM format). Ignored when `backend: laya`. |
| `api_key`     | no       | env (`${PROVIDER}_API_KEY`)    | Falls back to the provider's env var. Ignored when `backend: laya`. |
| `api_base`    | no       | —                              | Override the API base URL. Ignored when `backend: laya`. |
| `temperature` | no       | `0.0`                          | Sampling temperature. |
| `max_tokens`  | no       | —                              | Cap on the selector's response. |
| `laya_model`  | no       | `convaiinnovations/laya`       | Laya checkpoint id. Use `multilingual` for 100+ languages and longer context. Ignored when `backend: llm`. |
| `laya_device` | no       | auto                           | `"cpu"`, `"cuda"`, `"mps"`. Ignored when `backend: llm`. |
| `laya_max_len`| no       | —                              | Override the per-call token budget for the Laya forward pass. |

### `policy`

| Field      | Type    | Default              | Notes |
| ---------- | ------- | -------------------- | ----- |
| `type`     | enum    | `cheapest_capable`   | One of `cheapest_capable`, `highest_confidence`, `best_value` |
| `threshold`| int     | `85`                 | Confidence threshold for `cheapest_capable` |
| `cost_per_million` | bool | `true`         | Hint that costs are USD per 1M tokens (affects reports only) |

### `models[]`

| Field          | Required | Default     | Notes |
| -------------- | -------- | ----------- | ----- |
| `id`           | yes      | —           | Stable identifier used in output. **Use model names** (e.g. `gpt5mini`, `sonnet`) — not tier labels like `cheap`/`premium`. The engine doesn't know what tiers mean. |
| `name`         | yes      | —           | LiteLLM-format model name (the actual call) |
| `display_name` | no       | `name`      | Human-friendly label |
| `description`  | no       | `""`        | **Strongly recommended.** Free-form context for the selector — small LLMs may not recognise model names. |
| `strengths`    | no       | `[]`        | **Strongly recommended.** Structured task-type tags (e.g. `coding`, `architecture`, `long_context`). More reliable for small selectors than relying on the model name. |
| `capabilities` | yes      | —           | `reasoning`, `coding`, `context` required, `creativity` optional (0-100) |
| `priority`     | no       | `1`         | Lower = preferred. Tie-breaker for `cheapest_capable`. |
| `cost`         | yes      | —           | Per-1M-token cost in USD. `cost: {input: 0.15, output: 0.60}` for new configs. A bare number is accepted for back-compat and treated as `input=output=x`. |

---

## Interfaces

| Interface      | Use it for                                          | Entry point |
| -------------- | --------------------------------------------------- | ----------- |
| **Python SDK** | Embedding in an app                                 | `from modeldirector import ModelDirector` |
| **CLI**        | Shell scripts, cron jobs                            | `modeldirector select -c config.yaml -p "..."` |
| **REST**       | Microservices, polyglot stacks                      | `uvicorn modeldirector.adapters.rest:app` |
| **MCP**        | Claude Desktop, Continue, Roo Code                  | `modeldirector-mcp config.yaml` |

The same engine powers all four. Adapters are thin translation layers.

---

## Tests

```
$ pytest tests/ -q -m "not integration"
47 passed in 0.06s
```

Includes:

- **43 unit tests** for the policy engine, config loader, cost estimator,
  and selector (no network) — `pytest tests/`
- **4 integration tests** that hit OpenRouter with a real selector LLM —
  `pytest tests/test_integration.py` (skipped automatically if
  `OPENROUTER_API_KEY` isn't set)

CI on every commit runs the unit suite. The integration suite is opt-in
(`pytest -m integration`).

---

## Benchmark

`benchmarks/run_benchmark.py` runs 30 real tasks (translation, Q&A, code
generation, architecture design, prose) through the full ModelDirector
pipeline and reports:

- which model was selected per task
- per-task USD cost
- per-category cost savings vs an "always-opus" baseline
- selector latency (p50 / p95 / max)

It supports **both selector backends** — pass `--selector-backend laya`
to run the same 30-task battery against Laya. The output JSON records
which backend was used so the runs are comparable.

Run it:

```bash
# LLM backend (LiteLLM-routed judge, costs per token)
OPENROUTER_API_KEY=sk-or-... python -m benchmarks.run_benchmark \\
    -o benchmarks/output/results-llm.json

# Laya backend (local, $0/call, single forward pass)
pip install "modeldirector[laya]"
python -m benchmarks.run_benchmark --selector-backend laya \\
    -o benchmarks/output/results-laya.json
```

Latest committed result: [`benchmarks/output/results.json`](benchmarks/output/results.json).
Re-render below for the backend that produced it.

### LLM backend

> Captured 2026-06-07 with `selector: anthropic/claude-3.5-haiku` and a
> 3-model candidate set (`gpt5mini = gpt-4o-mini`,
> `sonnet = claude-3.5-sonnet`, `opus = claude-opus-4`) with real
> per-1M-token USD costs. 30 tasks, 8 categories.

**Headline numbers**

| Metric                                  | Value        |
| --------------------------------------- | ------------ |
| Backend                                 | `llm` (Claude 3.5 Haiku judge) |
| Tasks completed                         | **30 / 30**  |
| Errors                                  | **0**        |
| **Mean cost savings vs always-opus**    | **100.0 %**  |
| Tasks that picked `gpt5mini`            | 11 / 30      |
| Tasks that picked `sonnet`              | 19 / 30      |
| Tasks that picked `opus`                | 0 / 30       |
| Selector p50 latency                    | 3.9 s        |
| Selector p95 latency                    | 4.5 s        |
| Selector max latency                    | 5.1 s        |

**Per-task model picks**

| Category        | n | Mean savings | Picked               |
| --------------- | - | ------------ | -------------------- |
| trivial         | 5 | **100.0 %**  | 4× gpt5mini, 1× sonnet |
| simple_qa       | 5 | **100.0 %**  | 4× gpt5mini, 1× sonnet |
| summarisation   | 3 | **100.0 %**  | 2× gpt5mini, 1× sonnet |
| light_coding    | 4 | **100.0 %**  | 4× sonnet            |
| medium_coding   | 4 | **100.0 %**  | 4× sonnet            |
| reasoning       | 3 | **100.0 %**  | 1× gpt5mini, 2× sonnet |
| hard_coding     | 4 | **100.0 %**  | 4× sonnet            |
| writing         | 2 | **100.0 %**  | 2× sonnet            |

**What this means in real USD.** Per 1M input tokens:

| Model        | $/1M in | $/1M out | vs opus (input)   |
| ------------ | ------- | -------- | ----------------- |
| gpt5mini     | $0.15   | $0.60    | **100× cheaper**  |
| sonnet       | $3.00   | $15.00   | **5× cheaper**    |
| opus         | $15.00  | $75.00   | baseline          |

For a typical 200-token prompt:

- `gpt5mini` ≈ $0.00003 input / $0.00006 output
- `sonnet`   ≈ $0.0006 input / $0.003 output
- `opus`     ≈ $0.003 input / $0.015 output

**Why no `opus` picks?** Sonnet's overall score cleared the configured
threshold (80) for every task in the battery. The selector correctly
identified that the extra cost of the frontier model wasn't justified for
this task mix. Lower the threshold to 60 or tighten the `description`/
`strengths` on opus, and the selector will start to pick opus for the
hardest tasks.

**Raw data:** [`benchmarks/output/results.json`](benchmarks/output/results.json) — every score, every reason, every latency, every USD estimate.

### Laya backend

> **TODO — first Laya run on the same 30-task battery.**
>
> Reproduce locally with:
>
> ```bash
> pip install "modeldirector[laya]"
> python -m benchmarks.run_benchmark --selector-backend laya \\
>     -o benchmarks/output/results-laya.json
> ```
>
> Then commit `results-laya.json` and update the headline table below.

Expected ballpark (from the [Laya decision engine benchmarks](https://github.com/NandhaKishorM/laya#benchmarks),
sourced on the upstream repo, **not** measured on this benchmark yet):

| Metric                                  | Expected          | Source                                         |
| --------------------------------------- | ----------------- | ---------------------------------------------- |
| Backend                                 | `laya` (single forward pass) | —                                              |
| Per-call latency (GPU)                  | ~33–42 ms         | Laya repo p50/p95 on the English checkpoint    |
| Per-call latency (CPU / MPS)             | ~80–150 ms        | Same forward pass, no GPU                      |
| Per-call cost                           | **$0.00**         | Runs locally after checkpoint download         |
| First-call one-time                     | ~800 MB checkpoint download | One-time, then cached                         |

The Laya backend reports the same `overall` score per candidate (mapped
from its calibrated probability, × 100), so the **model picks and savings
columns should look very similar to the LLM run above** for most tasks.
The big difference is the latency (33 ms vs 3.9 s) and the per-call cost
($0.00 vs ~$0.001-$0.005 for Haiku). When the Laya run is committed,
its numbers slot into the table below.

| Metric                                  | Value        |
| --------------------------------------- | ------------ |
| Backend                                 | `laya`       |
| Tasks completed                         | _awaiting first run_ |
| Errors                                  | _awaiting first run_ |
| Mean cost savings vs always-opus        | _awaiting first run_ |
| Tasks that picked `gpt5mini`            | _awaiting first run_ |
| Tasks that picked `sonnet`              | _awaiting first run_ |
| Tasks that picked `opus`                | _awaiting first run_ |
| Selector p50 latency                    | _awaiting first run_ |
| Selector p95 latency                    | _awaiting first run_ |

---

## Architecture

```
modeldirector/
├── modeldirector/
│   ├── config.py            # Pydantic config models (Cost, ModelProfile, PolicyConfig, SelectorConfig)
│   ├── loader.py            # YAML / dict loader, ${ENV} expansion
│   ├── models.py            # Public types: ModelScore, SelectionResult (with estimated_cost_usd)
│   ├── policy.py            # Policy engine (3 built-ins + custom)
│   ├── selector.py          # Orchestrator: dispatch to llm or laya backend, apply policy, attach cost
│   ├── selectors/           # Pluggable selector backends
│   │   ├── llm.py           #   - LLM-as-judge (default; LiteLLM, per-axis scores)
│   │   └── laya.py          #   - Laya decision engine (single forward pass, $0/call)
│   └── adapters/
│       ├── cli.py           # click-based CLI (with --selector-backend override)
│       ├── rest.py          # FastAPI server
│       └── mcp.py           # FastMCP tool
├── benchmarks/
│   └── run_benchmark.py     # 30-task real-LLM benchmark, both backends supported
├── examples/
│   ├── config.yaml          # ready-to-use config (LLM backend)
│   └── config-laya.yaml     # ready-to-use config (Laya backend)
├── tests/                   # 56 unit tests (47 selector + 9 laya backend)
├── docs/
│   ├── hero.jpg             # README hero (top)
│   └── how-it-works.jpg     # "How it works" section diagram
├── prd.md
├── pyproject.toml
└── README.md
```

The engine is the product. Adapters are derived. Adding a new interface
(Discord bot, Slack command, custom agent, etc.) is a matter of
constructing a `ModelDirector` and translating the input/output to the
adapter's protocol.

---

## Design principles

- **Stateless** — no database, no storage, no learning, no telemetry. Every
  request is independent.
- **Model-agnostic** — never hardcodes model names. You define the candidate
  set; ModelDirector works with any provider LiteLLM supports (OpenAI,
  Anthropic, Ollama, vLLM, Bedrock, Azure, etc.).
- **Cost-transparent** — `cost` is a first-class field, `estimated_cost_usd`
  is in every response. The trade-off is visible, not hidden.
- **Configuration first** — every threshold, every policy, every model is
  config. No magic defaults that hide the cost trade-off.
- **Auditable** — the response includes per-model scores, per-model cost,
  the reason, and the policy that was applied. You always know *why* a
  model was picked.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Contributing

PRs welcome. Bug reports and feature requests go in
[GitHub Issues](https://github.com/aniketkarne-com/ModelDirector/issues).

For substantial changes, open an issue first to discuss the design — this
project prizes a small, stable API surface.
