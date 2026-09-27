# Using the Laya selector backend

ModelDirector ships with **two selector backends**:

| Backend | Cost | Latency | Per-axis scores | When to use |
|---|---|---|---|---|
| `llm` (default) | ~$0.001-$0.05 / call | ~500-3000 ms | Yes (reasoning/coding/context/creativity) | Production routing where you want a calibrated, opinionated judge |
| `laya` | $0.00 / call (self-hosted) | ~33-100 ms | No (single probability per model) | Cheap, fast routing for simple/medium tasks; production use of LLM selectors for hard reasoning |

`laya` is the **Laya decision engine** from
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — a
non-autoregressive System-1 model based on ModernBERT that scores every
candidate model in a **single forward pass** (~33 ms on GPU, ~80-150 ms on
CPU/MPS). The probabilities are calibrated (trained against strictly
proper scoring rules), so a high probability is a meaningful signal.

## Install

```bash
pip install "modeldirector[laya]"
```

This pulls in the `laya` package plus its PyTorch / Transformers
dependencies. The first time you load a checkpoint, Laya downloads the
weights (~800 MB for the English root checkpoint, ~647 MB for
`laya-multilingual`).

## Configure

Set `selector.backend: laya` in your config:

```yaml
selector:
  backend: laya
  laya_model: convaiinnovations/laya   # English root checkpoint
  # or:
  # laya_model: multilingual             # 100+ languages, longer context
  laya_device: null                     # auto-detect (cuda / mps / cpu)
  laya_max_len: null                    # per-call token budget override
```

All `selector.provider`, `selector.model`, `selector.api_key` fields are
**ignored** when `backend: laya`. See `examples/config-laya.yaml` for a
complete example.

## Override from the CLI

You can also override the backend without editing the config:

```bash
echo "Translate hello to French" | \
  modeldirector select -c examples/config.yaml --selector-backend laya
```

## What Laya gives back

For every candidate model, Laya returns a single probability that becomes
the `overall` score:

```json
{
  "selected_model": "gpt5mini",
  "scores": {
    "gpt5mini": { "overall": 92, "explanation": "Picked by Laya decision engine." },
    "sonnet":   { "overall": 6,  "explanation": "Laya P=0.06 (overall=6)." },
    "opus":     { "overall": 2,  "explanation": "Laya P=0.02 (overall=2)." }
  }
}
```

The per-axis fields (`reasoning`, `coding`, `context`, `creativity`) are
**left as `null`** because Laya does not produce them. If you need them,
use the `llm` backend.

## How the scoring works

For each prompt, ModelDirector builds a single Laya `choice` question:

```python
{
    "best_model": {
        "type": "choice",
        "instructions": "Which candidate model is the best fit for the user's task?",
        "criteria": {
            "gpt5mini": "OpenAI's small, fast, low-cost model. ...",
            "sonnet":   "Anthropic's mid-tier model. Strong at coding. ...",
            "opus":     "Anthropic's frontier model. Best reasoning. ...",
        }
    }
}
```

The `criteria` text packs each model's user-provided `description` +
`strengths` + `capabilities` + `cost` into a single string. The Laya
checkpoint reads that, runs one forward pass, and returns:

```python
{
    "choice": "gpt5mini",
    "probabilities": {"gpt5mini": 0.92, "sonnet": 0.06, "opus": 0.02},
    "confidence": 0.86,
}
```

ModelDirector maps each probability to an integer `overall` score
(`prob × 100`, clamped to `[0, 100]`) and applies your configured
`policy` as usual.

## Performance

Real measurements from the [Laya repo benchmarks](https://github.com/NandhaKishorM/laya#benchmarks):

| Setting | Latency (per call) | Cost (per call) |
|---|---|---|
| Laya on GPU | **38 ms** (p95: 42 ms) | $0.00 |
| Laya on CPU/MPS | ~80-150 ms | $0.00 |
| LiteLLM (Claude 3.5 Haiku) | ~500-1500 ms | ~$0.001-$0.005 |
| LiteLLM (Claude Opus 4) | ~1500-3000 ms | ~$0.05-$0.20 |

For 80% of routing decisions, the Laya backend is **free and instant**
once the checkpoint is loaded. The LLM backend is for the 20% of edge
cases where you genuinely need a more nuanced judge.

## Limitations

- **No per-axis scores.** The four capability axes are `null` with the
  Laya backend. The `cheapest_capable` policy works fine on `overall`
  alone; `highest_confidence` works the same way; `best_value` only
  uses `overall` and `cost.input`, so it works too.
- **First-call checkpoint download.** The first request after install
  downloads ~800 MB. Subsequent requests are instant.
- **Context length.** The default English Laya checkpoint has a 512
  token context. Use `laya-multilingual` (set `laya_model:
  multilingual`) for up to 8,192 tokens and 100+ languages.
- **Tied with the LLM backend on calibration.** Laya's probabilities
  are calibrated (RLCD-trained) but they're still a System-1 decision —
  they can be wrong. For a one-in-a-million critical routing decision,
  the LLM backend is still the better choice.

## Run the benchmark

The existing benchmark script works with both backends:

```bash
# LLM backend (needs OPENROUTER_API_KEY)
OPENROUTER_API_KEY=sk-or-... .venv/bin/python -m benchmarks.run_benchmark

# Laya backend (no API key needed)
.venv/bin/python -m benchmarks.run_benchmark --selector-backend laya
```

The output JSON now records the backend so you can compare latency and
savings side by side.