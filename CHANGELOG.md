# Changelog

All notable changes to ModelDirector are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- **README install and quickstart commands were broken end to end.**
  `pip install modeldirector` 404s — the package is not on PyPI; install is
  from this repo. `pip install "modeldirector[rest]"` / `[all]` pointed at
  extras that didn't exist (now defined). `modeldirector-mcp config.yaml`
  (Quickstart step 5) had no console script behind it — the `mcp` adapter
  was only runnable via `python -m`; `modeldirector-mcp` is now a real
  entry point that requires the config path. The clone URL and Issues link
  still used the pre-move `aniketkarne-com` owner and 404'd.
- **Selector request always sent `temperature=0.0`.** Current reasoning
  models (Claude 4.7+, OpenAI reasoning models) reject non-default sampling
  params with HTTP 400, so any swap of the selector to a current model failed
  at request time. `selector.temperature` now defaults to unset and is only
  sent when explicitly configured; the shipped examples drop it.

### Changed

- **Example configs and README quickstart moved to current model ids and
  prices.** `anthropic/claude-3.5-haiku`, `anthropic/claude-3.5-sonnet` and
  `anthropic/claude-opus-4` are gone from the live OpenRouter catalog, and
  `gpt-4o-mini` is deprecated at OpenAI (removal 2027-01-14) — copy-pasting
  the old quickstart produced a 404 selector and unbuyable candidates. Now
  `openai/gpt-6-luna` (selector + cheap candidate, $0.10/$0.50 per 1M),
  `anthropic/claude-sonnet-5.5` ($2/$10), `anthropic/claude-opus-5.5`
  ($4/$20) — ids and prices verified against the OpenRouter catalog
  2026-11-14. Candidate ids are `luna`/`sonnet`/`opus`. The dated 2026-06-07
  benchmark capture is untouched historical data.
- **Version strings unified at 0.2.0.** `pyproject.toml`,
  `modeldirector.__version__` and the REST app all said 0.1.0 while the
  README advertised "v0.2" and the Laya backend had shipped; the REST app
  now reports the package version instead of a frozen literal.

### Added

- **`[project.urls]` in `pyproject.toml`** — `Homepage`, `Repository`,
  `Issues`, `Changelog` keys pointing at the canonical URLs.
- **`[rest]` and `[all]` extras.** `[rest]` lists the REST adapter's
  fastapi/uvicorn needs (also core deps, so nothing changes for existing
  installs); `[all]` aggregates `[mcp,laya,rest]`.
- **Release-hygiene tests** (`tests/test_packaging.py`): version
  consistency, extras and console scripts matching the README, no
  pre-move URLs in the README, no retired model ids in `examples/`, example
  configs load clean, and the temperature-omission contract on the selector
  transport. The retired-id list in that file is the lint rule — extend it
  as models retire.
- **CI workflow** (`.github/workflows/tests.yml`) — pytest on every push
  and pull request against Python 3.11 and 3.12, no API key needed. The
  README claimed CI ran the unit suite on every commit; nothing did.

## [0.2.0] - 2026-09-27

- Add the **Laya selector backend** (`selector.backend: laya`):
  non-autoregressive System-1 decision engine that scores every candidate
  in a single forward pass (~33 ms on GPU, $0 per call after the
  checkpoint download). See [`docs/laya.md`](docs/laya.md) for the
  trade-offs.
- Add a pluggable selector backend interface
  (`modeldirector.selectors.{llm,laya}`); the original LLM-as-judge
  backend becomes the default `llm` selector.
- Add `--selector-backend {llm,laya}` to the CLI and benchmark runner.
- Add 9 unit tests for the Laya backend (no checkpoint download required
  in CI; the Laya agent is stubbed).

## [0.1.0] - 2026-06-07

- Initial release. Stateless AI model selection engine: a configured LLM
  scores user prompts against a candidate model set, and one of three
  built-in policies (`cheapest_capable`, `highest_confidence`,
  `best_value`) picks the winner with a per-model USD cost estimate.
