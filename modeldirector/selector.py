"""High-level ModelDirector orchestrator.

The selector backend is chosen by ``Config.selector.backend``:

  * ``"llm"``  (default) - delegates to ``modeldirector.selectors.llm.score``
  * ``"laya"``            - delegates to ``modeldirector.selectors.laya.score``

The orchestrator's only job is to:

  1. Count the input tokens (used for the ``estimated_cost_usd`` field).
  2. Dispatch to the configured backend.
  3. Apply the policy to the resulting scores.
  4. Attach the per-model USD cost estimate to the SelectionResult.

The backends themselves know nothing about policies or cost estimation.
"""

from __future__ import annotations

import logging
from typing import Any

from modeldirector.config import Config, SelectorBackend
from modeldirector.models import SelectionResult
from modeldirector.policy import build_policy, select_model
from modeldirector.selectors.llm import SelectorError, _extract_json

log = logging.getLogger(__name__)


# Backwards-compat re-exports so callers that did
#   from modeldirector.selector import SelectorError
# keep working.
__all__ = ["ModelDirector", "SelectorError", "_extract_json"]


# --- Token estimation ---------------------------------------------------------
#
# Best-effort token count. Used only to compute `estimated_cost_usd` in the
# output - this is an estimate, not a bill. We try tiktoken (OpenAI's
# tokenizer, ~2MB dep) for a real count, and fall back to a 4-chars-per-token
# heuristic if it's not installed or doesn't know the model.

try:
    import tiktoken  # type: ignore[import-not-found]

    def _count_tokens(text: str, model_hint: str | None = None) -> int:
        try:
            if model_hint:
                enc = tiktoken.encoding_for_model(model_hint)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

except ImportError:  # pragma: no cover

    def _count_tokens(text: str, model_hint: str | None = None) -> int:
        # ~4 chars per token is a common rule of thumb.
        return max(1, len(text) // 4)


# --- Errors -------------------------------------------------------------------


class ModelDirectorError(RuntimeError):
    """Generic ModelDirector failure. Use ``SelectorError`` for selector issues."""


# --- Backend dispatch ---------------------------------------------------------


def _resolve_backend(name: SelectorBackend):
    """Return the backend module for the given name. Import lazily so the
    ``laya`` dependency stays optional."""
    if name == "llm":
        from modeldirector.selectors import llm as backend

        return backend
    if name == "laya":
        from modeldirector.selectors import laya as backend

        return backend
    raise ModelDirectorError(f"Unknown selector backend: {name!r}")


# --- Public entry point -------------------------------------------------------


class ModelDirector:
    """High-level entry point.  Load a config, call ``.select(prompt)``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._backend = _resolve_backend(config.selector.backend)

    # --- public API ---

    def select(
        self, prompt: str, *, assumed_output_tokens: int | None = None
    ) -> SelectionResult:
        """Pick the best model for the given prompt.  Returns a SelectionResult.

        `assumed_output_tokens` lets the caller override the output-token
        estimate used in `estimated_cost_usd` (defaults to mirroring the
        input-token count, which is a reasonable assumption for chat-style
        tasks). Set to 0 if you only care about input cost.
        """
        scores, input_tokens = self._score(prompt)
        estimated_cost = _estimate_cost_usd(
            models=self.config.models,
            input_tokens=input_tokens,
            assumed_output_tokens=(
                assumed_output_tokens
                if assumed_output_tokens is not None
                else input_tokens
            ),
        )
        result = select_model(
            scores=scores,
            models=self.config.models,
            policy=build_policy(self.config.policy),
        )
        return result.model_copy(
            update={
                "estimated_cost_usd": estimated_cost,
                "input_tokens": input_tokens,
            }
        )

    def score(self, prompt: str) -> dict:
        """Public: return raw per-model scores without applying a policy."""
        scores, _ = self._score(prompt)
        return scores

    # --- internals ---

    def _score(self, prompt: str) -> tuple[dict, int]:
        # Token count for cost estimation - best-effort, only meaningful for
        # the LLM backend's ``model`` hint. Laya has its own tokenizer.
        model_hint = self.config.selector.model if self.config.selector.backend == "llm" else None
        input_tokens = _count_tokens(prompt, model_hint=model_hint)
        scores = self._backend.score(
            prompt=prompt,
            models=self.config.models,
            selector=self.config.selector,
        )
        return scores, input_tokens


# --- Cost estimation ---------------------------------------------------------


def _estimate_cost_usd(
    *,
    models: list,
    input_tokens: int,
    assumed_output_tokens: int,
) -> dict[str, float]:
    """Compute a per-model USD cost estimate for the given input size.

    Returns a dict keyed by model id. Models with missing or zero cost are
    included with 0.0 so the output is always complete.
    """
    out: dict[str, float] = {}
    for m in models:
        in_cost = float(getattr(m.cost, "input", 0.0))
        out_cost = float(getattr(m.cost, "output", 0.0))
        usd = (input_tokens / 1_000_000.0) * in_cost + (
            assumed_output_tokens / 1_000_000.0
        ) * out_cost
        out[m.id] = round(usd, 6)
    return out