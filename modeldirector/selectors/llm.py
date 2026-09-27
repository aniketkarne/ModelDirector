"""LLM-backed selector (the original ModelDirector selector).

The selector:
  1. Builds a JSON-only prompt that lists the user task, each model's profile
     (id, name, display name, description, capabilities, strengths, priority,
     cost), and the scoring rubric.
  2. Calls the configured LLM via LiteLLM.
  3. Validates that the response is JSON matching the ``ModelScore`` shape.
  4. Returns the parsed scores; the Policy engine then makes the final pick.

This is the default backend (``selector.backend: llm``). It is the slow and
expensive path: it makes a real LLM call per request and gets per-axis scores
back (reasoning / coding / context / creativity). For most routing decisions
the cheaper ``laya`` backend is sufficient.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import litellm
from pydantic import ValidationError

from modeldirector.config import SelectorConfig
from modeldirector.models import ModelScore

log = logging.getLogger(__name__)

# Tunable cap on the prompt we send to the selector.  Keeps selector cost
# predictable even with very large user prompts.
_MAX_PROMPT_CHARS = 8_000


class SelectorError(RuntimeError):
    """Raised when the selector model fails to produce a valid score response."""


# --- Public entry point -------------------------------------------------------


def score(prompt: str, models: list, selector: SelectorConfig) -> dict[str, ModelScore]:
    """Score each candidate model for the given prompt using the configured LLM.

    Returns a mapping of model id -> ModelScore. The four-axis breakdown
    (reasoning / coding / context / creativity) is filled in by the LLM.
    """
    selector_prompt = _build_selector_prompt(prompt, models)
    raw = _call_selector(selector_prompt, selector)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict) or "models" not in parsed:
        raise SelectorError(
            f"Selector response missing 'models' key. Got: {parsed!r}"
        )
    models_field = parsed["models"]
    if not isinstance(models_field, list):
        raise SelectorError(f"'models' must be a list, got {type(models_field).__name__}")

    scores: dict[str, ModelScore] = {}
    for entry in models_field:
        try:
            s = ModelScore.model_validate(entry)
        except ValidationError as e:
            raise SelectorError(
                f"Selector returned an invalid score entry: {entry!r} ({e})"
            ) from e
        scores[s.id] = s

    missing = [m.id for m in models if m.id not in scores]
    if missing:
        raise SelectorError(
            f"Selector did not return scores for: {missing}. "
            f"Got: {sorted(scores)}"
        )

    return scores


# --- LiteLLM call -------------------------------------------------------------


def _call_selector(prompt: str, sel: SelectorConfig) -> str:
    """Send the selector prompt to the configured LLM and return its raw text reply."""
    # Build the LiteLLM model name.  Always route through the configured
    # provider unless the model string already starts with that provider.
    assert sel.model is not None  # validated at config load time
    first_segment = sel.model.split("/", 1)[0]
    if first_segment == sel.provider:
        model_name = sel.model
    else:
        model_name = f"{sel.provider}/{sel.model}"

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ModelDirector, a routing model. You score how well "
                    "each candidate model can handle a user task. Always reply "
                    "with strict JSON. No prose, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": sel.temperature,
    }
    if sel.api_key:
        kwargs["api_key"] = sel.resolved_api_key()
    if sel.api_base:
        kwargs["api_base"] = sel.api_base
    if sel.max_tokens:
        kwargs["max_tokens"] = sel.max_tokens

    try:
        resp = litellm.completion(**kwargs)
    except Exception as e:
        raise SelectorError(f"Selector LLM call failed: {e}") from e

    try:
        return resp.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError, KeyError) as e:
        raise SelectorError(f"Malformed LLM response: {resp!r}") from e


# --- Prompt construction -----------------------------------------------------


def _build_selector_prompt(user_prompt: str, models: list) -> str:
    """Build the selector prompt sent to the LLM."""
    # Truncate the user prompt if it would blow up the selector's context.
    truncated = user_prompt
    if len(truncated) > _MAX_PROMPT_CHARS:
        truncated = truncated[:_MAX_PROMPT_CHARS] + "\n\n[... truncated for length ...]"

    model_blocks = []
    for m in models:
        cap = m.capabilities
        block = {
            "id": m.id,
            "name": m.name,
            "display_name": m.display_name or m.name,
            "description": m.description or "(no description provided)",
            "strengths": list(m.strengths) if m.strengths else [],
            "capabilities": {
                "reasoning": cap.reasoning,
                "coding": cap.coding,
                "context": cap.context,
                **({"creativity": cap.creativity} if cap.creativity is not None else {}),
            },
            "priority": m.priority,
            "cost_per_1m_tokens_usd": {
                "input": m.cost.input,
                "output": m.cost.output,
            },
        }
        model_blocks.append(block)

    schema_hint = json.dumps(
        {
            "models": [
                {
                    "id": "<model id>",
                    "overall": "<0-100 weighted score>",
                    "reasoning": "<0-100>",
                    "coding": "<0-100>",
                    "context": "<0-100>",
                    "creativity": "<0-100 or omit>",
                    "explanation": "<one short sentence>",
                }
            ]
        },
        indent=2,
    )

    return (
        "Score each candidate model for the user task below. "
        "Return JSON only, matching the schema exactly.\n\n"
        "## User task\n"
        f"{truncated}\n\n"
        "## Candidate models\n"
        f"{json.dumps(model_blocks, indent=2)}\n\n"
        "## Scoring rubric\n"
        "- `overall`: weighted score (0-100).  Higher is more suitable.\n"
        "- `reasoning`, `coding`, `context`, `creativity`: per-axis fit 0-100.\n"
        "- For each model, weigh its `description` and `strengths` heavily - "
        "you may not recognise the model name.\n"
        "- Be honest: a cheap model is usually fine for a cheap task.\n\n"
        "## Required output schema\n"
        f"{schema_hint}\n"
    )


# --- JSON extraction ---------------------------------------------------------


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Handles:
      * raw JSON
      * JSON wrapped in ```json ... ``` fences
      * JSON with stray prose before/after
    """
    text = text.strip()

    # Strip code fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Raw parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the outermost { ... } block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise SelectorError(f"Could not parse selector JSON: {e}\nText was:\n{text}") from e

    raise SelectorError(f"Could not find JSON in selector response:\n{text}")