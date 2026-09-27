"""Laya-backed selector (cheap, fast, non-autoregressive).

Laya is a non-autoregressive System-1 decision engine
(https://github.com/NandhaKishorM/laya). It runs all the questions in a
single forward pass (~33-38 ms on GPU, slightly longer on CPU/MPS) and
produces calibrated confidence scores.

We use Laya to pick the *single best model* out of the user's candidate set
in one shot:

  1. Build a ``choice`` question whose keys are the candidate model ids and
     whose criteria descriptions are derived from the user-provided
     ``description`` + ``strengths`` + ``capabilities``.
  2. Call ``agent.predict(state=prompt, questions=...)``.
  3. Map the returned ``probabilities`` (one per candidate) to a per-model
     ``ModelScore``. The probability × 100 becomes the ``overall`` score.
     The four-axis breakdown (``reasoning``/``coding``/``context``/
     ``creativity``) is left as ``None`` because Laya doesn't produce it -
     downstream policies can still use ``overall`` for ranking.

The Laya agent is heavy to load (it pulls a ModernBERT checkpoint), so we
cache the loaded agent as a module-level singleton keyed on the configured
``laya_model`` + ``laya_device`` + optional ``laya_max_len``. Subsequent
calls reuse the same agent instance.

Install with:    pip install "modeldirector[laya]"
Enable with:     selector.backend: laya
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

from modeldirector.config import SelectorConfig
from modeldirector.models import ModelScore

log = logging.getLogger(__name__)


class SelectorError(RuntimeError):
    """Raised when the Laya selector fails to produce a valid score response."""


# --- Module-level agent cache -------------------------------------------------
#
# Loading the Laya checkpoint (ModernBERT-large, ~421M params) takes
# several seconds and downloads ~800 MB. Reuse one instance across calls.
# The cache key is the tuple (model_id, device, max_len, use_router) so users
# can run multiple Laya checkpoints side by side if they want.

_AGENT_CACHE: dict[tuple, Any] = {}
_AGENT_LOCK = Lock()


def _load_agent(selector: SelectorConfig) -> Any:
    """Return a cached Laya ``Agent`` (or ``Router``) for the given config."""
    use_router = _wants_router(selector)
    cache_key = (
        selector.laya_model,
        selector.laya_device,
        selector.laya_max_len,
        use_router,
    )

    with _AGENT_LOCK:
        if cache_key in _AGENT_CACHE:
            return _AGENT_CACHE[cache_key]

        try:
            import laya  # noqa: F401  (probe import first for a clean error)
        except ImportError as e:
            raise SelectorError(
                "Laya backend requires the 'laya' package. "
                "Install with: pip install \"modeldirector[laya]\""
            ) from e

        if use_router:
            log.info("Loading Laya Router (models=%s, default=%s)",
                     selector.laya_model, selector.laya_device or "english")
            from laya import Router
            agent = Router(
                models={selector.laya_model: selector.laya_device or "english"},
                default=selector.laya_device or "english",
                device=selector.laya_device,
                max_loaded=1,
            )
        else:
            log.info("Loading Laya checkpoint %s on device=%s",
                     selector.laya_model, selector.laya_device or "auto")
            from laya import load
            agent = load(
                selector.laya_model,
                device=selector.laya_device,
            )

        _AGENT_CACHE[cache_key] = agent
        return agent


def _wants_router(selector: SelectorConfig) -> bool:
    """Decide whether to use a Router vs a single Agent.

    We use the Router when more than one checkpoint is configured (i.e. when
    ``laya_device`` is set to a non-default value that's actually a model
    alias). For the common single-checkpoint case we use a direct ``Agent``
    - it's a touch faster and avoids the routing pre-step.
    """
    # Heuristic: if the user has chosen the ``multilingual`` shortcut we
    # *could* use a router with english+multilingual, but the simpler path
    # is just to load the checkpoint they named. So: only use the Router
    # when ``LAYA_ROUTER=1`` is exported.
    return os.environ.get("MODELDIRECTOR_LAYA_ROUTER") == "1"


# --- Public entry point -------------------------------------------------------


def score(prompt: str, models: list, selector: SelectorConfig) -> dict[str, ModelScore]:
    """Score each candidate model with the Laya decision engine.

    Returns a mapping of model id -> ModelScore. The ``overall`` field is
    derived from Laya's calibrated probability; ``reasoning`` /
    ``coding`` / ``context`` are left as ``None`` (Laya does not provide
    them - see ``docs/laya.md``).
    """
    agent = _load_agent(selector)
    questions = _build_choice_question(models)
    payload = _predict(agent, prompt, questions, selector)

    answer = _extract_choice_answer(payload, models)
    probs: dict[str, float] = answer["probabilities"]
    explanation = answer.get("explanation", "Picked by Laya decision engine.")

    scores: dict[str, ModelScore] = {}
    for m in models:
        p = float(probs.get(m.id, 0.0))
        # Map [0,1] -> [0,100] and clamp. Laya's probabilities are already
        # calibrated so we treat them as-is - no rounding to integer tiers.
        overall = max(0, min(100, int(round(p * 100))))
        if m.id == answer["choice"]:
            note = explanation
        else:
            note = f"Laya P={p:.3f} (overall={overall})."
        scores[m.id] = ModelScore(
            id=m.id,
            overall=overall,
            reasoning=None,
            coding=None,
            context=None,
            explanation=note,
        )

    missing = [m.id for m in models if m.id not in probs]
    if missing:
        # This shouldn't happen - we built the question from these ids -
        # but surface a clear error if it does.
        raise SelectorError(
            f"Laya did not return probabilities for: {missing}. "
            f"Got keys: {sorted(probs)}"
        )

    return scores


# --- Question construction ---------------------------------------------------


def _build_choice_question(models: list) -> dict[str, Any]:
    """Build the single ``choice`` question we send to Laya.

    Laya's choice questions take a dict of ``{key: criteria_description}``.
    We use each model's ``id`` as the key (so the answer comes back in the
    same id space ModelDirector already uses) and pack the description,
    strengths, capabilities, and cost into the criteria text so Laya can
    rank them.
    """
    criteria: dict[str, str] = {}
    for m in models:
        cap = m.capabilities
        bits: list[str] = []
        if m.description:
            bits.append(m.description.strip())
        if m.strengths:
            bits.append("Strengths: " + ", ".join(m.strengths))
        bits.append(
            f"Capabilities: reasoning={cap.reasoning}/100, "
            f"coding={cap.coding}/100, context={cap.context}/100"
            + (f", creativity={cap.creativity}/100" if cap.creativity is not None else "")
        )
        bits.append(
            f"Cost: ${m.cost.input:.2f}/1M input, ${m.cost.output:.2f}/1M output. "
            f"Priority={m.priority} (lower = preferred)."
        )
        criteria[m.id] = " ".join(bits)

    return {
        "best_model": {
            "type": "choice",
            "instructions": (
                "Which candidate model is the best fit for the user's task? "
                "Consider task complexity, the model's stated strengths, and "
                "its cost. Prefer a cheap capable model over an expensive "
                "overkill unless the task clearly demands it."
            ),
            "criteria": criteria,
        }
    }


# --- Laya call ---------------------------------------------------------------


def _predict(agent: Any, prompt: str, questions: dict, selector: SelectorConfig) -> dict:
    """Run the Laya forward pass and return its full result dict."""
    kwargs: dict[str, Any] = {}
    if selector.laya_max_len is not None:
        kwargs["max_len"] = selector.laya_max_len

    try:
        # Laya's Agent.predict has slightly different kwargs than Router.predict
        # - we use ``state`` and ``questions`` for both.
        result = agent.predict(prompt, questions, **kwargs)
    except TypeError as e:
        # Fallback for older Laya versions or ONNX agents that don't accept kwargs.
        if "unexpected keyword argument" not in str(e):
            raise
        result = agent.predict(prompt, questions)
    except Exception as e:
        raise SelectorError(f"Laya predict() failed: {e}") from e

    if not isinstance(result, dict):
        raise SelectorError(
            f"Laya predict() returned non-dict: {type(result).__name__}"
        )
    if "answers" not in result:
        raise SelectorError(
            f"Laya response missing 'answers' key. Got: {sorted(result)[:10]}"
        )
    return result


def _extract_choice_answer(payload: dict, models: list) -> dict:
    """Pull the single ``choice`` answer out of Laya's result payload."""
    answers = payload["answers"]
    if not isinstance(answers, dict) or not answers:
        raise SelectorError(
            f"Laya returned no answers. Payload: {sorted(payload)[:5]}"
        )
    # We asked exactly one question. Take the first (and only) answer.
    answer = next(iter(answers.values()))
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise SelectorError(
            f"Laya answer is not a 'choice'. Got type={answer.get('type')!r}"
        )
    if "choice" not in answer or "probabilities" not in answer:
        raise SelectorError(
            f"Laya choice answer missing 'choice' or 'probabilities'. "
            f"Got keys: {sorted(answer)}"
        )
    return answer


# --- Testing helpers ---------------------------------------------------------


def _reset_cache() -> None:
    """Clear the loaded Laya agent cache. Test-only."""
    with _AGENT_LOCK:
        _AGENT_CACHE.clear()