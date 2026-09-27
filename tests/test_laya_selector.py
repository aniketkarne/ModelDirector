"""Tests for the Laya-backed selector. Stubs Laya entirely - no model download.

These tests use a fake ``Agent`` (an object with a ``predict`` method) that
gets injected into the module via ``patch.object`` on the module-level
``_load_agent`` helper. That way we exercise the real Laya selector code
(question construction, probability-to-score mapping, error handling)
without needing the actual Laya checkpoint on disk.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from modeldirector.config import (
    Capabilities,
    Cost,
    ModelProfile,
    SelectorConfig,
)
from modeldirector.selectors import laya as laya_backend
from modeldirector.selectors.laya import SelectorError, score


# --- Fixtures ----------------------------------------------------------------


def _models() -> list[ModelProfile]:
    return [
        ModelProfile(
            id="gpt5mini",
            name="gpt-4o-mini",
            description="Small, fast, cheap.",
            strengths=["classification", "short_qa"],
            capabilities=Capabilities(reasoning=70, coding=75, context=70),
            priority=1,
            cost=Cost(input=0.15, output=0.60),
        ),
        ModelProfile(
            id="sonnet",
            name="claude-3.5-sonnet",
            description="Mid-tier, strong at coding.",
            strengths=["coding", "architecture"],
            capabilities=Capabilities(reasoning=90, coding=92, context=92),
            priority=2,
            cost=Cost(input=3.00, output=15.00),
        ),
        ModelProfile(
            id="opus",
            name="claude-opus-4",
            description="Frontier model for hard tasks.",
            strengths=["hard_reasoning", "complex_coding"],
            capabilities=Capabilities(reasoning=99, coding=97, context=99),
            priority=3,
            cost=Cost(input=15.00, output=75.00),
        ),
    ]


def _selector(**overrides) -> SelectorConfig:
    base = SelectorConfig(backend="laya", model=None)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _fake_agent(answer_choice: str, probabilities: dict[str, float]) -> Any:
    """Return a stub object that quacks like laya.Agent (has .predict)."""

    class _Stub:
        def predict(self, state, questions, **kwargs):
            assert isinstance(state, str)  # state must be the user prompt
            assert isinstance(questions, dict)  # question schema
            return {
                "answers": {
                    "best_model": {
                        "type": "choice",
                        "choice": answer_choice,
                        "probabilities": probabilities,
                        "confidence": 0.9,
                    }
                },
                "routing": {"model": "english", "task": "classification"},
            }

    return _Stub()


# --- Tests -------------------------------------------------------------------


def test_score_maps_probabilities_to_overall():
    """Each model's probability * 100 becomes its overall score."""
    models = _models()
    sel = _selector()
    agent = _fake_agent(
        answer_choice="sonnet",
        probabilities={"gpt5mini": 0.05, "sonnet": 0.70, "opus": 0.25},
    )
    with patch.object(laya_backend, "_load_agent", return_value=agent):
        scores = score("Build a small REST API", models, sel)

    assert set(scores) == {"gpt5mini", "sonnet", "opus"}
    assert scores["gpt5mini"].overall == 5
    assert scores["sonnet"].overall == 70
    assert scores["opus"].overall == 25


def test_score_leaves_axes_as_none():
    """Laya doesn't produce per-axis scores. reasoning/coding/context must be None."""
    models = _models()
    sel = _selector()
    agent = _fake_agent("gpt5mini", {"gpt5mini": 1.0, "sonnet": 0.0, "opus": 0.0})
    with patch.object(laya_backend, "_load_agent", return_value=agent):
        scores = score("hi", models, sel)

    for s in scores.values():
        assert s.reasoning is None
        assert s.coding is None
        assert s.context is None
        assert s.creativity is None
        assert s.explanation  # non-empty


def test_score_clamps_to_zero_hundred():
    """Probabilities outside [0, 1] must clamp to [0, 100]."""
    models = _models()
    sel = _selector()
    # The dictionary says probabilities are in [0,1], but Laya could in theory
    # misbehave - verify we don't blow up.
    agent = _fake_agent("opus", {"gpt5mini": -0.1, "sonnet": 0.5, "opus": 1.5})
    with patch.object(laya_backend, "_load_agent", return_value=agent):
        scores = score("hard math proof", models, sel)

    assert scores["gpt5mini"].overall == 0
    assert scores["opus"].overall == 100


def test_choice_gets_explanation():
    """The chosen model should carry a human-readable explanation; others get a P=note."""
    models = _models()
    sel = _selector()

    agent = _fake_agent("opus", {"gpt5mini": 0.1, "sonnet": 0.2, "opus": 0.7})

    # Replace predict() so it includes an "explanation" field in the answer.
    original_predict = agent.predict

    def predict_with_explanation(state, questions, **kwargs):
        result = original_predict(state, questions, **kwargs)
        result["answers"]["best_model"]["explanation"] = "Hard task needs frontier model."
        return result

    agent.predict = predict_with_explanation

    with patch.object(laya_backend, "_load_agent", return_value=agent):
        scores = score("design raft consensus", models, sel)

    assert "frontier" in scores["opus"].explanation.lower() or "Hard task" in scores["opus"].explanation
    assert "P=" in scores["gpt5mini"].explanation


def test_score_raises_on_missing_probabilities():
    """If Laya forgets a candidate, raise SelectorError naming it."""
    models = _models()
    sel = _selector()
    # Probabilities missing the ``opus`` key
    agent = _fake_agent("sonnet", {"gpt5mini": 0.4, "sonnet": 0.6})
    with patch.object(laya_backend, "_load_agent", return_value=agent):
        with pytest.raises(SelectorError, match="opus"):
            score("hello", models, sel)


def test_score_raises_on_empty_answers():
    models = _models()
    sel = _selector()

    class _EmptyStub:
        def predict(self, state, questions, **kwargs):
            return {"answers": {}}

    with patch.object(laya_backend, "_load_agent", return_value=_EmptyStub()):
        with pytest.raises(SelectorError, match="no answers"):
            score("hello", models, sel)


def test_score_raises_on_wrong_question_type():
    models = _models()
    sel = _selector()

    class _WrongTypeStub:
        def predict(self, state, questions, **kwargs):
            return {"answers": {"best_model": {"type": "score", "score": 0.0}}}

    with patch.object(laya_backend, "_load_agent", return_value=_WrongTypeStub()):
        with pytest.raises(SelectorError, match="not a 'choice'"):
            score("hello", models, sel)


def test_score_wraps_predict_failure():
    models = _models()
    sel = _selector()

    class _BoomStub:
        def predict(self, state, questions, **kwargs):
            raise RuntimeError("kaboom")

    with patch.object(laya_backend, "_load_agent", return_value=_BoomStub()):
        with pytest.raises(SelectorError, match="kaboom"):
            score("hello", models, sel)


def test_question_construction_uses_model_ids_as_keys():
    """The choice question keys must match the candidate model ids."""
    models = _models()
    sel = _selector()
    captured: dict[str, Any] = {}

    def fake_predict(self, state, questions, **kwargs):
        captured["questions"] = questions
        return {
            "answers": {
                "best_model": {
                    "type": "choice",
                    "choice": "gpt5mini",
                    "probabilities": {"gpt5mini": 1.0, "sonnet": 0.0, "opus": 0.0},
                }
            }
        }

    class _Stub:
        predict = fake_predict

    with patch.object(laya_backend, "_load_agent", return_value=_Stub()):
        score("hi", models, sel)

    q = captured["questions"]["best_model"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"gpt5mini", "sonnet", "opus"}
    assert "Small, fast, cheap" in q["criteria"]["gpt5mini"]
    assert "coding, architecture" in q["criteria"]["sonnet"]
    assert "$0.15/1M input" in q["criteria"]["gpt5mini"]
    assert "$15.00/1M input" in q["criteria"]["opus"]