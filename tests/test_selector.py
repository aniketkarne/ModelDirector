"""Tests for the selector engine.  Mocks the LLM call so no API is hit."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from modeldirector.config import Config
from modeldirector.selector import ModelDirector, SelectorError, _extract_json
from modeldirector.selectors import llm as llm_backend
from tests.conftest import make_score


def _make_director(config: Config) -> ModelDirector:
    return ModelDirector(config)


def test_score_parses_clean_json_response(three_model_config):
    director = _make_director(three_model_config)
    fake_response = json.dumps(
        {
            "models": [
                make_score("gpt5mini", 90, explanation="Easy task, cheap model is fine"),
                make_score("sonnet", 85, explanation="More than enough"),
                make_score("opus", 60, explanation="Overkill"),
            ]
        }
    )

    with patch.object(llm_backend, "_call_selector", return_value=fake_response):
        scores = director.score("Summarise this article.")

    assert set(scores) == {"gpt5mini", "sonnet", "opus"}
    assert scores["gpt5mini"].overall == 90
    assert scores["opus"].overall == 60


def test_score_strips_markdown_fences(three_model_config):
    director = _make_director(three_model_config)
    fenced = "```json\n" + json.dumps(
        {"models": [make_score("gpt5mini", 80), make_score("sonnet", 80), make_score("opus", 80)]}
    ) + "\n```"

    with patch.object(llm_backend, "_call_selector", return_value=fenced):
        scores = director.score("test prompt")
    assert set(scores) == {"gpt5mini", "sonnet", "opus"}


def test_score_rejects_missing_models_key(three_model_config):
    director = _make_director(three_model_config)
    with patch.object(llm_backend, "_call_selector", return_value='{"foo": "bar"}'):
        with pytest.raises(SelectorError, match="missing 'models' key"):
            director.score("test")


def test_score_rejects_missing_candidate(three_model_config):
    director = _make_director(three_model_config)
    response = json.dumps({"models": [make_score("gpt5mini", 90), make_score("sonnet", 85)]})  # opus missing

    with patch.object(llm_backend, "_call_selector", return_value=response):
        with pytest.raises(SelectorError, match="did not return scores for"):
            director.score("test")


def test_score_rejects_invalid_entry(three_model_config):
    director = _make_director(three_model_config)
    response = json.dumps(
        {
            "models": [
                {"id": "gpt5mini", "overall": 200, "reasoning": 80, "coding": 80, "context": 80, "explanation": "x"},
                make_score("sonnet", 80),
                make_score("opus", 80),
            ]
        }
    )

    with patch.object(llm_backend, "_call_selector", return_value=response):
        with pytest.raises(SelectorError, match="invalid score entry"):
            director.score("test")


def test_select_applies_default_cheapest_capable_policy(three_model_config):
    director = _make_director(three_model_config)
    response = json.dumps(
        {
            "models": [
                make_score("gpt5mini", 90, explanation="Easy"),
                make_score("sonnet", 95, explanation="Ok"),
                make_score("opus", 99, explanation="Overkill"),
            ]
        }
    )
    with patch.object(llm_backend, "_call_selector", return_value=response):
        result = director.select("Summarise this short article.")

    assert result.selected_model == "gpt5mini"
    assert result.policy == "cheapest_capable"


def test_select_includes_estimated_cost_usd_for_every_model(three_model_config):
    """`estimated_cost_usd` should always have an entry for every candidate."""
    director = _make_director(three_model_config)
    response = json.dumps(
        {
            "models": [
                make_score("gpt5mini", 90, explanation="Easy"),
                make_score("sonnet", 95, explanation="Ok"),
                make_score("opus", 99, explanation="Overkill"),
            ]
        }
    )
    with patch.object(llm_backend, "_call_selector", return_value=response):
        result = director.select("Short prompt.")

    assert set(result.estimated_cost_usd) == {"gpt5mini", "sonnet", "opus"}
    # gpt5mini is the cheapest by ~20x; opus is the most expensive.
    assert result.estimated_cost_usd["gpt5mini"] < result.estimated_cost_usd["sonnet"]
    assert result.estimated_cost_usd["sonnet"] < result.estimated_cost_usd["opus"]
    # And a real, positive number for each.
    assert all(v > 0 for v in result.estimated_cost_usd.values())


def test_select_input_tokens_field_populated(three_model_config):
    director = _make_director(three_model_config)
    response = json.dumps(
        {
            "models": [
                make_score("gpt5mini", 80),
                make_score("sonnet", 80),
                make_score("opus", 80),
            ]
        }
    )
    with patch.object(llm_backend, "_call_selector", return_value=response):
        result = director.select("This is a test prompt with a few words in it.")
    assert result.input_tokens > 0


def test_select_assumed_output_tokens_override(three_model_config):
    """Caller can override the output-token estimate used in cost calculation."""
    director = _make_director(three_model_config)
    response = json.dumps(
        {
            "models": [
                make_score("gpt5mini", 80),
                make_score("sonnet", 80),
                make_score("opus", 80),
            ]
        }
    )
    with patch.object(llm_backend, "_call_selector", return_value=response):
        result_default = director.select("Short.")
        result_zero = director.select("Short.", assumed_output_tokens=0)
    # output is a non-trivial component of cost for opus; zeroing it should
    # make opus cheaper than the default-assumption case.
    assert result_zero.estimated_cost_usd["opus"] < result_default.estimated_cost_usd["opus"]


def test_extracted_json_handles_prose_around_block():
    raw = 'Here you go:\n{"models": []}\nDone!'
    parsed = _extract_json(raw)
    assert parsed == {"models": []}


def test_extracted_json_raises_on_garbage():
    with pytest.raises(SelectorError):
        _extract_json("not json at all")


def test_extracted_json_raises_when_no_braces():
    with pytest.raises(SelectorError, match="Could not find JSON"):
        _extract_json("just plain text")


def test_prompt_includes_description(three_model_config):
    """The selector prompt must surface each model's description to the LLM."""
    director = _make_director(three_model_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str, selector) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("gpt5mini", 80), make_score("sonnet", 80), make_score("opus", 80)]}
        )

    with patch.object(llm_backend, "_call_selector", side_effect=fake_call):
        director.score("test")

    assert "OpenAI's small, fast, low-cost model" in captured["prompt"]
    assert "Anthropic's flagship" in captured["prompt"]
    assert "id" in captured["prompt"]
    assert "capabilities" in captured["prompt"]


def test_prompt_includes_strengths(three_model_config):
    """The selector prompt must surface each model's `strengths` tags."""
    director = _make_director(three_model_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str, selector) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("gpt5mini", 80), make_score("sonnet", 80), make_score("opus", 80)]}
        )

    with patch.object(llm_backend, "_call_selector", side_effect=fake_call):
        director.score("test")

    # sonnet's strengths must be in the prompt
    assert "coding" in captured["prompt"]
    assert "architecture" in captured["prompt"]
    assert "refactoring" in captured["prompt"]
    # strengths is a structured field
    assert '"strengths"' in captured["prompt"]


def test_prompt_includes_cost_per_1m(three_model_config):
    """The selector prompt should include per-1M cost for context."""
    director = _make_director(three_model_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str, selector) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("gpt5mini", 80), make_score("sonnet", 80), make_score("opus", 80)]}
        )

    with patch.object(llm_backend, "_call_selector", side_effect=fake_call):
        director.score("test")

    assert "cost_per_1m_tokens_usd" in captured["prompt"]


def test_prompt_truncates_huge_user_prompt(three_model_config):
    director = _make_director(three_model_config)
    captured: dict[str, Any] = {}

    def fake_call(prompt: str, selector) -> str:
        captured["prompt"] = prompt
        return json.dumps(
            {"models": [make_score("gpt5mini", 80), make_score("sonnet", 80), make_score("opus", 80)]}
        )

    huge = "x" * 20_000
    with patch.object(llm_backend, "_call_selector", side_effect=fake_call):
        director.score(huge)

    assert "[... truncated for length ...]" in captured["prompt"]
    # The full 20k should not make it into the prompt
    assert "x" * 20_000 not in captured["prompt"]
