"""Public data types: model scores, selection result, and policy enum."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PolicyType = Literal["cheapest_capable", "highest_confidence", "best_value"]


class ModelScore(BaseModel):
    """Score assigned to a single candidate model by the selector.

    The four capability axes (``reasoning`` / ``coding`` / ``context`` /
    ``creativity``) are populated by the LLM backend. The ``laya`` backend
    does not produce per-axis scores - it returns a single calibrated
    probability per candidate, which becomes ``overall`` - so those
    fields are left as ``None``.
    """

    id: str = Field(..., description="The model profile id being scored")
    overall: int = Field(..., ge=0, le=100, description="Weighted overall confidence 0-100")
    reasoning: int | None = Field(
        None, ge=0, le=100,
        description="Reasoning capability fit 0-100. None when the selector does not produce per-axis scores.",
    )
    coding: int | None = Field(
        None, ge=0, le=100,
        description="Coding capability fit 0-100. None when the selector does not produce per-axis scores.",
    )
    context: int | None = Field(
        None, ge=0, le=100,
        description="Context length fit 0-100. None when the selector does not produce per-axis scores.",
    )
    creativity: int | None = Field(None, ge=0, le=100, description="Optional creativity fit 0-100")
    explanation: str = Field(..., min_length=1, description="Why this model got this score")

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v


class SelectionResult(BaseModel):
    """The full output of a model selection call.

    `estimated_cost_usd` is a first-class field: a per-model USD cost estimate
    for the prompt that was just scored, computed from each model's
    per-1M-token cost and a token-count estimate of the input. Output token
    cost is also included (assumed to mirror input unless the policy
    supplies an override). Callers can render this directly to their users
    so the cost trade-off is transparent.
    """

    selected_model: str = Field(..., description="The id of the chosen model")
    policy: PolicyType = Field(..., description="The policy that made the decision")
    scores: dict[str, ModelScore] = Field(..., description="All candidate scores keyed by id")
    reason: str = Field(..., min_length=1, description="Human-readable explanation of the decision")
    estimated_cost_usd: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-model USD cost estimate for this prompt, keyed by model id. "
            "Computed from the input token estimate and each model's "
            "per-1M-token cost. Useful for showing users the cost trade-off. "
            "Populated by ModelDirector.select(); empty when a Policy is "
            "applied directly via select_model()."
        ),
    )
    input_tokens: int = Field(
        0,
        ge=0,
        description=(
            "Estimated input token count used to compute estimated_cost_usd. "
            "Populated by ModelDirector.select(); 0 when a Policy is applied "
            "directly via select_model()."
        ),
    )

    def cost_saved_vs(
        self,
        baseline_model_id: str,
        models: list[dict],
    ) -> float:
        """Return the cost saved if we used `self.selected_model` instead of `baseline_model_id`.

        Both models are looked up in `models` (a list of dicts with `id` and `cost`).
        Returns a fraction between -1.0 and 1.0 (positive = saved, negative = cost more).
        """
        baseline_cost = None
        chosen_cost = None
        for m in models:
            if m["id"] == baseline_model_id:
                baseline_cost = _cost_to_float(m.get("cost"))
            if m["id"] == self.selected_model:
                chosen_cost = _cost_to_float(m.get("cost"))

        if baseline_cost is None:
            raise ValueError(f"baseline_model_id '{baseline_model_id}' not in models")
        if chosen_cost is None:
            raise ValueError(f"selected_model '{self.selected_model}' not in models")
        if baseline_cost == 0:
            return 0.0

        return (baseline_cost - chosen_cost) / baseline_cost


def _cost_to_float(cost: object) -> float:
    """Coerce a `cost` field (Cost model, dict, or number) to a single float.

    For `{input, output}` dicts, return the average so a relative cost
    comparison remains meaningful when the caller passes a plain dict.
    """
    if cost is None:
        return 0.0
    if isinstance(cost, (int, float)):
        return float(cost)
    if isinstance(cost, dict):
        i = float(cost.get("input", 0))
        o = float(cost.get("output", 0))
        return (i + o) / 2
    # Pydantic Cost model
    if hasattr(cost, "input") and hasattr(cost, "output"):
        return (float(cost.input) + float(cost.output)) / 2  # type: ignore[union-attr]
    return 0.0
