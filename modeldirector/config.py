"""Pydantic configuration models for ModelDirector.

A `Config` is a single YAML file (or programmatic dict) that fully describes:
  * which LLM to use as the selector
  * which decision policy to apply
  * which candidate models to score against

ModelDirector is model-agnostic. It does not know the meaning of model names
or "tiers" like cheap/mid/premium. The user defines the candidate set and the
per-model capabilities, strengths, and cost.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator


# --- Cost ---------------------------------------------------------------------


class Cost(BaseModel):
    """Per-1M-token cost in USD. First-class in the engine, not an afterthought.

    `input`  - USD per 1M input tokens.
    `output` - USD per 1M output tokens.

    Used by:
      * `best_value` policy     - score / input cost ratio (input is deterministic)
      * `SelectionResult.estimated_cost_usd` - per-model estimate shown to callers
    """

    input: float = Field(..., ge=0, description="USD per 1M input tokens")
    output: float = Field(..., ge=0, description="USD per 1M output tokens")


# Backward-compat: accept a bare number (legacy single-cost form) and coerce
# to `Cost(input=x, output=x)`. Documented as deprecated; new configs should
# use the `cost: {input, output}` form.
_LegacyCost = Annotated[float, Field(ge=0)]


def _coerce_cost(v: Any) -> Cost:
    """Accept either a Cost-shaped dict or a single float (legacy)."""
    if isinstance(v, Cost):
        return v
    if isinstance(v, (int, float)):
        return Cost(input=float(v), output=float(v))
    if isinstance(v, dict):
        return Cost.model_validate(v)
    raise ValueError(
        f"cost must be a number (USD/1M tokens) or {{input, output}} dict, got {type(v).__name__}"
    )


# --- Capabilities / strengths --------------------------------------------------


class Capabilities(BaseModel):
    """Static capability scores for a model (user-defined, 0-100)."""

    reasoning: int = Field(..., ge=0, le=100)
    coding: int = Field(..., ge=0, le=100)
    context: int = Field(..., ge=0, le=100)
    creativity: int | None = Field(None, ge=0, le=100)


# --- Model profile ------------------------------------------------------------


class ModelProfile(BaseModel):
    """A single candidate model the selector can choose from.

    ModelDirector knows nothing about model names or tiers. Every field here
    is user-defined; the engine treats all candidates uniformly and lets the
    configured policy pick.
    """

    id: str = Field(..., min_length=1, description="Stable identifier used in output")
    name: str = Field(..., min_length=1, description="Provider-specific model name passed to LiteLLM")
    display_name: str | None = Field(None, description="Human-friendly label (optional)")
    description: str = Field(
        "",
        description=(
            "Free-form description of the model. Strongly recommended - smaller "
            "selector models may not recognise bare model names."
        ),
    )
    capabilities: Capabilities
    strengths: list[str] = Field(
        default_factory=list,
        description=(
            "Task-type tags the model is good at (e.g. ['coding', 'architecture', "
            "'long_context']). Surfaced to the selector as structured signals. "
            "Recommended over relying on the selector to know the model name."
        ),
    )
    priority: int = Field(1, ge=1, description="Lower = preferred. Tie-breaker for `cheapest_capable`.")
    cost: Cost = Field(
        ...,
        description="Per-1M-token cost in USD. Can be a {input, output} dict or a single number.",
    )

    @field_validator("id", "name")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v

    @field_validator("cost", mode="before")
    @classmethod
    def _coerce_cost_field(cls, v: Any) -> Any:
        return _coerce_cost(v)


# --- Selector -----------------------------------------------------------------


# Supported selector backends. ``llm`` is the default and asks a configurable
# LiteLLM model to score the candidates. ``laya`` uses the Laya decision
# engine (https://github.com/NandhaKishorM/laya) - a non-autoregressive
# System-1 model that scores in a single forward pass on CPU/GPU.
SelectorBackend = Literal["llm", "laya"]


class SelectorConfig(BaseModel):
    """Configuration of the selector that scores the candidates.

    Two backends are supported:

    * ``llm``  (default) - ask a configurable LLM via LiteLLM. The model
      is named by ``provider`` + ``model`` and authenticated by ``api_key``.
    * ``laya`` - use the Laya decision engine. ``laya_model`` selects the
      checkpoint (e.g. ``"convaiinnovations/laya"`` or ``"multilingual"``);
      ``provider`` / ``model`` / ``api_key`` are unused in this mode.
    """

    backend: SelectorBackend = Field(
        "llm",
        description=(
            "Selector backend. 'llm' (default) uses a LiteLLM-routed LLM. "
            "'laya' uses the Laya decision engine (single forward pass, "
            "CPU/GPU, no per-token cost)."
        ),
    )

    # --- LLM backend fields (used when backend == 'llm') ---
    provider: str = Field(
        "openrouter",
        min_length=1,
        description="LiteLLM provider, e.g. 'openrouter', 'openai'",
    )
    model: str | None = Field(
        None,
        min_length=1,
        description=(
            "Model name in LiteLLM format. Required when backend == 'llm'; "
            "ignored otherwise."
        ),
    )
    api_key: str | None = Field(
        None,
        description=(
            "API key. If unset, ModelDirector falls back to the env var "
            "<PROVIDER>_API_KEY (uppercased)."
        ),
    )
    api_base: str | None = Field(None, description="Optional base URL override")
    temperature: float = Field(0.0, ge=0, le=2, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Optional cap on selector output")

    # --- Laya backend fields (used when backend == 'laya') ---
    laya_model: str = Field(
        "convaiinnovations/laya",
        description=(
            "Laya checkpoint identifier. The default is the English root "
            "checkpoint. Use 'multilingual' (or "
            "'convaiinnovations/laya-multilingual') for 100+ languages "
            "and longer documents."
        ),
    )
    laya_device: str | None = Field(
        None,
        description=(
            "Torch device override for Laya (e.g. 'cpu', 'cuda', 'mps'). "
            "Defaults to Laya's auto-detection."
        ),
    )
    laya_max_len: int | None = Field(
        None,
        gt=0,
        description=(
            "Override the per-call token budget for the Laya forward pass. "
            "Mostly useful when scoring very long user prompts."
        ),
    )

    def resolved_api_key(self) -> str:
        """Return the API key, falling back to the provider's env var."""
        if self.backend == "laya":
            # LLM-backend fields are unused when running Laya.
            return ""
        if self.api_key and not self.api_key.startswith("${"):
            return self.api_key
        if self.api_key and self.api_key.startswith("${"):
            return _resolve_env(self.api_key)
        env_name = f"{self.provider.upper()}_API_KEY"
        val = os.environ.get(env_name)
        if not val:
            raise ValueError(
                f"No API key for provider '{self.provider}'. "
                f"Set 'selector.api_key' or env var ${env_name}."
            )
        return val

    @field_validator("model", mode="after")
    @classmethod
    def _require_model_for_llm(cls, v: str | None, info) -> str | None:
        """``model`` is required only when the LLM backend is selected."""
        backend = info.data.get("backend", "llm")
        if backend == "llm" and not v:
            raise ValueError("selector.model is required when selector.backend == 'llm'")
        return v


# --- Policy -------------------------------------------------------------------


class PolicyConfig(BaseModel):
    """Decision policy applied after scoring."""

    type: Literal["cheapest_capable", "highest_confidence", "best_value"] = "cheapest_capable"
    threshold: int = Field(85, ge=0, le=100, description="Confidence threshold for 'cheapest_capable'")
    cost_per_million: bool = Field(
        True,
        description=(
            "If true, costs are interpreted as USD per 1M tokens. Affects the "
            "`best_value` policy and the `estimated_cost_usd` output only - "
            "not the decision in `cheapest_capable` or `highest_confidence`."
        ),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_legacy(cls, v: object) -> object:
        # accept `cheapest_capable`, `cheapest-capable`, `Cheapest Capable` etc.
        # also accept `balanced` as a legacy alias for `best_value`.
        if isinstance(v, str):
            v_norm = v.strip().lower().replace("-", "_").replace(" ", "_")
            if v_norm == "balanced":
                return "best_value"
            return v_norm
        return v

    @classmethod
    def default(cls) -> "PolicyConfig":
        """Return a default policy config (helper for Pydantic field defaults)."""
        return cls(type="cheapest_capable")  # type: ignore[call-arg]


# --- Top-level config ---------------------------------------------------------


class Config(BaseModel):
    """Top-level configuration."""

    selector: SelectorConfig
    policy: PolicyConfig = Field(default_factory=PolicyConfig.default)
    models: list[ModelProfile] = Field(..., min_length=1)

    @field_validator("models")
    @classmethod
    def _unique_ids(cls, v: list[ModelProfile]) -> list[ModelProfile]:
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate model ids: {dupes}")
        return v


def _resolve_env(placeholder: str) -> str:
    """Resolve ${VAR_NAME} style placeholders against the process environment."""
    m = re.match(r"^\$\{([A-Z0-9_]+)\}$", placeholder)
    if not m:
        raise ValueError(f"Invalid env placeholder: {placeholder!r}")
    val = os.environ.get(m.group(1))
    if not val:
        raise ValueError(f"Environment variable ${m.group(1)} is not set")
    return val
