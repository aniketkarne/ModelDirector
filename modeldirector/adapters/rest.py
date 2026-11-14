"""FastAPI REST adapter.

Run with:
    MODELDIRECTOR_CONFIG=./examples/config.yaml \
        uvicorn modeldirector.adapters.rest:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modeldirector import __version__
from modeldirector.loader import load_config
from modeldirector.selector import ModelDirector, SelectorError

app = FastAPI(
    title="ModelDirector",
    description="Stateless AI model selection engine.",
    version=__version__,
)

# Load the default config from MODELDIRECTOR_CONFIG env var once at startup.
_default_config_path = os.environ.get("MODELDIRECTOR_CONFIG")
_default_director: Optional[ModelDirector] = None
if _default_config_path and Path(_default_config_path).exists():
    _default_director = ModelDirector(load_config(_default_config_path))


class SelectRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user task to route")
    config_path: Optional[str] = Field(
        None,
        description="Optional path to a YAML config. If omitted, uses the server's default.",
    )


class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    config_path: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "default_config_loaded": _default_director is not None}


@app.post("/select")
def select(req: SelectRequest) -> dict:
    director = _resolve_director(req.config_path)
    try:
        result = director.select(req.prompt)
    except SelectorError as e:
        raise HTTPException(status_code=502, detail=f"Selector failed: {e}") from e
    return result.model_dump()


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    director = _resolve_director(req.config_path)
    try:
        scores = director.score(req.prompt)
    except SelectorError as e:
        raise HTTPException(status_code=502, detail=f"Selector failed: {e}") from e
    return {k: v.model_dump() for k, v in scores.items()}


def _resolve_director(config_path: Optional[str]) -> ModelDirector:
    if config_path:
        return ModelDirector(load_config(config_path))
    if _default_director is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No default config. Pass config_path in the request, or set "
                "MODELDIRECTOR_CONFIG in the server environment."
            ),
        )
    return _default_director
