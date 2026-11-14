"""Optional MCP adapter using the official `mcp` SDK / FastMCP.

Install with:  pip install "modeldirector[mcp]"

The tool exposes the same logic as the SDK / REST endpoints to MCP clients
(Claude Desktop, Continue, Roo Code, etc.).
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MCP support requires the `mcp` package. Install with: "
        "pip install \"modeldirector[mcp]\""
    ) from e

from modeldirector.loader import load_config
from modeldirector.selector import ModelDirector, SelectorError

mcp = FastMCP("ModelDirector")
_director: ModelDirector | None = None


def configure(config_path: str) -> None:
    """Initialise the global director with the given config file path."""
    global _director
    _director = ModelDirector(load_config(config_path))


@mcp.tool()
def select_model(prompt: str) -> dict:
    """Pick the best model for the given prompt.

    Returns a dict with `selected_model`, `policy`, `scores`, and `reason`.
    """
    if _director is None:
        return {"error": "ModelDirector not configured. Call configure(config_path) first."}
    try:
        result = _director.select(prompt)
    except SelectorError as e:
        return {"error": f"Selector failed: {e}"}
    return result.model_dump()


@mcp.tool()
def score_models(prompt: str) -> dict:
    """Return raw per-model scores without applying a decision policy."""
    if _director is None:
        return {"error": "ModelDirector not configured. Call configure(config_path) first."}
    try:
        scores = _director.score(prompt)
    except SelectorError as e:
        return {"error": f"Selector failed: {e}"}
    return {k: v.model_dump() for k, v in scores.items()}


def main() -> None:
    """Console-script entry point: ``modeldirector-mcp CONFIG.yaml``."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: modeldirector-mcp CONFIG.yaml", file=sys.stderr)
        raise SystemExit(2)
    configure(sys.argv[1])
    mcp.run()

if __name__ == "__main__":  # pragma: no cover
    main()
