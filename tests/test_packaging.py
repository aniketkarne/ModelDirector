"""Release-hygiene tests: packaging metadata, console scripts, and the
example configs the README tells users to copy.

These pin the promises the README makes:
  * `pip install "modeldirector[rest]"` / `[all]` resolve to real extras
  * `modeldirector-mcp CONFIG.yaml` exists as a console script
  * version strings cannot drift apart (pyproject vs package vs REST app)
  * the example configs stay valid and free of retired model ids
  * the README never re-gains the pre-move `aniketkarne-com` URLs
"""
from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

import modeldirector
from modeldirector.loader import load_config

_ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())


# --- versions ----------------------------------------------------------------


def test_version_consistent_across_package_and_pyproject():
    data = _pyproject()
    assert data["project"]["version"] == modeldirector.__version__, (
        "pyproject.toml version and modeldirector.__version__ must match"
    )


def test_rest_adapter_version_not_hardcoded():
    # The REST app must report the package version, not a frozen literal.
    source = (_ROOT / "modeldirector" / "adapters" / "rest.py").read_text()
    assert 'version="0.1.0"' not in source and "version='0.1.0'" not in source
    assert "__version__" in source


# --- install promises --------------------------------------------------------


def test_readme_install_extras_exist():
    data = _pyproject()
    extras = data["project"]["optional-dependencies"]
    for name in ("mcp", "laya", "rest", "all"):
        assert name in extras, f"README advertises modeldirector[{name}] but it is not defined"


def test_all_extra_aggregates_the_others():
    extras = _pyproject()["project"]["optional-dependencies"]
    all_extra = " ".join(extras["all"])
    for name in ("mcp", "laya", "rest"):
        assert name in all_extra, f"modeldirector[all] must include the {name} extra"


def test_console_scripts_match_readme_commands():
    scripts = _pyproject()["project"]["scripts"]
    assert "modeldirector" in scripts
    assert "modeldirector-mcp" in scripts, (
        "README quickstart runs `modeldirector-mcp config.yaml`; the console "
        "script must exist"
    )
    assert scripts["modeldirector-mcp"] == "modeldirector.adapters.mcp:main"


def test_mcp_main_entry_point_resolves():
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("mcp package not installed (optional extra)")
    from modeldirector.adapters import mcp as mcp_adapter

    assert callable(mcp_adapter.main)


# --- README / example drift guards ------------------------------------------


def test_readme_has_no_pre_move_urls():
    # The repo moved aniketkarne-com -> aniketkarne; the v0.2 restructure
    # silently re-introduced the dead owner once. This pins the fix.
    readme = (_ROOT / "README.md").read_text()
    assert "aniketkarne-com" not in readme


# Retired / deprecated model ids that must not appear in the copy-paste
# config surface. Update this list as models retire; treat it as code.
# Verified against the live OpenRouter catalog 2026-11-14:
#   claude-3.5-* and claude-opus-4 are gone from the catalog entirely;
#   gpt-4o-mini is deprecated at OpenAI (removal 2027-01-14).
_STALE_MODEL_IDS = re.compile(
    r"claude-3\.5|claude-3-5|gpt-3\.5|gpt-4o(-mini)?|claude-opus-4(?![\w.])"
)


def test_example_configs_have_no_retired_model_ids():
    for cfg_path in sorted((_ROOT / "examples").glob("*.yaml")):
        matches = _STALE_MODEL_IDS.findall(cfg_path.read_text())
        assert not matches, f"{cfg_path.name} references retired model ids: {matches}"


def test_example_configs_validate():
    for cfg_path in sorted((_ROOT / "examples").glob("*.yaml")):
        cfg = load_config(cfg_path)
        assert cfg.models, f"{cfg_path.name} loaded with no candidate models"


# --- selector transport contract ---------------------------------------------


def test_temperature_not_sent_when_unset(monkeypatch):
    """Current reasoning models reject non-default sampling params (400).

    The selector must not send `temperature` unless the user configured it.
    """
    import litellm

    from modeldirector.config import SelectorConfig
    from modeldirector.selectors import llm as llm_backend

    captured: list[dict] = []

    class _Msg:
        content = "{}"

    class _Choice:
        message = _Msg()

    class _Resp:
        def __init__(self) -> None:
            self.choices = [_Choice()]

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return _Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    sel = SelectorConfig(provider="openrouter", model="openai/gpt-6-luna")
    assert sel.temperature is None
    llm_backend._call_selector("prompt", sel)
    assert "temperature" not in captured[0], (
        "temperature must be omitted when unset - current models 400 on it"
    )

    sel_set = SelectorConfig(
        provider="openrouter", model="openai/gpt-6-luna", temperature=0.5
    )
    llm_backend._call_selector("prompt", sel_set)
    assert captured[1]["temperature"] == 0.5
