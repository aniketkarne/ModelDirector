"""Command-line interface for ModelDirector.

Usage:
    modeldirector select PROMPT --config CONFIG.yaml
    modeldirector select PROMPT_FILE --config CONFIG.yaml
    echo "PROMPT" | modeldirector select --config CONFIG.yaml
    modeldirector score PROMPT --config CONFIG.yaml
    modeldirector validate --config CONFIG.yaml
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from modeldirector.loader import load_config
from modeldirector.selector import ModelDirector, SelectorError


def _read_prompt(prompt: Optional[str], file: Optional[Path]) -> str:
    if file is not None:
        return file.read_text()
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise click.UsageError(
        "Provide a prompt via --prompt, --file, or stdin."
    )


def _override_backend(cfg_dict: dict, backend: Optional[str]) -> dict:
    """Return a copy of ``cfg_dict`` with ``selector.backend`` overridden if the
    user passed ``--selector-backend`` on the CLI. Mutates the deep copy only."""
    if not backend:
        return cfg_dict
    out = copy.deepcopy(cfg_dict)
    sel = out.setdefault("selector", {})
    if not isinstance(sel, dict):
        raise click.UsageError(
            "selector section in config must be a mapping when overriding --selector-backend"
        )
    sel["backend"] = backend
    return out


def _load_cfg(config_path: Path, backend_override: Optional[str]):
    """Load + validate the config, optionally overriding the selector backend."""
    cfg_dict = yaml.safe_load(config_path.read_text()) or {}
    cfg_dict = _override_backend(cfg_dict, backend_override)
    # Expand env vars the same way the loader does.
    from modeldirector.loader import _expand_env
    cfg_dict = _expand_env(cfg_dict)
    return load_config(cfg_dict)


@click.group()
@click.version_option()
def cli() -> None:
    """ModelDirector - stateless AI model selection engine."""


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", "-p", default=None, help="Prompt text. Use --file or stdin if omitted.")
@click.option("--file", "-f", "file", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--pretty/--compact", default=True, help="Pretty-print the JSON output.")
@click.option(
    "--selector-backend",
    "selector_backend",
    default=None,
    type=click.Choice(["llm", "laya"], case_sensitive=False),
    help="Override the selector backend from the config. 'laya' uses the "
         "non-autoregressive Laya decision engine (cheap, no per-token cost).",
)
def select(
    config_path: Path,
    prompt: Optional[str],
    file: Optional[Path],
    pretty: bool,
    selector_backend: Optional[str],
) -> None:
    """Pick the best model for the given prompt."""
    text = _read_prompt(prompt, file)
    cfg = _load_cfg(config_path, selector_backend)
    director = ModelDirector(cfg)
    try:
        result = director.select(text)
    except SelectorError as e:
        click.echo(f"Selector failed: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps(result.model_dump(), indent=2 if pretty else None))


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prompt", "-p", default=None)
@click.option("--file", "-f", "file", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--pretty/--compact", default=True)
@click.option(
    "--selector-backend",
    "selector_backend",
    default=None,
    type=click.Choice(["llm", "laya"], case_sensitive=False),
    help="Override the selector backend from the config.",
)
def score(
    config_path: Path,
    prompt: Optional[str],
    file: Optional[Path],
    pretty: bool,
    selector_backend: Optional[str],
) -> None:
    """Print raw per-model scores without applying a policy."""
    text = _read_prompt(prompt, file)
    cfg = _load_cfg(config_path, selector_backend)
    director = ModelDirector(cfg)
    try:
        scores = director.score(text)
    except SelectorError as e:
        click.echo(f"Selector failed: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps({k: v.model_dump() for k, v in scores.items()}, indent=2 if pretty else None))


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path) -> None:
    """Validate a config file.  Exits 0 on success, 1 on error."""
    try:
        cfg = load_config(config_path)
    except Exception as e:
        click.echo(f"Invalid: {e}", err=True)
        sys.exit(1)
    click.echo(f"OK - {len(cfg.models)} candidate model(s), policy={cfg.policy.type}, "
               f"selector={cfg.selector.backend}/{cfg.selector.provider}/{cfg.selector.model or '-'}")


@cli.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def show(config_path: Path) -> None:
    """Print the resolved configuration."""
    cfg = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    raw["resolved"] = {
        "selector": cfg.selector.model_dump(),
        "policy": cfg.policy.model_dump(),
        "model_count": len(cfg.models),
    }
    click.echo(yaml.safe_dump(raw, sort_keys=False))


if __name__ == "__main__":
    cli()
