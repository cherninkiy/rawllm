#!/usr/bin/env python
"""Command-line interface for RawLLM.

Install (editable):
    pip install -e .

Or run directly:
    python cli.py <command>

Examples
--------
::

    # Start the orchestrator
    rawllm run --provider anthropic

    # Plugin management
    rawllm plugin list
    rawllm plugin show my_plugin
    rawllm plugin rollback my_plugin

    # Dependency approval workflow
    rawllm deps pending
    rawllm deps approve requests

    # Metrics & analytics
    rawllm metrics show --plugin my_plugin --format table
    rawllm metrics evolution my_plugin

    # Configuration
    rawllm config show
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

PLUGINS_DIR = Path(os.environ.get("PLUGINS_DIR", "plugins"))
PENDING_FILE = Path("pending_requirements.txt")
ENV_FILE = Path(".env")

# ---------------------------------------------------------------------------
# CLI root group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """RawLLM – command-line management tool."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command("run")
@click.option("--provider", default=None, envvar="LLM_PROVIDER", help="LLM provider to use.")
def run_cmd(provider: str | None) -> None:
    """Start the orchestrator (equivalent to python run.py)."""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    # Import here to avoid heavy startup cost for other commands.
    from run import main  # type: ignore[import]
    main()


# ---------------------------------------------------------------------------
# plugin group
# ---------------------------------------------------------------------------


@cli.group("plugin")
def plugin_group() -> None:
    """Manage plugins."""


@plugin_group.command("list")
def plugin_list() -> None:
    """List all plugins currently on disk."""
    if not PLUGINS_DIR.exists():
        click.echo(f"Plugins directory not found: {PLUGINS_DIR}", err=True)
        sys.exit(1)
    plugins = sorted(p.stem for p in PLUGINS_DIR.glob("*.py"))
    if not plugins:
        click.echo("No plugins found.")
        return
    for name in plugins:
        click.echo(name)


@plugin_group.command("show")
@click.argument("name")
def plugin_show(name: str) -> None:
    """Display the source code of a plugin."""
    path = PLUGINS_DIR / f"{name}.py"
    if not path.exists():
        click.echo(f"Plugin {name!r} not found at {path}", err=True)
        sys.exit(1)
    click.echo(path.read_text(encoding="utf-8"))


@plugin_group.command("add")
@click.argument("name")
@click.argument("code_file", type=click.Path(exists=True, readable=True))
def plugin_add(name: str, code_file: str) -> None:
    """Add or update a plugin from CODE_FILE.

    The code is passed through the same import-gating and sandboxing
    as plugins added by the LLM.
    """
    code = Path(code_file).read_text(encoding="utf-8")
    from core.plugin_manager import PluginManager

    pm = PluginManager(PLUGINS_DIR)
    pm.load_plugins()
    result = pm.add_plugin(name, code)
    _print_result(result)


@plugin_group.command("rollback")
@click.argument("name")
@click.option("--version", default=None, help="Specific version string (e.g. v2_20240101). Not yet implemented.")
def plugin_rollback(name: str, version: str | None) -> None:
    """Rollback a plugin to its most recently archived version."""
    if version is not None:
        click.echo("--version is not yet implemented; rolling back to the latest archived version.", err=True)
    from core.plugin_manager import PluginManager

    pm = PluginManager(PLUGINS_DIR)
    pm.load_plugins()
    result = pm.rollback_plugin(name)
    _print_result(result)


# ---------------------------------------------------------------------------
# deps group
# ---------------------------------------------------------------------------


@cli.group("deps")
def deps_group() -> None:
    """Manage pending dependency approvals."""


@deps_group.command("pending")
def deps_pending() -> None:
    """Show modules waiting for approval."""
    if not PENDING_FILE.exists():
        click.echo("No pending dependencies.")
        return
    content = PENDING_FILE.read_text(encoding="utf-8").strip()
    if not content:
        click.echo("No pending dependencies.")
        return
    click.echo("Pending modules:")
    for line in sorted(set(content.splitlines())):
        click.echo(f"  {line}")


@deps_group.command("approve")
@click.argument("module")
def deps_approve(module: str) -> None:
    """Approve MODULE and add it to ALLOWED_REQUIREMENTS.

    Updates the .env file (or creates it) with the new value.
    """
    _update_allowed_requirements(module, approved=True)
    # Remove from pending file.
    _remove_from_pending(module)
    click.echo(f"Approved: {module}")


@deps_group.command("reject")
@click.argument("module")
def deps_reject(module: str) -> None:
    """Reject MODULE and remove it from pending."""
    _remove_from_pending(module)
    click.echo(f"Rejected: {module}")


# ---------------------------------------------------------------------------
# metrics group
# ---------------------------------------------------------------------------


@cli.group("metrics")
def metrics_group() -> None:
    """View runtime metrics."""


@metrics_group.command("show")
@click.option("--plugin", default=None, help="Filter by plugin name.")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), help="Output format.")
def metrics_show(plugin: str | None, fmt: str) -> None:
    """Show aggregated execution metrics."""
    import core.metrics as metrics_mod

    stats = metrics_mod.aggregate_by_plugin(plugin_name=plugin)
    if not stats:
        click.echo("No metrics found.")
        return
    if fmt == "json":
        click.echo(json.dumps(stats, indent=2))
        return
    # Table format
    _print_table(
        headers=["Plugin", "Executions", "Success", "Failed", "Avg ms", "Versions", "Rollbacks"],
        rows=[
            [
                name,
                str(s["total_executions"]),
                str(s["successful"]),
                str(s["failed"]),
                f"{s['avg_exec_ms']:.1f}",
                str(s["version_changes"]),
                str(s["rollbacks"]),
            ]
            for name, s in sorted(stats.items())
        ],
    )


@metrics_group.command("evolution")
@click.argument("plugin_name")
def metrics_evolution(plugin_name: str) -> None:
    """Show execution timeline for PLUGIN_NAME (version changes and performance)."""
    import core.metrics as metrics_mod

    events = metrics_mod.get_events(plugin_name=plugin_name)
    if not events:
        click.echo(f"No metrics found for plugin {plugin_name!r}.")
        return
    for event in events:
        ts = event.get("timestamp", "?")[:19].replace("T", " ")
        etype = event.get("event", "?")
        if etype == "plugin_execution":
            success = "✓" if event.get("success") else "✗"
            ms = event.get("execution_time_ms", 0)
            click.echo(f"{ts}  exec {success}  {ms:.1f} ms")
        elif etype == "version_change":
            old = event.get("old_version", "?")
            new = event.get("new_version", "?")
            click.echo(f"{ts}  version {old} → {new}")
        elif etype == "rollback":
            frm = event.get("from_version", "?")
            to = event.get("to_version", "?")
            click.echo(f"{ts}  rollback {frm} → {to}")
        elif etype == "dependency_request":
            pending = event.get("pending", [])
            click.echo(f"{ts}  dep-request pending={pending}")


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group() -> None:
    """View and modify configuration."""


@config_group.command("show")
def config_show() -> None:
    """Print all configuration variables."""
    from core.config import (
        ALLOWED_REQUIREMENTS,
        RAWLLM_CORE_USER,
        SANDBOX_BACKEND,
        SANDBOX_CORE_REPO_VOLUME,
        SANDBOX_DOCKER_IMAGE,
        SANDBOX_PLUGIN_STORE_VOLUME,
        SANDBOX_PLUGIN_USER,
        SANDBOX_TIMEOUT,
        SANDBOX_WORKSPACE_VOLUME,
        TRUSTED_PLUGINS,
    )
    from core.llm.registry import LLM_PROVIDERS

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL", LLM_PROVIDERS.get(provider, {}).get("model", "?"))
    click.echo(f"LLM_PROVIDER         = {provider}")
    click.echo(f"LLM_MODEL            = {model}")
    click.echo(f"TRUSTED_PLUGINS      = {TRUSTED_PLUGINS}")
    click.echo(f"ALLOWED_REQUIREMENTS = {ALLOWED_REQUIREMENTS}")
    click.echo(f"RAWLLM_CORE_USER     = {RAWLLM_CORE_USER}")
    click.echo(f"SANDBOX_PLUGIN_USER  = {SANDBOX_PLUGIN_USER}")
    click.echo(f"SANDBOX_BACKEND      = {SANDBOX_BACKEND}")
    click.echo(f"SANDBOX_DOCKER_IMAGE = {SANDBOX_DOCKER_IMAGE}")
    click.echo(f"WORKSPACE_VOLUME     = {SANDBOX_WORKSPACE_VOLUME}")
    click.echo(f"CORE_REPO_VOLUME     = {SANDBOX_CORE_REPO_VOLUME}")
    click.echo(f"PLUGIN_STORE_VOLUME  = {SANDBOX_PLUGIN_STORE_VOLUME}")
    click.echo(f"SANDBOX_TIMEOUT      = {SANDBOX_TIMEOUT}s")
    click.echo(f"PLUGINS_DIR          = {PLUGINS_DIR}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set KEY=VALUE in the .env file."""
    _set_env_var(key, value)
    click.echo(f"Set {key}={value!r} in {ENV_FILE}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _print_result(result: dict[str, Any]) -> None:
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        sys.exit(1)
    click.echo(json.dumps(result, indent=2))


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        click.echo("  ".join(headers))
        return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    click.echo(header_line)
    click.echo("-" * len(header_line))
    for row in rows:
        click.echo(sep.join(c.ljust(w) for c, w in zip(row, widths)))


def _read_allowed_requirements() -> list[str]:
    """Read ALLOWED_REQUIREMENTS from .env, falling back to env var."""
    _load_dotenv_if_present()
    default = "json,datetime,math,re,collections,itertools,typing"
    raw = os.environ.get("ALLOWED_REQUIREMENTS", default)
    return [r.strip() for r in raw.split(",") if r.strip()]


def _update_allowed_requirements(module: str, approved: bool) -> None:
    current = _read_allowed_requirements()
    if approved and module not in current:
        current.append(module)
    elif not approved and module in current:
        current.remove(module)
    _set_env_var("ALLOWED_REQUIREMENTS", ",".join(current))


def _remove_from_pending(module: str) -> None:
    if not PENDING_FILE.exists():
        return
    lines = PENDING_FILE.read_text(encoding="utf-8").splitlines()
    remaining = [line for line in lines if line.strip() != module]
    PENDING_FILE.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
    except ImportError:
        pass


def _set_env_var(key: str, value: str) -> None:
    """Write or update KEY=VALUE in .env."""
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    key_found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            new_lines.append(f"{key}={value}")
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli()
