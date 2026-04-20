from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import cli


def test_resources_list_shows_assignments(tmp_path: Path, monkeypatch) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    assignments_path = tmp_path / "plugins_store" / "resource_assignments.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(
        json.dumps({"demo": {"ports": [8000], "volumes": ["data"], "services": {"postgres": "uri"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PLUGINS_DIR", plugins_dir)

    result = CliRunner().invoke(cli.cli, ["resources", "list"])

    assert result.exit_code == 0
    assert "demo" in result.output
    assert "8000" in result.output
    assert "postgres" in result.output


def test_resources_show_returns_error_for_unknown_plugin(
    tmp_path: Path, monkeypatch
) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setattr(cli, "PLUGINS_DIR", plugins_dir)

    result = CliRunner().invoke(cli.cli, ["resources", "show", "missing"])

    assert result.exit_code != 0
    assert "No resource assignment found" in result.output
