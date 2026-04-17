"""Tests for ToolExecutor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tool_executor import ToolExecutor, _append_pending_requirements


@pytest.fixture()
def mock_pm() -> MagicMock:
    pm = MagicMock()
    return pm


@pytest.fixture()
def executor(mock_pm: MagicMock) -> ToolExecutor:
    return ToolExecutor(mock_pm)


def test_add_plugin_delegates_to_pm(executor: ToolExecutor, mock_pm: MagicMock) -> None:
    mock_pm.add_plugin.return_value = {"status": "ok", "plugin": "myplugin"}
    result = executor.add_plugin("myplugin", "def run(d): return d")
    mock_pm.add_plugin.assert_called_once_with("myplugin", "def run(d): return d")
    assert result == {"status": "ok", "plugin": "myplugin"}


def test_run_plugin_delegates_to_pm(executor: ToolExecutor, mock_pm: MagicMock) -> None:
    mock_pm.call_plugin.return_value = {"result": 42}
    result = executor.run_plugin("calc", {"x": 1})
    mock_pm.call_plugin.assert_called_once_with("calc", {"x": 1})
    assert result == {"result": 42}


def test_unload_plugin_delegates_to_pm(executor: ToolExecutor, mock_pm: MagicMock) -> None:
    mock_pm.unload_plugin.return_value = {"status": "ok", "plugin": "old"}
    result = executor.unload_plugin("old")
    mock_pm.unload_plugin.assert_called_once_with("old")
    assert result == {"status": "ok", "plugin": "old"}


def test_add_plugin_propagates_error(executor: ToolExecutor, mock_pm: MagicMock) -> None:
    mock_pm.add_plugin.return_value = {"error": "syntax error"}
    result = executor.add_plugin("bad", "not python")
    assert "error" in result


def test_run_plugin_propagates_error(executor: ToolExecutor, mock_pm: MagicMock) -> None:
    mock_pm.call_plugin.return_value = {"error": "plugin not loaded"}
    result = executor.run_plugin("missing", {})
    assert "error" in result


def test_add_plugin_blocks_unapproved_imports(tmp_path: Path) -> None:
    """Plugins importing modules outside ALLOWED_REQUIREMENTS must be held back."""
    pm = MagicMock()
    executor = ToolExecutor(pm)

    code = "import requests\n\ndef run(d): return {}\n"

    pending_file = tmp_path / "pending_requirements.txt"
    with patch("core.tool_executor.ALLOWED_REQUIREMENTS", ["json", "datetime"]):
        with patch("core.tool_executor.PENDING_REQUIREMENTS_FILE", pending_file):
            result = executor.add_plugin("net_plugin", code)

    assert result["status"] == "pending_approval"
    assert result["plugin"] == "net_plugin"
    assert "requests" in result["message"]
    pm.add_plugin.assert_not_called()
    assert pending_file.exists()
    content = pending_file.read_text()
    assert "requests" in content


def test_add_plugin_allows_approved_imports(mock_pm: MagicMock) -> None:
    """Plugins importing only approved modules should be passed to PluginManager."""
    mock_pm.add_plugin.return_value = {"status": "ok", "plugin": "math_plugin"}
    executor = ToolExecutor(mock_pm)

    code = "import math\n\ndef run(d): return {'pi': math.pi}\n"

    with patch("core.tool_executor.ALLOWED_REQUIREMENTS", ["math"]):
        result = executor.add_plugin("math_plugin", code)

    mock_pm.add_plugin.assert_called_once()
    assert result.get("status") == "ok"


def test_append_pending_requirements_logs_on_oserror(tmp_path: Path) -> None:
    """I/O errors while writing pending requirements must be logged, not raised."""
    pending_file = tmp_path / "pending_requirements.txt"

    with patch("core.tool_executor.PENDING_REQUIREMENTS_FILE", pending_file), patch(
        "pathlib.Path.open", side_effect=OSError("disk error")
    ), patch("core.tool_executor.logger.exception") as log_exc:
        _append_pending_requirements("plugin_x", ["requests"])

    log_exc.assert_called_once_with("Failed to write pending_requirements.txt")
