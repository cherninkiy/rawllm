"""Targeted branch tests to raise coverage on critical PluginManager paths."""

from __future__ import annotations

import queue
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from core.plugin_manager import (
    PluginManager,
    _read_version_meta,
    _touch_future,
    _update_current_symlink,
)


def test_touch_future_handles_cache_path_errors(tmp_path: Path) -> None:
    path = tmp_path / "p.py"
    path.write_text("x=1", encoding="utf-8")

    with patch("importlib.util.cache_from_source", side_effect=ValueError("no cache")):
        _touch_future(path)

    assert path.exists()


def test_read_version_meta_invalid_json_returns_defaults(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    arch = plugins_dir.parent / "plugins_store" / "archive" / "demo"
    arch.mkdir(parents=True)
    (arch / "version.json").write_text("{broken", encoding="utf-8")

    meta = _read_version_meta(plugins_dir, "demo")
    assert meta == {"current_version": 0, "rollback_count": 0}


def test_update_current_symlink_falls_back_to_copy(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    target = plugins_dir / "a.py"
    target.write_text("def run(x): return x", encoding="utf-8")

    with patch("pathlib.Path.symlink_to", side_effect=OSError("no symlink")), patch("shutil.copy2") as copy2:
        _update_current_symlink(plugins_dir, "a", target)

    copy2.assert_called_once()


def test_add_plugin_returns_error_on_write_failure(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        result = manager.add_plugin("x", "def run(x): return {}")

    assert "error" in result


def test_reload_plugin_import_error_returns_error(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    path = plugins_dir / "bad.py"
    path.write_text("x = 1", encoding="utf-8")
    result = manager.reload_plugin("bad")

    assert "error" in result


def test_unload_plugin_shutdown_exception_is_swallowed(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    code = """
def shutdown():
    raise RuntimeError('boom')

def run(input_data):
    return {'ok': True}
"""
    (plugins_dir / "s.py").write_text(code, encoding="utf-8")
    manager.load_plugins()

    result = manager.unload_plugin("s")
    assert result == {"status": "ok", "plugin": "s"}


def test_reload_plugin_shutdown_exception_is_swallowed(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    path = plugins_dir / "r.py"
    path.write_text("def run(input_data):\n    return {'ok': True}\n", encoding="utf-8")
    manager.load_plugins()

    # Force old instance shutdown() to raise; reload must still continue.
    def _boom() -> None:
        raise RuntimeError("shutdown failed")

    manager.plugins["r"].shutdown = _boom  # type: ignore[attr-defined]
    with patch("core.plugin_manager.logger.exception") as log_exc:
        result = manager.reload_plugin("r")

    assert result == {"status": "ok", "plugin": "r"}
    assert log_exc.called


def test_import_plugin_spec_none_raises(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    path = plugins_dir / "x.py"
    path.write_text("def run(x): return {}", encoding="utf-8")

    with patch("core.plugin_manager.importlib.util.spec_from_file_location", return_value=None):
        try:
            manager._import_plugin("x", path)
            assert False, "expected ImportError"
        except ImportError:
            pass


def test_import_plugin_init_typeerror_is_ignored(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    code = """
def init(required):
    return required

def run(input_data):
    return {'ok': True}
"""
    path = plugins_dir / "t.py"
    path.write_text(code, encoding="utf-8")

    manager._import_plugin("t", path)
    assert "t" in manager.plugins


def test_import_plugin_init_exception_is_ignored(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    code = """
def init():
    raise RuntimeError('init failed')

def run(input_data):
    return {'ok': True}
"""
    path = plugins_dir / "e.py"
    path.write_text(code, encoding="utf-8")

    manager._import_plugin("e", path)
    assert "e" in manager.plugins


def test_call_inprocess_without_run_returns_attribute_error(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    plugin = ModuleType("m")
    result, _ms, ok, err_type, tb = manager._call_inprocess("m", plugin, {}, 1)

    assert ok is False
    assert err_type == "AttributeError"
    assert tb is None
    assert "no run()" in result["error"]


def test_call_inprocess_timeout_branch(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    plugin = ModuleType("m")
    plugin.run = lambda _x: {"ok": True}  # type: ignore[attr-defined]

    with patch("queue.Queue.get", side_effect=queue.Empty):
        result, _ms, ok, err_type, tb = manager._call_inprocess("m", plugin, {}, 1)

    assert ok is False
    assert err_type == "TimeoutError"
    assert tb is None
    assert "timed out" in result["error"]


def test_call_inprocess_exception_path_returns_error_with_traceback(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    plugin = ModuleType("m")

    def _raise(_input: dict[str, object]) -> dict[str, object]:
        raise ValueError("boom")

    plugin.run = _raise  # type: ignore[attr-defined]

    result, _ms, ok, err_type, tb = manager._call_inprocess("m", plugin, {}, 1)
    assert ok is False
    assert err_type == "ValueError"
    assert tb is not None
    assert "ValueError: boom" in tb
    assert result["error"] == "boom"
    assert "traceback" in result


def test_call_inprocess_systemexit_worker_path(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    plugin = ModuleType("m")

    def _exit(_input: dict[str, object]) -> dict[str, object]:
        raise SystemExit(0)

    plugin.run = _exit  # type: ignore[attr-defined]

    class _FakeThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self) -> None:
            try:
                self._target()
            except SystemExit:
                # Mirror worker termination without bubbling to pytest warnings.
                pass

    with patch("core.plugin_manager.threading.Thread", _FakeThread):
        # Worker exits without writing to queue; main path times out.
        result, _ms, ok, err_type, tb = manager._call_inprocess("m", plugin, {}, 0.05)
    assert ok is False
    assert err_type == "TimeoutError"
    assert tb is None


def test_call_subprocess_invalid_json_branch(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    proc = SimpleNamespace(stdout="not-json", stderr="trace")
    with patch("core.plugin_manager.subprocess.run", return_value=proc):
        result, _ms, ok, err_type, tb = manager._call_subprocess("x", {}, 2, {})

    assert ok is False
    assert err_type == "JSONDecodeError"
    assert tb == "trace"


def test_call_subprocess_error_payload_branch(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    proc = SimpleNamespace(stdout='{"error": "sandbox failed"}', stderr="")
    with patch("core.plugin_manager.subprocess.run", return_value=proc):
        result, _ms, ok, err_type, tb = manager._call_subprocess("x", {}, 2, {})

    assert ok is False
    assert err_type == "RuntimeError"
    assert tb == "sandbox failed"
    assert result["traceback"] == "sandbox failed"


def test_call_subprocess_timeout_branch(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    with patch("core.plugin_manager.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=1)):
        result, _ms, ok, err_type, tb = manager._call_subprocess("x", {}, 2, {})

    assert ok is False
    assert err_type == "TimeoutError"
    assert tb is None


def test_call_docker_subprocess_required_returns_failure_tuple(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    with patch("core.plugin_manager.DockerSandboxRunner") as runner_cls, patch(
        "core.plugin_manager.SANDBOX_DOCKER_REQUIRED", True
    ):
        runner_cls.return_value.run_plugin.return_value = ({"error": "d"}, 1.0, False, "DockerError", "d")
        result = manager._call_docker_subprocess("x", {}, 30, {})

    assert result is not None
    assert result[2] is False


def test_read_plugin_code_oserror_returns_empty(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    with patch("pathlib.Path.read_text", side_effect=OSError("nope")):
        assert manager._read_plugin_code("x") == ""
