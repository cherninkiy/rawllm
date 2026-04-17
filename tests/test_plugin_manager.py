"""Tests for PluginManager."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from core.plugin_manager import PluginManager


DUMMY_PLUGIN_CODE = textwrap.dedent(
    """\
    def run(input_data):
        return {"echo": input_data.get("value")}
    """
)

DUMMY_PLUGIN_WITH_INIT = textwrap.dedent(
    """\
    _initialised = False

    def init():
        global _initialised
        _initialised = True

    def run(input_data):
        return {"initialised": _initialised}
    """
)

DUMMY_PLUGIN_WITH_SHUTDOWN = textwrap.dedent(
    """\
    _alive = True

    def shutdown():
        global _alive
        _alive = False

    def run(input_data):
        return {"alive": _alive}
    """
)


@pytest.fixture()
def plugins_dir(tmp_path: Path) -> Path:
    return tmp_path / "plugins"


@pytest.fixture()
def manager(plugins_dir: Path) -> PluginManager:
    plugins_dir.mkdir()
    return PluginManager(plugins_dir)


# ---------------------------------------------------------------------------


def test_load_plugins_discovers_py_files(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "dummy.py").write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()
    assert "dummy" in manager.plugins


def test_load_plugins_skips_files_without_run(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "bad.py").write_text("x = 1")
    manager.load_plugins()
    assert "bad" not in manager.plugins


def test_call_plugin_returns_result(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "echo.py").write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()
    result = manager.call_plugin("echo", {"value": 42})
    assert result == {"echo": 42}


def test_call_plugin_unknown_returns_error(manager: PluginManager) -> None:
    result = manager.call_plugin("nonexistent", {})
    assert "error" in result


def test_call_plugin_timeout(manager: PluginManager, plugins_dir: Path) -> None:
    code = textwrap.dedent(
        """\
        import time

        def run(input_data):
            time.sleep(60)
            return {}
        """
    )
    (plugins_dir / "slow.py").write_text(code)
    manager.load_plugins()
    result = manager.call_plugin("slow", {}, timeout=1)
    assert "error" in result
    assert "timed out" in result["error"]


def test_call_plugin_uses_docker_backend_when_enabled(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "echo.py").write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()

    with patch("core.plugin_manager.SANDBOX_BACKEND", "docker"), patch(
        "core.plugin_manager.SANDBOX_TIMEOUT", 45
    ), patch(
        "core.plugin_manager.DockerSandboxRunner"
    ) as runner_cls:
        runner = runner_cls.return_value
        runner.run_plugin.return_value = ({"echo": 42}, 1.0, True, None, None)

        result = manager.call_plugin("echo", {"value": 42})

    assert result == {"echo": 42}
    runner.run_plugin.assert_called_once_with("echo", {"value": 42}, 45)


def test_call_plugin_falls_back_to_subprocess_when_docker_not_required(
    manager: PluginManager, plugins_dir: Path
) -> None:
    (plugins_dir / "echo.py").write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()

    with patch("core.plugin_manager.SANDBOX_BACKEND", "docker"), patch(
        "core.plugin_manager.SANDBOX_DOCKER_REQUIRED", False
    ), patch("core.plugin_manager.DockerSandboxRunner") as runner_cls:
        runner = runner_cls.return_value
        runner.run_plugin.return_value = ({"error": "docker down"}, 1.0, False, "DockerError", "docker down")

        result = manager.call_plugin("echo", {"value": 5})

    assert result == {"echo": 5}


def test_add_plugin_writes_and_loads(manager: PluginManager, plugins_dir: Path) -> None:
    result = manager.add_plugin("new_plugin", DUMMY_PLUGIN_CODE)
    assert result.get("status") == "ok"
    assert "new_plugin" in manager.plugins
    assert (plugins_dir / "new_plugin.py").exists()


def test_add_plugin_overwrites_existing(manager: PluginManager, plugins_dir: Path) -> None:
    from unittest.mock import patch

    manager.add_plugin("overwrite", DUMMY_PLUGIN_CODE)
    new_code = textwrap.dedent(
        """\
        def run(input_data):
            return {"version": 2}
        """
    )
    manager.add_plugin("overwrite", new_code)
    with patch("core.plugin_manager.TRUSTED_PLUGINS", ["overwrite"]):
        result = manager.call_plugin("overwrite", {})
    assert result == {"version": 2}


def test_add_plugin_archives_previous_version(manager: PluginManager, plugins_dir: Path) -> None:
    """Overwriting a plugin should create an archive entry."""
    manager.add_plugin("versioned_plugin", DUMMY_PLUGIN_CODE)
    new_code = textwrap.dedent(
        """\
        def run(input_data):
            return {"version": 2}
        """
    )
    manager.add_plugin("versioned_plugin", new_code)
    arch_dir = plugins_dir.parent / "plugins_store" / "archive" / "versioned_plugin"
    assert arch_dir.exists()
    archived = list(arch_dir.glob("v*.py"))
    assert len(archived) >= 1


def test_reload_plugin(manager: PluginManager, plugins_dir: Path) -> None:
    path = plugins_dir / "reloadable.py"
    path.write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()

    # Overwrite file and reload
    path.write_text(
        textwrap.dedent(
            """\
            def run(input_data):
                return {"reloaded": True}
            """
        )
    )
    result = manager.reload_plugin("reloadable")
    assert result.get("status") == "ok"
    assert manager.call_plugin("reloadable", {}) == {"reloaded": True}


def test_reload_plugin_nonexistent_file_returns_error(manager: PluginManager) -> None:
    result = manager.reload_plugin("not_on_disk")
    assert "error" in result


def test_unload_plugin_calls_shutdown(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "shutdownable.py").write_text(DUMMY_PLUGIN_WITH_SHUTDOWN)
    manager.load_plugins()
    result = manager.unload_plugin("shutdownable")
    assert result.get("status") == "ok"
    assert "shutdownable" not in manager.plugins


def test_unload_http_is_forbidden(manager: PluginManager) -> None:
    result = manager.unload_plugin("http")
    assert "error" in result
    assert "protected" in result["error"]


def test_unload_plugin_not_loaded_returns_error(manager: PluginManager) -> None:
    result = manager.unload_plugin("not_loaded_plugin")
    assert "error" in result


def test_get_plugin_returns_none_for_unknown(manager: PluginManager) -> None:
    assert manager.get_plugin("does_not_exist") is None


def test_get_all_plugins_returns_snapshot(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "snap.py").write_text(DUMMY_PLUGIN_CODE)
    manager.load_plugins()
    snapshot = manager.get_all_plugins()
    assert "snap" in snapshot
    # Mutating snapshot should not affect registry
    del snapshot["snap"]
    assert "snap" in manager.plugins


def test_init_is_called_on_load(manager: PluginManager, plugins_dir: Path) -> None:
    (plugins_dir / "withinit.py").write_text(DUMMY_PLUGIN_WITH_INIT)
    manager.load_plugins()
    # After loading, calling run() should show it was initialised.
    result = manager.call_plugin("withinit", {})
    assert result == {"initialised": True}


def test_call_plugin_non_dict_return_is_wrapped(manager: PluginManager, plugins_dir: Path) -> None:
    """A plugin returning a non-dict value should be wrapped in {"result": ...}."""
    code = textwrap.dedent(
        """\
        def run(input_data):
            return 42
        """
    )
    (plugins_dir / "scalar.py").write_text(code)
    manager.load_plugins()
    result = manager.call_plugin("scalar", {})
    assert result == {"result": 42}


def test_call_plugin_raises_exception_returns_error_with_traceback(
    manager: PluginManager, plugins_dir: Path
) -> None:
    """Plugins that raise exceptions should return an error dict with traceback."""
    code = textwrap.dedent(
        """\
        def run(input_data):
            raise ValueError("oops")
        """
    )
    (plugins_dir / "faulty.py").write_text(code)
    manager.load_plugins()
    result = manager.call_plugin("faulty", {})
    assert "error" in result
    assert "traceback" in result
    assert "ValueError" in result["traceback"]


def test_reload_plugin_calls_shutdown_on_old_version(manager: PluginManager, plugins_dir: Path) -> None:
    """reload_plugin() must call shutdown() on the previous instance to prevent resource leaks."""
    shutdown_calls: list[int] = []

    # First version: has a shutdown() that records the call.
    v1_code = textwrap.dedent(
        """\
        import sys as _sys

        def shutdown():
            # Append to the list held in the test scope via the module reference.
            _sys.modules[__name__]._shutdown_calls.append(1)

        def run(input_data):
            return {"version": 1}
        """
    )
    manager.add_plugin("versioned", v1_code)
    # Inject the shared list into the loaded module so shutdown() can record the call.
    manager.plugins["versioned"]._shutdown_calls = shutdown_calls  # type: ignore[attr-defined]

    v2_code = textwrap.dedent(
        """\
        def run(input_data):
            return {"version": 2}
        """
    )
    # Reload should call shutdown() on v1 before replacing it.
    (plugins_dir / "versioned.py").write_text(v2_code)
    manager.reload_plugin("versioned")

    assert len(shutdown_calls) == 1, "shutdown() should have been called exactly once during reload"
    assert manager.call_plugin("versioned", {}) == {"version": 2}


def test_rollback_plugin_no_archive_returns_error(manager: PluginManager) -> None:
    result = manager.rollback_plugin("no_such_plugin")
    assert "error" in result


def test_rollback_plugin_restores_previous_version(manager: PluginManager, plugins_dir: Path) -> None:
    """After adding two versions, rollback should restore the archived (v1) code."""
    from unittest.mock import patch as mpatch

    v1_code = textwrap.dedent(
        """\
        def run(input_data):
            return {"version": 1}
        """
    )
    v2_code = textwrap.dedent(
        """\
        def run(input_data):
            return {"version": 2}
        """
    )
    manager.add_plugin("rollme", v1_code)
    manager.add_plugin("rollme", v2_code)

    # Use in-process (trusted) execution to bypass subprocess .pyc caching.
    with mpatch("core.plugin_manager.TRUSTED_PLUGINS", ["rollme"]):
        assert manager.call_plugin("rollme", {}) == {"version": 2}

        result = manager.rollback_plugin("rollme")
        assert result.get("status") == "ok"
        assert manager.call_plugin("rollme", {}) == {"version": 1}


def test_rollback_plugin_empty_archive_returns_error(manager: PluginManager, plugins_dir: Path) -> None:
    """Archive directory exists but has no versioned .py files."""
    arch_dir = plugins_dir.parent / "plugins_store" / "archive" / "myplug"
    arch_dir.mkdir(parents=True)
    result = manager.rollback_plugin("myplug")
    assert "error" in result


def teardown_module(module: object) -> None:
    # Clean up any lingering test modules from sys.modules
    to_remove = [k for k in sys.modules if k.startswith("plugins.")]
    for k in to_remove:
        del sys.modules[k]
