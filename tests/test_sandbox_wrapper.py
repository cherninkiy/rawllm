"""Tests for core.sandbox_wrapper – subprocess plugin isolation."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_sandbox(plugin_code: str, input_data: dict, tmp_path: Path) -> dict:
    """Helper: write plugin to a temp file and invoke the sandbox wrapper."""
    plugin_path = tmp_path / "test_plugin.py"
    plugin_path.write_text(plugin_code, encoding="utf-8")

    payload = json.dumps({"plugin_path": str(plugin_path), "input_data": input_data})
    proc = subprocess.run(
        [sys.executable, "-m", "core.sandbox_wrapper"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(proc.stdout)


def test_sandbox_runs_simple_plugin(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """\
        def run(input_data):
            return {"echo": input_data.get("value")}
        """
    )
    result = _run_sandbox(code, {"value": 42}, tmp_path)
    assert result == {"result": {"echo": 42}}


def test_sandbox_returns_error_on_exception(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """\
        def run(input_data):
            raise ValueError("something went wrong")
        """
    )
    result = _run_sandbox(code, {}, tmp_path)
    assert "error" in result
    assert "ValueError" in result["error"]


def test_sandbox_returns_error_for_missing_run(tmp_path: Path) -> None:
    code = "x = 1  # no run() function\n"
    result = _run_sandbox(code, {}, tmp_path)
    assert "error" in result


def test_sandbox_calls_init_if_present(tmp_path: Path) -> None:
    """init() with no required args should be called automatically."""
    code = textwrap.dedent(
        """\
        _initialized = False

        def init():
            global _initialized
            _initialized = True

        def run(input_data):
            return {"initialized": _initialized}
        """
    )
    result = _run_sandbox(code, {}, tmp_path)
    assert result == {"result": {"initialized": True}}


def test_sandbox_skips_init_with_required_args(tmp_path: Path) -> None:
    """init(callback) requiring arguments should be silently skipped."""
    code = textwrap.dedent(
        """\
        def init(callback):
            callback("something")

        def run(input_data):
            return {"ok": True}
        """
    )
    result = _run_sandbox(code, {}, tmp_path)
    assert result == {"result": {"ok": True}}


def test_sandbox_handles_invalid_json_input(tmp_path: Path) -> None:
    """Sending invalid JSON to stdin should produce an error in stdout."""
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text("def run(d): return {}\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "core.sandbox_wrapper"],
        input="NOT VALID JSON",
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads(proc.stdout)
    assert "error" in result


def test_sandbox_handles_missing_plugin_path(tmp_path: Path) -> None:
    """A plugin_path pointing to a non-existent file should produce an error."""
    payload = json.dumps(
        {"plugin_path": str(tmp_path / "does_not_exist.py"), "input_data": {}}
    )
    proc = subprocess.run(
        [sys.executable, "-m", "core.sandbox_wrapper"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads(proc.stdout)
    assert "error" in result


def test_sandbox_does_not_write_pyc(tmp_path: Path) -> None:
    """sandbox_wrapper must not create any .pyc files (sys.dont_write_bytecode=True)."""
    code = "def run(input_data): return {}\n"
    plugin_path = tmp_path / "no_cache_plugin.py"
    plugin_path.write_text(code, encoding="utf-8")

    _run_sandbox(code, {}, tmp_path)

    pycache = tmp_path / "__pycache__"
    if pycache.exists():
        pyc_files = list(pycache.glob("*.pyc"))
        # There should be no .pyc files for this plugin.
        assert pyc_files == [], f"Unexpected .pyc files created: {pyc_files}"


def test_sandbox_reflects_rapid_update(tmp_path: Path) -> None:
    """Rapid plugin updates must be reflected immediately in subprocess execution.

    This regression test guards against stale .pyc bytecode being loaded when
    a plugin source file is overwritten within the same filesystem-mtime second.
    We simulate this by forcing both versions to have the same mtime via os.utime.
    """
    import os

    plugin_path = tmp_path / "rapid_plugin.py"

    v1_code = "def run(input_data): return {'version': 1}\n"
    v2_code = "def run(input_data): return {'version': 2}\n"

    # Write v1 and execute – establishes any cached state.
    plugin_path.write_text(v1_code, encoding="utf-8")
    payload_v1 = json.dumps({"plugin_path": str(plugin_path), "input_data": {}})
    proc1 = subprocess.run(
        [sys.executable, "-m", "core.sandbox_wrapper"],
        input=payload_v1,
        capture_output=True,
        text=True,
        timeout=10,
    )
    r1 = json.loads(proc1.stdout)
    assert r1 == {"result": {"version": 1}}, f"v1 returned unexpected result: {r1}"

    # Overwrite with v2 but force the same mtime as v1 to simulate same-second update.
    original_mtime = os.stat(plugin_path).st_mtime
    plugin_path.write_text(v2_code, encoding="utf-8")
    os.utime(plugin_path, (original_mtime, original_mtime))

    # The sandbox must still load v2 despite the identical mtime.
    payload_v2 = json.dumps({"plugin_path": str(plugin_path), "input_data": {}})
    proc2 = subprocess.run(
        [sys.executable, "-m", "core.sandbox_wrapper"],
        input=payload_v2,
        capture_output=True,
        text=True,
        timeout=10,
    )
    r2 = json.loads(proc2.stdout)
    assert r2 == {"result": {"version": 2}}, (
        f"Stale bytecode detected: sandbox returned {r2} instead of version 2. "
        "This means sys.dont_write_bytecode is not working correctly."
    )
