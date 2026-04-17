from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.docker_sandbox import DockerSandboxError, DockerSandboxRunner


def _runner(tmp_path: Path) -> DockerSandboxRunner:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True)
    DockerSandboxRunner._prepared_signature = None
    return DockerSandboxRunner(plugins_dir)


def test_run_plugin_success(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with patch.object(runner, "_prepare_volumes"), patch(
        "core.docker_sandbox.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout='{"result": {"ok": true}}', stderr=""),
    ):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {"v": 1}, 5)

    assert ok is True
    assert result == {"ok": True}
    assert err_type is None
    assert tb is None


def test_run_plugin_timeout(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with patch.object(runner, "_prepare_volumes"), patch(
        "core.docker_sandbox.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=5),
    ):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {}, 5)

    assert ok is False
    assert "timed out" in result["error"]
    assert err_type == "TimeoutError"
    assert tb is None


def test_run_plugin_prepare_volumes_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with patch.object(runner, "_prepare_volumes", side_effect=DockerSandboxError("docker down")):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {}, 5)

    assert ok is False
    assert result == {"error": "docker down"}
    assert err_type == "DockerSandboxError"
    assert tb == "docker down"


def test_run_plugin_nonzero_docker_exit(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    with patch.object(runner, "_prepare_volumes"), patch("core.docker_sandbox.subprocess.run", return_value=proc):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {}, 5)

    assert ok is False
    assert result == {"error": "boom"}
    assert err_type == "DockerError"
    assert tb == "boom"


def test_run_plugin_invalid_json_output(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=0, stdout="not-json", stderr="trace")
    with patch.object(runner, "_prepare_volumes"), patch("core.docker_sandbox.subprocess.run", return_value=proc):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {}, 5)

    assert ok is False
    assert "Sandbox produced invalid output" in result["error"]
    assert err_type == "JSONDecodeError"
    assert tb == "trace"


def test_run_plugin_runtime_error_payload(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=0, stdout='{"error": "plugin failed"}', stderr="")
    with patch.object(runner, "_prepare_volumes"), patch("core.docker_sandbox.subprocess.run", return_value=proc):
        result, _ms, ok, err_type, tb = runner.run_plugin("echo", {}, 5)

    assert ok is False
    assert result["error"] == "plugin failed"
    assert result["traceback"] == "plugin failed"
    assert err_type == "RuntimeError"
    assert tb == "plugin failed"


def test_prepare_volumes_calls_expected_steps(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    with patch.object(runner, "_ensure_docker_available") as p_docker, patch.object(
        runner, "_resolve_source_dir", side_effect=[tmp_path, tmp_path]
    ) as p_resolve, patch.object(runner, "_copy_tree") as p_copy, patch.object(
        runner, "_init_workspace_volume"
    ) as p_init_ws, patch.object(runner, "_sync_volume_from_dir") as p_sync:
        runner._prepare_volumes()

    assert p_docker.called
    assert p_resolve.call_count == 2
    assert p_copy.call_count == 2
    p_init_ws.assert_called_once()
    assert p_sync.call_count == 2


def test_prepare_volumes_uses_cache_when_sources_unchanged(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    with patch.object(runner, "_ensure_docker_available"), patch.object(
        runner,
        "_resolve_source_dir",
        side_effect=[tmp_path, tmp_path, tmp_path, tmp_path],
    ), patch.object(
        runner,
        "_source_mtime_ns",
        side_effect=[111, 222, 111, 222],
    ), patch.object(
        runner, "_copy_tree"
    ) as p_copy, patch.object(
        runner, "_init_workspace_volume"
    ) as p_init_ws, patch.object(runner, "_sync_volume_from_dir") as p_sync:
        runner._prepare_volumes()
        runner._prepare_volumes()

    assert p_copy.call_count == 2
    assert p_sync.call_count == 2
    p_init_ws.assert_called_once()


def test_resolve_source_dir_relative_and_fallback(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    existing = tmp_path / "src"
    existing.mkdir()
    resolved = runner._resolve_source_dir("src", fallback=tmp_path / "fallback")
    assert resolved == existing.resolve()

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    resolved_missing = runner._resolve_source_dir("missing", fallback=fallback)
    assert resolved_missing == fallback


def test_copy_tree_replaces_destination(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")

    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "old.txt").write_text("old", encoding="utf-8")

    runner._copy_tree(src, dst)

    assert (dst / "a.txt").read_text(encoding="utf-8") == "x"
    assert not (dst / "old.txt").exists()


def test_ensure_docker_available_raises_on_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=1, stdout="", stderr="denied")
    with patch("core.docker_sandbox.subprocess.run", return_value=proc):
        with pytest.raises(DockerSandboxError, match="Docker is unavailable"):
            runner._ensure_docker_available()


def test_ensure_volume_raises_on_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=1, stdout="", stderr="no space")
    with patch("core.docker_sandbox.subprocess.run", return_value=proc):
        with pytest.raises(DockerSandboxError, match="Failed to create volume"):
            runner._ensure_volume("vol")


def test_sync_volume_from_dir_raises_on_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    source = tmp_path / "source"
    source.mkdir()

    proc = SimpleNamespace(returncode=1, stdout="", stderr="sync failed")
    with patch.object(runner, "_ensure_volume"), patch("core.docker_sandbox.subprocess.run", return_value=proc):
        with pytest.raises(DockerSandboxError, match="Failed to sync volume"):
            runner._sync_volume_from_dir("vol", source)


def test_init_workspace_volume_raises_on_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=1, stdout="", stderr="init failed")
    with patch.object(runner, "_ensure_volume"), patch("core.docker_sandbox.subprocess.run", return_value=proc):
        with pytest.raises(DockerSandboxError, match="Failed to initialize workspace volume"):
            runner._init_workspace_volume()


def test_init_workspace_volume_uses_restricted_permissions(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    proc = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch.object(runner, "_ensure_volume"), patch("core.docker_sandbox.subprocess.run", return_value=proc) as p_run:
        runner._init_workspace_volume()

    called_cmd = p_run.call_args[0][0]
    assert called_cmd[-1].endswith("chmod 0700 /workspace")


def test_copy_tree_rejects_nested_destination(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    dst = src / "nested"

    with pytest.raises(DockerSandboxError, match="nested"):
        runner._copy_tree(src, dst)
