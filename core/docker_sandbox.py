"""Docker-backed sandbox runner for untrusted plugins.

This backend executes plugin code in a container under ``rawllm-plugin`` user
with strict mounts:
- workspace (rw volume)
- core repo snapshot (ro volume)
- plugin store snapshot (ro volume)

The container root filesystem is read-only and network access is disabled.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from core.config import (
    SANDBOX_CORE_REPO_MOUNT,
    SANDBOX_CORE_REPO_SOURCE,
    SANDBOX_CORE_REPO_VOLUME,
    SANDBOX_DOCKER_IMAGE,
    SANDBOX_DOCKER_SYNC_IMAGE,
    SANDBOX_PLUGIN_STORE_MOUNT,
    SANDBOX_PLUGIN_STORE_SOURCE,
    SANDBOX_PLUGIN_STORE_VOLUME,
    SANDBOX_PLUGIN_USER,
    SANDBOX_STAGING_DIR,
    SANDBOX_WORKSPACE_MOUNT,
    SANDBOX_WORKSPACE_VOLUME,
)


class DockerSandboxError(RuntimeError):
    """Raised when docker-backed sandbox orchestration fails."""


class DockerSandboxRunner:
    """Prepare docker volumes and run plugin sandbox calls in Docker."""

    _prepare_lock = threading.Lock()
    _prepared_signature: tuple[str, int, str, int] | None = None

    def __init__(self, plugins_dir: Path) -> None:
        self._plugins_dir = plugins_dir.resolve()
        self._repo_root = self._plugins_dir.parent.resolve()
        self._staging_root = Path(SANDBOX_STAGING_DIR).resolve()
        self._staging_root.mkdir(parents=True, exist_ok=True)

    def run_plugin(
        self,
        name: str,
        input_data: dict[str, Any],
        timeout: int,
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None]:
        """Run plugin *name* in Docker and return execution tuple."""
        import time

        start = time.monotonic()
        try:
            self._prepare_volumes()

            plugin_path = f"{SANDBOX_PLUGIN_STORE_MOUNT}/{name}.py"
            payload = json.dumps({"plugin_path": plugin_path, "input_data": input_data})

            cmd = [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                SANDBOX_PLUGIN_USER,
                "--mount",
                f"type=volume,src={SANDBOX_WORKSPACE_VOLUME},dst={SANDBOX_WORKSPACE_MOUNT}",
                "--mount",
                f"type=volume,src={SANDBOX_CORE_REPO_VOLUME},dst={SANDBOX_CORE_REPO_MOUNT},readonly",
                "--mount",
                f"type=volume,src={SANDBOX_PLUGIN_STORE_VOLUME},dst={SANDBOX_PLUGIN_STORE_MOUNT},readonly",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-e",
                f"PYTHONPATH={SANDBOX_CORE_REPO_MOUNT}",
                SANDBOX_DOCKER_IMAGE,
                "python",
                "-m",
                "core.sandbox_wrapper",
            ]

            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            exec_ms = (time.monotonic() - start) * 1000
            return {"error": f"Plugin {name!r} timed out after {timeout}s."}, exec_ms, False, "TimeoutError", None
        except (DockerSandboxError, OSError) as exc:
            exec_ms = (time.monotonic() - start) * 1000
            return {"error": str(exc)}, exec_ms, False, type(exc).__name__, str(exc)

        exec_ms = (time.monotonic() - start) * 1000

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "docker run failed"
            return {"error": err}, exec_ms, False, "DockerError", err

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            tb_str = proc.stderr or proc.stdout
            return {"error": f"Sandbox produced invalid output: {tb_str[:500]}"}, exec_ms, False, "JSONDecodeError", tb_str

        if "error" in data:
            tb_str = data["error"]
            return {"error": tb_str, "traceback": tb_str}, exec_ms, False, "RuntimeError", tb_str

        result = data.get("result", data)
        return result if isinstance(result, dict) else {"result": result}, exec_ms, True, None, None

    def _prepare_volumes(self) -> None:
        self._ensure_docker_available()

        core_source = self._resolve_source_dir(SANDBOX_CORE_REPO_SOURCE, fallback=self._repo_root)
        plugin_source = self._resolve_source_dir(SANDBOX_PLUGIN_STORE_SOURCE, fallback=self._plugins_dir)

        signature = (
            str(core_source),
            self._source_mtime_ns(core_source),
            str(plugin_source),
            self._source_mtime_ns(plugin_source),
        )

        with self._prepare_lock:
            if self.__class__._prepared_signature == signature:
                return

            run_staging = Path(tempfile.mkdtemp(prefix="rawllm_staging_", dir=str(self._staging_root)))
            try:
                core_staging = run_staging / "core_repo"
                plugin_staging = run_staging / "plugin_store"
                self._copy_tree(core_source, core_staging)
                self._copy_tree(plugin_source, plugin_staging)

                self._init_workspace_volume()
                self._sync_volume_from_dir(SANDBOX_CORE_REPO_VOLUME, core_staging)
                self._sync_volume_from_dir(SANDBOX_PLUGIN_STORE_VOLUME, plugin_staging)
                self.__class__._prepared_signature = signature
            finally:
                shutil.rmtree(run_staging, ignore_errors=True)

    def _source_mtime_ns(self, source: Path) -> int:
        latest = source.stat().st_mtime_ns
        for path in source.rglob("*"):
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
        return latest

    def _resolve_source_dir(self, configured: str, fallback: Path) -> Path:
        p = Path(configured)
        if not p.is_absolute():
            p = (self._repo_root / p).resolve()
        if not p.exists() or not p.is_dir():
            return fallback
        return p

    def _copy_tree(self, src: Path, dst: Path) -> None:
        # Guard against copying a source tree into its own descendant.
        try:
            src.relative_to(dst)
            raise DockerSandboxError(f"Invalid staging path: source {src} is nested under destination {dst}")
        except ValueError:
            pass
        try:
            dst.relative_to(src)
            raise DockerSandboxError(f"Invalid staging path: destination {dst} is nested under source {src}")
        except ValueError:
            pass

        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)

        staging_name = self._staging_root.name
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".venv",
                "venv",
                staging_name,
            ),
            dirs_exist_ok=False,
        )

    def _ensure_docker_available(self) -> None:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "docker info failed"
            raise DockerSandboxError(f"Docker is unavailable: {err}")

    def _ensure_volume(self, name: str) -> None:
        proc = subprocess.run(
            ["docker", "volume", "create", name],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "volume create failed"
            raise DockerSandboxError(f"Failed to create volume {name!r}: {err}")

    def _init_workspace_volume(self) -> None:
        self._ensure_volume(SANDBOX_WORKSPACE_VOLUME)
        script = (
            "mkdir -p /workspace && "
            "chown -R \"$PLUGIN_USER\" /workspace 2>/dev/null || true; "
            "chmod 0700 /workspace"
        )
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "root",
                "-e",
                f"PLUGIN_USER={SANDBOX_PLUGIN_USER}",
                "-v",
                f"{SANDBOX_WORKSPACE_VOLUME}:/workspace",
                SANDBOX_DOCKER_IMAGE,
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "workspace init failed"
            raise DockerSandboxError(f"Failed to initialize workspace volume {SANDBOX_WORKSPACE_VOLUME!r}: {err}")

    def _sync_volume_from_dir(self, volume_name: str, source_dir: Path) -> None:
        self._ensure_volume(volume_name)
        source_mount = source_dir.as_posix()
        script = "find /target -mindepth 1 -delete; cp -a /source/. /target/"
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:/target",
                "-v",
                f"{source_mount}:/source:ro",
                SANDBOX_DOCKER_SYNC_IMAGE,
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "volume sync failed"
            raise DockerSandboxError(f"Failed to sync volume {volume_name!r}: {err}")
