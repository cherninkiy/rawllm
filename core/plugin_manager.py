"""Plugin manager: load, reload, call, and unload plugins dynamically.

Enhancements over the original:
- Hybrid execution: trusted plugins run in-process; all others run via subprocess.
- Plugin versioning: old versions are archived under plugins_store/.
- Dependency gating: imports outside ALLOWED_REQUIREMENTS are written to
  pending_requirements.txt and the plugin is not executed until approved.
- Metrics: every execution is logged to metrics.jsonl via core.metrics.
- Full traceback returned on error (both in-process and subprocess paths).
"""

import importlib
import importlib.util
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncio
from typing import List, Tuple

import core.config as runtime_config
import core.metrics as metrics
from core.config import ALLOWED_REQUIREMENTS, SANDBOX_BACKEND, SANDBOX_DOCKER_REQUIRED, SANDBOX_TIMEOUT, TRUSTED_PLUGINS
from core.docker_sandbox import DockerSandboxRunner
from core.utils import extract_imports

logger = logging.getLogger(__name__)

PROTECTED_PLUGINS: set[str] = set()


def _touch_future(path: Path) -> None:
    """Invalidate Python's .pyc cache for *path* and set mtime into the future.

    Python caches compiled bytecode using (mtime, size) of the source file.  If a
    plugin file is rewritten within the same second the OS-level mtime doesn't
    change, so Python re-uses the stale .pyc.  This helper deletes the .pyc and
    sets the source mtime one second ahead to guarantee recompilation.

    The +1 s offset assumes the system clock is not adjusted backwards during an
    orchestrator session (a safe assumption for a POC).  Increase the offset or
    switch to a hash-based invalidation scheme if sub-second forward-clock-skew
    ever becomes a concern.
    """
    import importlib.util as _iu
    # Delete the existing .pyc to force recompilation.
    try:
        pyc_path = _iu.cache_from_source(str(path))
        Path(pyc_path).unlink(missing_ok=True)
    except (NotImplementedError, ValueError, OSError):
        pass
    importlib.invalidate_caches()
    # Also advance the mtime to ensure any remaining cached mtime is stale.
    future = time.time() + 1
    os.utime(path, (future, future))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_risk_score(code: str) -> int:
    """Count of imports that are outside ALLOWED_REQUIREMENTS."""
    imports = extract_imports(code)
    return sum(1 for imp in imports if imp not in ALLOWED_REQUIREMENTS)


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------


def _plugins_store(plugins_dir: Path) -> Path:
    """Return the plugins_store directory (sibling of plugins_dir)."""
    return plugins_dir.parent / "plugins_store"


def _current_dir(plugins_dir: Path) -> Path:
    return _plugins_store(plugins_dir) / "current"


def _archive_dir(plugins_dir: Path, name: str) -> Path:
    return _plugins_store(plugins_dir) / "archive" / name


def _version_json_path(plugins_dir: Path, name: str) -> Path:
    return _archive_dir(plugins_dir, name) / "version.json"


def _resource_assignments_path(plugins_dir: Path) -> Path:
    return _plugins_store(plugins_dir) / "resource_assignments.json"


def _read_version_meta(plugins_dir: Path, name: str) -> dict[str, Any]:
    path = _version_json_path(plugins_dir, name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"current_version": 0, "rollback_count": 0}


def _write_version_meta(plugins_dir: Path, name: str, meta: dict[str, Any]) -> None:
    path = _version_json_path(plugins_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _archive_plugin(plugins_dir: Path, name: str, code: str, metrics_snapshot: dict[str, Any]) -> str:
    """Archive *code* as a new versioned file.  Returns the new version string."""
    meta = _read_version_meta(plugins_dir, name)
    version_num = meta["current_version"] + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_str = f"v{version_num}_{timestamp}"

    arch_dir = _archive_dir(plugins_dir, name)
    arch_dir.mkdir(parents=True, exist_ok=True)

    (arch_dir / f"{version_str}.py").write_text(code, encoding="utf-8")
    (arch_dir / f"{version_str}_metrics.json").write_text(
        json.dumps(metrics_snapshot, indent=2), encoding="utf-8"
    )

    meta["current_version"] = version_num
    meta["last_version_str"] = version_str
    _write_version_meta(plugins_dir, name, meta)

    # Maintain symlink (or copy on platforms without symlink support).
    _update_current_symlink(plugins_dir, name, plugins_dir / f"{name}.py")

    return version_str


def _update_current_symlink(plugins_dir: Path, name: str, target: Path) -> None:
    """Point plugins_store/current/{name}.py at *target*."""
    current = _current_dir(plugins_dir)
    current.mkdir(parents=True, exist_ok=True)
    link = current / f"{name}.py"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve())
    except (OSError, NotImplementedError):
        # Windows without developer mode – fall back to a plain copy.
        import shutil
        shutil.copy2(target, link)


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------


class PluginManager:
    """Manages the lifecycle of plugins: discovery, loading, execution, and unloading."""

    def __init__(self, plugins_dir: Path) -> None:
        """Initialise the manager pointing at *plugins_dir*."""
        self.plugins_dir = plugins_dir
        self.plugins: dict[str, ModuleType] = {}
        self._lock = threading.RLock()
        # In-memory metrics snapshot per plugin (updated after each execution).
        self._exec_counts: dict[str, dict[str, Any]] = {}
        self._docker_runner: DockerSandboxRunner | None = None
        self._resource_assignments: dict[str, dict[str, Any]] = self._load_resource_assignments()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_plugins(self) -> None:
        """Scan *plugins_dir* for .py files and import each one.

        For every plugin that exposes an ``init()`` function (without arguments),
        that function is called immediately.  The ``http`` plugin is intentionally
        skipped here – its ``init(callback)`` is called from ``run.py`` once the
        TAOR loop is ready.
        """
        for path in sorted(self.plugins_dir.glob("*.py")):
            name = path.stem
            try:
                with self._lock:
                    self._import_plugin(name, path)
            except Exception:
                logger.exception("Failed to load plugin %r", name)

    def call_plugin(self, name: str, input_data: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
        """Execute ``plugin.run(input_data)`` and return the result.

        Trusted plugins (in TRUSTED_PLUGINS or PROTECTED_PLUGINS) run in-process.
        All others run in an isolated subprocess via ``core.sandbox_wrapper``.

        Returns the result dict, or an error dict (with full traceback) on failure.
        """
        with self._lock:
            plugin = self.get_plugin(name)
            if plugin is None:
                return {"error": f"Plugin {name!r} is not loaded."}

        version_str = self._current_version_str(name)
        is_trusted = name in TRUSTED_PLUGINS or name in PROTECTED_PLUGINS
        env = self._get_plugin_env(name)

        if is_trusted:
            result, exec_ms, success, err_type, tb_str = self._call_inprocess(
                name,
                plugin,
                input_data,
                timeout,
                env,
            )
        else:
            result, exec_ms, success, err_type, tb_str = self._call_subprocess(
                name,
                input_data,
                timeout,
                env,
            )

        risk = _import_risk_score(self._read_plugin_code(name))
        metrics.log_execution(
            plugin_name=name,
            version=version_str,
            execution_time_ms=exec_ms,
            success=success,
            error_type=err_type,
            traceback_str=tb_str,
            import_risk_score=risk,
        )
        self._exec_counts[name] = {"last_exec_ms": exec_ms, "success": success}
        return result

    def add_plugin(self, name: str, code: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write *code* to ``plugins/{name}.py`` and hot-reload it.

        Archives the previous version (if any) and updates symlinks in
        ``plugins_store/current/``.
        """
        path = self.plugins_dir / f"{name}.py"
        old_assignment = self._resource_assignments.get(name)
        assignment_result = self._resolve_manifest_resources(name, manifest)
        if "error" in assignment_result:
            return assignment_result

        assigned_resources = assignment_result["assigned"]
        try:
            # Archive existing version before overwriting.
            if path.exists():
                old_code = path.read_text(encoding="utf-8")
                snapshot = self._exec_counts.get(name, {})
                old_ver = self._current_version_str(name)
                new_ver = _archive_plugin(self.plugins_dir, name, old_code, snapshot)
                metrics.log_version_change(name, old_ver, new_ver)

            path.write_text(code, encoding="utf-8")
            # Ensure mtime differs from any cached .pyc so Python recompiles on import.
            _touch_future(path)
            result = self.reload_plugin(name)
            if "status" in result:
                if assigned_resources:
                    self._resource_assignments[name] = assigned_resources
                else:
                    self._resource_assignments.pop(name, None)
                self._save_resource_assignments()
                _update_current_symlink(self.plugins_dir, name, path)
                result["assigned"] = assigned_resources
            return result
        except Exception as exc:
            if old_assignment is not None:
                self._resource_assignments[name] = old_assignment
            else:
                self._resource_assignments.pop(name, None)
            self._save_resource_assignments()
            logger.exception("Failed to add plugin %r", name)
            return {"error": str(exc)}

    def reload_plugin(self, name: str) -> dict[str, Any]:
        """Shut down the old instance (if any), then re-import and update the registry."""
        path = self.plugins_dir / f"{name}.py"
        if not path.exists():
            return {"error": f"Plugin file {path} does not exist."}

        with self._lock:
            old_plugin = self.plugins.get(name)
            if old_plugin is not None:
                old_shutdown = getattr(old_plugin, "shutdown", None)
                if old_shutdown is not None:
                    try:
                        old_shutdown()
                    except Exception:
                        logger.exception("Plugin %r shutdown() raised during reload", name)

            module_name = f"plugins.{name}"
            sys.modules.pop(module_name, None)

            try:
                self._import_plugin(name, path)
                return {"status": "ok", "plugin": name}
            except Exception as exc:
                logger.exception("Failed to reload plugin %r", name)
                return {"error": str(exc)}

    def unload_plugin(self, name: str) -> dict[str, Any]:
        """Shut down and remove *name* from the registry (file stays on disk)."""
        if name in PROTECTED_PLUGINS:
            return {"error": f"Plugin {name!r} is protected and cannot be unloaded."}

        with self._lock:
            plugin = self.get_plugin(name)
            if plugin is None:
                return {"error": f"Plugin {name!r} is not loaded."}

            shutdown_fn = getattr(plugin, "shutdown", None)
            if shutdown_fn is not None:
                try:
                    shutdown_fn()
                except Exception:
                    logger.exception("Plugin %r shutdown() raised an exception", name)

            del self.plugins[name]
            sys.modules.pop(f"plugins.{name}", None)
            self._resource_assignments.pop(name, None)
            self._save_resource_assignments()

        logger.info("Plugin %r unloaded.", name)
        return {"status": "ok", "plugin": name}

    def rollback_plugin(self, name: str) -> dict[str, Any]:
        """Switch to the previous archived version of *name*.

        Updates the symlink in ``plugins_store/current/`` and hot-reloads.
        """
        arch_dir = _archive_dir(self.plugins_dir, name)
        if not arch_dir.exists():
            return {"error": f"No archive found for plugin {name!r}."}

        # Collect all versioned .py files (exclude *_metrics.json companions).
        versions = sorted(arch_dir.glob("v*_*.py"))
        if not versions:
            return {"error": f"No archived versions found for plugin {name!r}."}

        prev_path = versions[-1]  # most recent archive = the one just before current
        meta = _read_version_meta(self.plugins_dir, name)
        from_ver = self._current_version_str(name)

        # Copy archived file back to active location.
        active_path = self.plugins_dir.joinpath(f"{name}.py")
        active_path.write_text(prev_path.read_text(encoding="utf-8"), encoding="utf-8")
        _touch_future(active_path)
        _update_current_symlink(self.plugins_dir, name, active_path)

        meta["rollback_count"] = meta.get("rollback_count", 0) + 1
        _write_version_meta(self.plugins_dir, name, meta)
        metrics.log_rollback(name, from_ver, prev_path.stem)

        return self.reload_plugin(name)

    def get_plugin(self, name: str) -> ModuleType | None:
        """Return the loaded module for *name*, or ``None`` if not found."""
        return self.plugins.get(name)

    def get_all_plugins(self) -> dict[str, ModuleType]:
        """Return a snapshot copy of the current plugin registry (thread-safe)."""
        with self._lock:
            return dict(self.plugins)

    def get_resource_assignments(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._resource_assignments))

    def get_resource_assignment(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            assignment = self._resource_assignments.get(name)
            if assignment is None:
                return None
            return json.loads(json.dumps(assignment))

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def call_plugin_async(
        self, name: str, input_data: dict[str, Any], timeout: int = 30
    ) -> dict[str, Any]:
        """Async wrapper for :meth:`call_plugin`."""
        return await asyncio.to_thread(self.call_plugin, name, input_data, timeout)

    async def add_plugin_async(
        self,
        name: str,
        code: str,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async wrapper for :meth:`add_plugin`."""
        return await asyncio.to_thread(self.add_plugin, name, code, manifest)

    async def unload_plugin_async(self, name: str) -> dict[str, Any]:
        """Async wrapper for :meth:`unload_plugin`."""
        return await asyncio.to_thread(self.unload_plugin, name)

    async def reload_plugin_async(self, name: str) -> dict[str, Any]:
        """Async wrapper for :meth:`reload_plugin`."""
        return await asyncio.to_thread(self.reload_plugin, name)

    async def rollback_plugin_async(self, name: str) -> dict[str, Any]:
        """Async wrapper for :meth:`rollback_plugin`."""
        return await asyncio.to_thread(self.rollback_plugin, name)

    async def call_plugins_parallel(
        self,
        calls: List[Tuple[str, dict[str, Any]]],
        timeout: int = 30,
    ) -> List[dict[str, Any]]:
        """Execute multiple plugins concurrently and return results in order."""
        return list(
            await asyncio.gather(
                *[self.call_plugin_async(name, inp, timeout) for name, inp in calls]
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_resource_assignments(self) -> dict[str, dict[str, Any]]:
        path = _resource_assignments_path(self.plugins_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read persisted resource assignments")
            return {}
        return data if isinstance(data, dict) else {}

    def _save_resource_assignments(self) -> None:
        path = _resource_assignments_path(self.plugins_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(self._resource_assignments, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to persist resource assignments")

    def _validate_manifest(self, manifest: dict[str, Any] | None) -> dict[str, Any] | None:
        if manifest is None:
            return None
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object.")

        validated: dict[str, Any] = {}
        for section in ("requires", "publishes"):
            raw_section = manifest.get(section, {})
            if raw_section is None:
                raw_section = {}
            if not isinstance(raw_section, dict):
                raise ValueError(f"Manifest section {section!r} must be an object.")
            unknown_keys = set(raw_section) - {"ports", "volumes", "services"}
            if unknown_keys:
                raise ValueError(
                    f"Manifest section {section!r} contains unsupported keys: {sorted(unknown_keys)}."
                )

            validated_section: dict[str, Any] = {}
            ports = raw_section.get("ports", [])
            volumes = raw_section.get("volumes", [])
            services = raw_section.get("services", [])

            if not isinstance(ports, list) or any(not isinstance(port, int) for port in ports):
                raise ValueError(f"Manifest section {section!r}.ports must be a list of integers.")
            if not isinstance(volumes, list) or any(not isinstance(volume, str) for volume in volumes):
                raise ValueError(f"Manifest section {section!r}.volumes must be a list of strings.")
            if not isinstance(services, list) or any(not isinstance(service, str) for service in services):
                raise ValueError(f"Manifest section {section!r}.services must be a list of strings.")

            validated_section["ports"] = list(dict.fromkeys(ports))
            validated_section["volumes"] = list(dict.fromkeys(volumes))
            validated_section["services"] = list(dict.fromkeys(services))
            validated[section] = validated_section

        return validated

    def _resolve_manifest_resources(
        self,
        name: str,
        manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            validated_manifest = self._validate_manifest(manifest)
        except ValueError as exc:
            return {"error": str(exc)}

        if validated_manifest is None:
            return {"status": "ok", "assigned": {}}

        requires = validated_manifest["requires"]
        publishes = validated_manifest["publishes"]
        requested_ports = list(dict.fromkeys(requires["ports"] + publishes["ports"]))
        requested_volumes = requires["volumes"]
        requested_services = requires["services"]

        other_assignments = {
            plugin_name: assignment
            for plugin_name, assignment in self._resource_assignments.items()
            if plugin_name != name
        }
        reserved_ports = {
            port
            for assignment in other_assignments.values()
            for port in assignment.get("ports", [])
            if isinstance(port, int)
        }

        errors: list[str] = []
        assigned: dict[str, Any] = {}

        unavailable_ports = [
            port for port in requested_ports if port not in runtime_config.AVAILABLE_PORTS
        ]
        if unavailable_ports:
            errors.append(
                f"Requested ports are not available: {sorted(unavailable_ports)}."
            )
        busy_ports = [port for port in requested_ports if port in reserved_ports]
        if busy_ports:
            errors.append(f"Requested ports are already reserved: {sorted(busy_ports)}.")
        if requested_ports and not unavailable_ports and not busy_ports:
            assigned["ports"] = requested_ports

        if requested_volumes:
            workspace_root = runtime_config.WORKSPACE_PATH.resolve()
            invalid_volumes: list[str] = []
            for volume in requested_volumes:
                volume_path = Path(volume)
                if not volume_path.is_absolute():
                    volume_path = workspace_root / volume_path
                try:
                    volume_path.resolve().relative_to(workspace_root)
                except ValueError:
                    invalid_volumes.append(volume)
            if invalid_volumes:
                errors.append(
                    f"Requested volumes must stay under {runtime_config.WORKSPACE_PATH}: {invalid_volumes}."
                )
            else:
                assigned["volumes"] = requested_volumes
                assigned["workspace"] = str(runtime_config.WORKSPACE_PATH)

        if requested_services:
            missing_services = [
                service for service in requested_services if service not in runtime_config.AVAILABLE_SERVICES
            ]
            if missing_services:
                errors.append(f"Requested services are not available: {sorted(missing_services)}.")
            else:
                assigned["services"] = {
                    service: runtime_config.AVAILABLE_SERVICES[service]
                    for service in requested_services
                }

        if errors:
            return {"error": "Resource request failed.", "details": errors}
        return {"status": "ok", "assigned": assigned}

    def _import_plugin(self, name: str, path: Path) -> None:
        """Import *path* as ``plugins.<name>`` and store in registry.

        Must be called with ``self._lock`` held.
        """
        module_name = f"plugins.{name}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, "run"):
            del sys.modules[module_name]
            raise ImportError(f"Plugin {name!r} must define a run() function.")

        # Call init() only if it accepts no required arguments (i.e. not http).
        init_fn = getattr(module, "init", None)
        if init_fn is not None:
            try:
                init_fn()
            except TypeError:
                pass  # init requires arguments – caller will handle it
            except Exception:
                logger.exception("Plugin %r init() raised an exception", name)

        self.plugins[name] = module
        logger.info("Plugin %r loaded from %s", name, path)

    def _get_plugin_env(self, name: str) -> dict[str, str]:
        env = os.environ.copy()
        assignments = self._resource_assignments.get(name, {})
        for port in assignments.get("ports", []):
            env[f"PORT_{port}"] = str(port)
        if "workspace" in assignments:
            env["WORKSPACE_PATH"] = str(assignments["workspace"])
        for svc_name, uri in assignments.get("services", {}).items():
            env[f"{svc_name.upper()}_URI"] = uri
        return env

    def _call_inprocess(
        self,
        name: str,
        plugin: ModuleType,
        input_data: dict[str, Any],
        timeout: int,
        env: dict[str, str],
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None]:
        """Run plugin.run() in a background thread with a timeout.

        Returns (result, exec_ms, success, error_type, traceback_str).
        """
        run_fn = getattr(plugin, "run", None)
        if run_fn is None:
            return {"error": f"Plugin {name!r} has no run() function."}, 0.0, False, "AttributeError", None

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _run() -> None:
            old_env = os.environ.copy()
            try:
                os.environ.clear()
                os.environ.update(env)
                result_queue.put(("ok", run_fn(input_data)))
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("err", (exc, tb.format_exc())))
            finally:
                os.environ.clear()
                os.environ.update(old_env)

        start = time.monotonic()
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            status, value = result_queue.get(timeout=timeout)
        except queue.Empty:
            exec_ms = (time.monotonic() - start) * 1000
            return {"error": f"Plugin {name!r} timed out after {timeout}s."}, exec_ms, False, "TimeoutError", None

        exec_ms = (time.monotonic() - start) * 1000

        if status == "err":
            exc, tb_str = value
            logger.error("Plugin %r raised: %s", name, tb_str)
            return {"error": str(exc), "traceback": tb_str}, exec_ms, False, type(exc).__name__, tb_str

        result = value if isinstance(value, dict) else {"result": value}
        return result, exec_ms, True, None, None

    def _call_subprocess(
        self,
        name: str,
        input_data: dict[str, Any],
        timeout: int,
        env: dict[str, str],
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None]:
        """Run plugin in an isolated subprocess via core.sandbox_wrapper.

        Returns (result, exec_ms, success, error_type, traceback_str).
        """
        if SANDBOX_BACKEND == "docker":
            docker_result = self._call_docker_subprocess(name, input_data, timeout, env)
            if docker_result is not None:
                return docker_result

        plugin_path = self.plugins_dir / f"{name}.py"
        payload = json.dumps(
            {
                "plugin_path": str(plugin_path.resolve()),
                "input_data": input_data,
                "env": env,
            }
        )

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "core.sandbox_wrapper"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT if timeout == 30 else timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            exec_ms = (time.monotonic() - start) * 1000
            return {"error": f"Plugin {name!r} timed out after {timeout}s."}, exec_ms, False, "TimeoutError", None

        exec_ms = (time.monotonic() - start) * 1000

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

    def _call_docker_subprocess(
        self,
        name: str,
        input_data: dict[str, Any],
        timeout: int,
        env: dict[str, str],
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None] | None:
        if self._docker_runner is None:
            self._docker_runner = DockerSandboxRunner(self.plugins_dir)

        result = self._docker_runner.run_plugin(
            name,
            input_data,
            SANDBOX_TIMEOUT if timeout == 30 else timeout,
            env,
        )
        if result[2] is True:
            return result
        if SANDBOX_DOCKER_REQUIRED:
            return result
        return None

    def _read_plugin_code(self, name: str) -> str:
        path = self.plugins_dir / f"{name}.py"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _current_version_str(self, name: str) -> str:
        meta = _read_version_meta(self.plugins_dir, name)
        v = meta.get("current_version", 0)
        return f"v{v}"
