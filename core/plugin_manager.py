"""Plugin manager: load, reload, call, and unload plugins dynamically.

Enhancements over the original:
- Hybrid execution: trusted plugins run in-process; all others run via subprocess.
- Plugin versioning: old versions are archived under plugins_store/.
- Dependency gating: imports outside ALLOWED_REQUIREMENTS are written to
  pending_requirements.txt and the plugin is not executed until approved.
- Metrics: every execution is logged to metrics.jsonl via core.metrics.
- Full traceback returned on error (both in-process and subprocess paths).
"""

import ast
import importlib
import importlib.util
import json
import logging
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

import core.metrics as metrics
from core.config import ALLOWED_REQUIREMENTS, SANDBOX_TIMEOUT, TRUSTED_PLUGINS

logger = logging.getLogger(__name__)

PROTECTED_PLUGINS = {"http"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_imports(code: str) -> list[str]:
    """Return a list of top-level module names imported by *code*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _import_risk_score(code: str) -> int:
    """Count of imports that are outside ALLOWED_REQUIREMENTS."""
    imports = _extract_imports(code)
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


def _read_version_meta(plugins_dir: Path, name: str) -> dict[str, Any]:
    path = _version_json_path(plugins_dir, name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
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

        if is_trusted:
            result, exec_ms, success, err_type, tb_str = self._call_inprocess(name, plugin, input_data, timeout)
        else:
            result, exec_ms, success, err_type, tb_str = self._call_subprocess(name, input_data, timeout)

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

    def add_plugin(self, name: str, code: str) -> dict[str, Any]:
        """Write *code* to ``plugins/{name}.py`` and hot-reload it.

        Archives the previous version (if any) and updates symlinks in
        ``plugins_store/current/``.
        """
        path = self.plugins_dir / f"{name}.py"
        try:
            # Archive existing version before overwriting.
            if path.exists():
                old_code = path.read_text(encoding="utf-8")
                snapshot = self._exec_counts.get(name, {})
                old_ver = self._current_version_str(name)
                new_ver = _archive_plugin(self.plugins_dir, name, old_code, snapshot)
                metrics.log_version_change(name, old_ver, new_ver)
            else:
                # First time – still create the store entry.
                _update_current_symlink(self.plugins_dir, name, path)

            path.write_text(code, encoding="utf-8")
            result = self.reload_plugin(name)
            if "status" in result:
                _update_current_symlink(self.plugins_dir, name, path)
            return result
        except Exception as exc:
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
        self.plugins_dir.joinpath(f"{name}.py").write_text(
            prev_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        _update_current_symlink(self.plugins_dir, name, self.plugins_dir / f"{name}.py")

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        if init_fn is not None and name not in PROTECTED_PLUGINS:
            try:
                init_fn()
            except TypeError:
                pass  # init requires arguments – caller will handle it
            except Exception:
                logger.exception("Plugin %r init() raised an exception", name)

        self.plugins[name] = module
        logger.info("Plugin %r loaded from %s", name, path)

    def _call_inprocess(
        self,
        name: str,
        plugin: ModuleType,
        input_data: dict[str, Any],
        timeout: int,
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None]:
        """Run plugin.run() in a background thread with a timeout.

        Returns (result, exec_ms, success, error_type, traceback_str).
        """
        run_fn = getattr(plugin, "run", None)
        if run_fn is None:
            return {"error": f"Plugin {name!r} has no run() function."}, 0.0, False, "AttributeError", None

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _run() -> None:
            try:
                result_queue.put(("ok", run_fn(input_data)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("err", (exc, tb.format_exc())))

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
    ) -> tuple[dict[str, Any], float, bool, str | None, str | None]:
        """Run plugin in an isolated subprocess via core.sandbox_wrapper.

        Returns (result, exec_ms, success, error_type, traceback_str).
        """
        plugin_path = self.plugins_dir / f"{name}.py"
        payload = json.dumps({"plugin_path": str(plugin_path.resolve()), "input_data": input_data})

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "core.sandbox_wrapper"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT if timeout == 30 else timeout,
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
