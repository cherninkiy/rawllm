"""Plugin manager: load, reload, call, and unload plugins dynamically."""

import importlib
import importlib.util
import logging
import queue
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

PROTECTED_PLUGINS = {"http"}


class PluginManager:
    """Manages the lifecycle of plugins: discovery, loading, execution, and unloading."""

    def __init__(self, plugins_dir: Path) -> None:
        """Initialise the manager pointing at *plugins_dir*."""
        self.plugins_dir = plugins_dir
        self.plugins: dict[str, ModuleType] = {}
        self._lock = threading.RLock()

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
        """Execute ``plugin.run(input_data)`` in a thread with a timeout.

        Returns the result dict, or an error dict on failure / timeout.
        """
        with self._lock:
            plugin = self.get_plugin(name)
            if plugin is None:
                return {"error": f"Plugin {name!r} is not loaded."}
            run_fn = getattr(plugin, "run", None)

        if run_fn is None:
            return {"error": f"Plugin {name!r} has no run() function."}

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _run() -> None:
            try:
                result_queue.put(("ok", run_fn(input_data)))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("err", exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            status, value = result_queue.get(timeout=timeout)
        except queue.Empty:
            return {"error": f"Plugin {name!r} timed out after {timeout}s."}

        if status == "err":
            logger.exception("Plugin %r raised an exception", name)
            return {"error": str(value)}
        return value if isinstance(value, dict) else {"result": value}

    def add_plugin(self, name: str, code: str) -> dict[str, Any]:
        """Write *code* to ``plugins/{name}.py`` and hot-reload it."""
        path = self.plugins_dir / f"{name}.py"
        try:
            path.write_text(code, encoding="utf-8")
            return self.reload_plugin(name)
        except Exception as exc:
            logger.exception("Failed to add plugin %r", name)
            return {"error": str(exc)}

    def reload_plugin(self, name: str) -> dict[str, Any]:
        """Shut down the old instance (if any), then re-import and update the registry.

        Calling ``shutdown()`` on the previous instance prevents resource leaks
        (e.g. a stale HTTP server holding onto port 8080).
        """
        path = self.plugins_dir / f"{name}.py"
        if not path.exists():
            return {"error": f"Plugin file {path} does not exist."}

        with self._lock:
            # Shut down the previous version before replacing it.
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
        """Shut down and remove *name* from the registry (file stays on disk).

        Protected plugins (e.g. ``http``) cannot be unloaded.
        """
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
