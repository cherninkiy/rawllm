"""Tool executor: bridges the LLM tool calls and the PluginManager."""

import logging
from typing import Any

import core.metrics as metrics
from core.config import ALLOWED_REQUIREMENTS, PENDING_REQUIREMENTS_FILE
from core.plugin_manager import PluginManager
from core.utils import extract_imports

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Exposes the three core tools (add_plugin, run_plugin, unload_plugin) to the LLM."""

    def __init__(self, plugin_manager: PluginManager) -> None:
        """Initialise with a PluginManager instance."""
        self._pm = plugin_manager

    def add_plugin(self, name: str, code: str) -> dict[str, Any]:
        """Write *code* as plugin *name* and hot-reload it.

        Checks imports in *code* against ``ALLOWED_REQUIREMENTS``.  If any
        module is outside the allow-list the request is appended to
        ``pending_requirements.txt`` and an informational message is returned
        to the LLM asking it to wait for human approval.

        Returns a dict with ``status`` on success or ``error`` on failure.
        """
        logger.info("Tool: add_plugin(%r)", name)

        imports = extract_imports(code)
        blocked = [imp for imp in imports if imp not in ALLOWED_REQUIREMENTS]
        if blocked:
            pending = sorted(set(blocked))
            _append_pending_requirements(name, pending)
            metrics.log_dependency_request(name, imports, pending)
            logger.warning(
                "Plugin %r requests unapproved modules %s – held for human review.",
                name,
                pending,
            )
            return {
                "status": "pending_approval",
                "plugin": name,
                "message": (
                    f"Plugin {name!r} requires modules that are not yet approved: "
                    f"{pending}. The request has been logged to pending_requirements.txt. "
                    "Please wait for a human to review and approve before retrying."
                ),
            }

        return self._pm.add_plugin(name, code)

    def run_plugin(self, name: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute plugin *name* with *input_data*.

        Returns the plugin's result dict or an error dict.
        """
        logger.info("Tool: run_plugin(%r, %s)", name, list(input_data.keys()))
        return self._pm.call_plugin(name, input_data)

    def unload_plugin(self, name: str) -> dict[str, Any]:
        """Shut down and remove plugin *name* from the registry.

        Returns a dict with ``status`` on success or ``error`` on failure.
        """
        logger.info("Tool: unload_plugin(%r)", name)
        return self._pm.unload_plugin(name)


def _append_pending_requirements(plugin_name: str, modules: list[str]) -> None:
    """Append a human-readable entry to pending_requirements.txt."""
    line = f"plugin={plugin_name} modules={','.join(modules)}\n"
    try:
        with PENDING_REQUIREMENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        logger.exception("Failed to write pending_requirements.txt")
