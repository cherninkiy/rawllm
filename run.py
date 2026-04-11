"""Entry point for the Dumb Orchestrator – Smart Model."""

import signal
import threading

from core.plugin_manager import PluginManager
from core.llm_client import LLMClient
from core.tool_executor import ToolExecutor
from core.taor_loop import TAORLoop
from core.utils import load_env, read_system_prompt, ensure_dir, get_api_key, configure_logging

PLUGINS_DIR = ensure_dir("plugins")
SYSTEM_PROMPT_PATH = "system_prompt.txt"


def main() -> None:
    """Bootstrap and run the orchestrator."""
    configure_logging()
    load_env()

    system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

    plugin_manager = PluginManager(PLUGINS_DIR)
    plugin_manager.load_plugins()

    api_key = get_api_key("ANTHROPIC_API_KEY")
    llm_client = LLMClient(api_key=api_key)
    tool_executor = ToolExecutor(plugin_manager)
    taor_loop = TAORLoop(llm_client, tool_executor, system_prompt)

    # Wire up the HTTP plugin callback now that the loop is ready.
    http_plugin = plugin_manager.get_plugin("http")
    if http_plugin is not None:
        init_fn = getattr(http_plugin, "init", None)
        if init_fn is not None:
            init_fn(callback=taor_loop.process_request)

    # Keep the main thread alive until SIGINT / SIGTERM.
    stop_event = threading.Event()

    def _handle_signal(sig: int, _frame: object) -> None:
        print(f"\nReceived signal {sig}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Dumb Orchestrator is running. Press Ctrl+C to stop.")
    stop_event.wait()

    # Gracefully shut down all loaded plugins.
    for name, plugin in plugin_manager.get_all_plugins().items():
        shutdown_fn = getattr(plugin, "shutdown", None)
        if shutdown_fn is not None:
            try:
                shutdown_fn()
            except Exception as exc:
                print(f"Warning: plugin {name!r} shutdown() raised: {exc}")


if __name__ == "__main__":
    main()
