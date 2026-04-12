"""Entry point for the Dumb Orchestrator.

Selects the LLM backend via the ``LLM_PROVIDER`` environment variable
(default: ``anthropic``).  All security features (sandboxing, versioning,
dependency gating, metrics) are included automatically because the same
:class:`~core.plugin_manager.PluginManager` /
:class:`~core.tool_executor.ToolExecutor` / :class:`~core.taor_loop.TAORLoop`
are used regardless of provider.

Supported providers
-------------------
``anthropic``
    Uses :class:`core.llm.clients.anthropic.AnthropicClient`.
    Requires ``ANTHROPIC_API_KEY``.
``groq`` / ``gemini`` / ``openrouter`` / ``ollama``
    Use :class:`core.llm.clients.openai_compat.OpenAICompatibleClient`.
    Require the respective API key env var
    (see :data:`core.llm.registry.LLM_PROVIDERS`).

Environment overrides
---------------------
``LLM_PROVIDER``
    Provider to use (default: ``anthropic``).
``LLM_MODEL``
    Override the default model for any provider.
``LLM_BASE_URL``
    Override the base URL for OpenAI-compatible providers.
"""

import signal
import threading

from core.llm import get_llm_client
from core.plugin_manager import PluginManager
from core.taor_loop import TAORLoop
from core.tool_executor import ToolExecutor
from core.utils import configure_logging, ensure_dir, load_env, read_system_prompt

PLUGINS_DIR = ensure_dir("plugins")
SYSTEM_PROMPT_PATH = "system_prompt.txt"


def main() -> None:
    """Bootstrap and run the orchestrator."""
    configure_logging()
    load_env()

    llm_client = get_llm_client()
    print(f"Provider: {llm_client.__class__.__name__} | Model: {llm_client.model}")

    system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

    plugin_manager = PluginManager(PLUGINS_DIR)
    plugin_manager.load_plugins()

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
