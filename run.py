"""Entry point for the Dumb Orchestrator.

Selects the LLM backend via the ``LLM_PROVIDER`` environment variable
(default: ``anthropic``).  All Part-A security features (sandboxing,
versioning, dependency gating, metrics) are included automatically because
the same :class:`~core.plugin_manager.PluginManager` /
:class:`~core.tool_executor.ToolExecutor` / :class:`~core.taor_loop.TAORLoop`
are used regardless of provider.

Supported providers
-------------------
``anthropic``
    Uses :class:`core.llm_client.LLMClient`.  Requires ``ANTHROPIC_API_KEY``.
``groq`` / ``gemini`` / ``openrouter`` / ``ollama``
    Use :class:`core.llm_client_openai.LLMClientOpenAI`.  Require the
    respective API key env var (see :data:`core.config.LLM_PROVIDERS`).

Environment overrides
---------------------
``LLM_MODEL``
    Override the default model for any provider.
``LLM_BASE_URL``
    Override the base URL for OpenAI-compatible providers.
"""

import os
import signal
import threading
from typing import Any

from core.config import LLM_PROVIDERS
from core.llm_client import LLMClient
from core.llm_client_openai import LLMClientOpenAI
from core.llm_protocol import LLMClientProtocol
from core.plugin_manager import PluginManager
from core.tool_executor import ToolExecutor
from core.taor_loop import TAORLoop
from core.utils import configure_logging, ensure_dir, load_env, read_system_prompt

PLUGINS_DIR = ensure_dir("plugins")
SYSTEM_PROMPT_PATH = "system_prompt.txt"


def _build_llm_client(provider: str) -> LLMClientProtocol:
    """Construct and return the appropriate LLM client for *provider*.

    Args:
        provider: One of the keys in :data:`core.config.LLM_PROVIDERS`.

    Raises:
        RuntimeError: If *provider* is unknown or its API key is missing.
    """
    cfg: dict[str, Any] | None = LLM_PROVIDERS.get(provider)
    if cfg is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {provider!r}. "
            f"Choose from: {sorted(LLM_PROVIDERS)}"
        )

    api_key_env: str | None = cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set. "
                "Add it to your .env file or export it before running."
            )
    else:
        api_key = ""

    model: str = os.environ.get("LLM_MODEL", cfg["model"])

    if provider == "anthropic":
        return LLMClient(api_key=api_key, model=model)

    # All other providers use an OpenAI-compatible endpoint.
    base_url: str = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    return LLMClientOpenAI(api_key=api_key, base_url=base_url, model=model)


def main() -> None:
    """Bootstrap and run the orchestrator."""
    configure_logging()
    load_env()

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    llm_client = _build_llm_client(provider)
    print(f"Provider: {provider} | Model: {llm_client.model}")

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

