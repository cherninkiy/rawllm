"""Entry point for the Dumb Orchestrator with lightweight / OpenAI-compatible LLMs.

Reads LLM_PROVIDER from the environment to select the backend:
  groq        – Groq (llama3-70b-8192)
  gemini      – Google Gemini via OpenAI-compat shim (gemini-2.0-flash)
  openrouter  – OpenRouter (mistralai/mistral-7b-instruct:free)
  ollama      – Local Ollama (llama3.2:3b)

Override model with LLM_MODEL and base URL with LLM_BASE_URL.
All Part-A enhancements (sandboxing, versioning, metrics) are automatically
included because the same PluginManager / ToolExecutor / TAORLoop are used.
"""

import os
import signal
import threading

from core.llm_client_openai import LLMClientOpenAI
from core.plugin_manager import PluginManager
from core.taor_loop import TAORLoop
from core.tool_executor import ToolExecutor
from core.utils import configure_logging, ensure_dir, load_env, read_system_prompt

PLUGINS_DIR = ensure_dir("plugins")
SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Per-provider defaults -------------------------------------------------------
_PROVIDERS: dict[str, dict] = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama3-70b-8192",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "mistralai/mistral-7b-instruct:free",
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
    },
}


def main() -> None:
    """Bootstrap and run the orchestrator with a lightweight LLM."""
    configure_logging()
    load_env()

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {provider!r}. "
            f"Choose from: {sorted(_PROVIDERS)}"
        )

    api_key_env: str | None = cfg["api_key_env"]
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set. "
                "Add it to your .env file or export it before running."
            )
    else:
        api_key = ""

    base_url: str = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    model: str = os.environ.get("LLM_MODEL", cfg["model"])

    print(f"Provider: {provider} | Model: {model}")

    system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

    plugin_manager = PluginManager(PLUGINS_DIR)
    plugin_manager.load_plugins()

    llm_client = LLMClientOpenAI(api_key=api_key, base_url=base_url, model=model)
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

    print("Dumb Orchestrator (light) is running. Press Ctrl+C to stop.")
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
