"""Entry point for RawLLM."""

from __future__ import annotations

import argparse
import os
import signal
import threading

import core.config as config
from core.llm import get_llm_client
from core.plugin_manager import PluginManager
from core.prompt_builder import build_startup_prompt
from core.taor_loop import TAORLoop
from core.tool_executor import ToolExecutor
from core.utils import configure_logging, ensure_dir, load_env, read_system_prompt

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous AI agent operating inside RawLLM. "
    "Use add_plugin, run_plugin, and unload_plugin to create the interface needed for this session."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RawLLM orchestrator.")
    parser.add_argument(
        "--ports",
        default=os.environ.get("RAWLLM_PORTS", ""),
        help="Optional available ports as a comma-separated list or ranges (for example 8000-8008).",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("RAWLLM_WORKSPACE", "./workspace"),
        help="Workspace directory made available to plugins.",
    )
    parser.add_argument(
        "--services",
        default=os.environ.get("RAWLLM_SERVICES", ""),
        help="Comma-separated services in name:uri format.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional startup task for the model.",
    )
    return parser.parse_args(argv)


def _load_or_default_system_prompt() -> str:
    try:
        return read_system_prompt(config.SYSTEM_PROMPT_PATH)
    except FileNotFoundError:
        return DEFAULT_SYSTEM_PROMPT


def main(argv: list[str] | None = None) -> None:
    """Bootstrap and run the orchestrator."""
    configure_logging()
    load_env()
    args = _parse_args(argv)

    ports = config._parse_ports(args.ports)
    workspace_path = ensure_dir(config._parse_workspace(args.workspace))
    services = config._parse_services(args.services)
    config.configure_runtime_resources(
        ports=ports,
        workspace_path=workspace_path,
        services=services,
    )

    llm_client = get_llm_client()
    print(f"Provider: {llm_client.__class__.__name__} | Model: {llm_client.model}")

    system_prompt = _load_or_default_system_prompt()
    startup_prompt = build_startup_prompt(
        {
            "ports": list(config.AVAILABLE_PORTS),
            "workspace": config.WORKSPACE_PATH,
            "services": dict(config.AVAILABLE_SERVICES),
        },
        args.prompt,
    )

    plugins_dir = ensure_dir(config.PLUGINS_DIR)
    plugin_manager = PluginManager(plugins_dir)
    tool_executor = ToolExecutor(plugin_manager)
    taor_loop = TAORLoop(llm_client, tool_executor, system_prompt, startup_prompt)

    initial_response = taor_loop.process_request()
    if initial_response:
        print(initial_response)

    if not plugin_manager.get_all_plugins():
        print("No plugins remain loaded after startup; exiting.")
        return

    stop_event = threading.Event()

    def _handle_signal(sig: int, _frame: object) -> None:
        print(f"\nReceived signal {sig}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("RawLLM is running. Press Ctrl+C to stop.")
    stop_event.wait()

    for name, plugin in plugin_manager.get_all_plugins().items():
        shutdown_fn = getattr(plugin, "shutdown", None)
        if shutdown_fn is not None:
            try:
                shutdown_fn()
            except Exception as exc:
                print(f"Warning: plugin {name!r} shutdown() raised: {exc}")


if __name__ == "__main__":
    main()
