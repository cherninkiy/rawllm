"""Core package for RawLLM.

Keep package import lightweight so helper entry points such as
``python -m core.sandbox_wrapper`` do not trigger full LLM/runtime imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["AnthropicClient", "PluginManager", "ToolExecutor", "TAORLoop"]


def __getattr__(name: str) -> Any:
    if name == "AnthropicClient":
        return import_module("core.llm.clients.anthropic").AnthropicClient
    if name == "PluginManager":
        return import_module("core.plugin_manager").PluginManager
    if name == "ToolExecutor":
        return import_module("core.tool_executor").ToolExecutor
    if name == "TAORLoop":
        return import_module("core.taor_loop").TAORLoop
    raise AttributeError(f"module 'core' has no attribute {name!r}")
