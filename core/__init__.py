"""Core package for the Dumb Orchestrator – Smart Model POC."""

from core.llm.clients.anthropic import AnthropicClient
from core.plugin_manager import PluginManager
from core.taor_loop import TAORLoop
from core.tool_executor import ToolExecutor

# Backward-compatible alias
LLMClient = AnthropicClient

__all__ = ["AnthropicClient", "LLMClient", "PluginManager", "ToolExecutor", "TAORLoop"]
