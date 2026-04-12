"""Backward-compatible re-export – use ``core.llm.clients.anthropic`` for new code."""

from core.llm.clients.anthropic import AnthropicClient  # noqa: F401

# Legacy alias kept for backward compatibility.
LLMClient = AnthropicClient
