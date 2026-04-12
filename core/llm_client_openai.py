"""Backward-compatible re-export – use ``core.llm.clients.openai_compat`` for new code."""

from core.llm.clients.openai_compat import OpenAICompatibleClient  # noqa: F401

# Legacy alias kept for backward compatibility.
LLMClientOpenAI = OpenAICompatibleClient
