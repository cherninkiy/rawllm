"""LLM client wrapping any OpenAI-compatible API (Groq, Gemini, Ollama, OpenRouter, …).

The ``chat()`` method is API-compatible with ``LLMClient`` (Anthropic), so
``TAORLoop`` works with both clients without modification.

Message translation:
- Anthropic-format assistant tool-call turns (list of ``{"id", "name", "input"}``)
  are converted to OpenAI ``tool_calls`` on the assistant message.
- Anthropic-format tool-result turns (user message with ``{"type": "tool_result",
  "tool_use_id", "content"}`` list) are split into individual ``role="tool"``
  messages.
- Tool definitions are converted from Anthropic ``input_schema`` to OpenAI
  ``parameters``.
"""

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClientOpenAI:
    """Thin wrapper around any OpenAI-compatible chat-completion endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        """Initialise the client.

        Args:
            api_key: Provider API key (pass ``""`` for Ollama local).
            base_url: Provider base URL, e.g. ``https://api.groq.com/openai/v1``.
            model: Model identifier used for all requests.
        """
        self.model = model
        self._client = OpenAI(api_key=api_key or "none", base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = "",
    ) -> dict[str, Any]:
        """Send *messages* to the LLM and return a normalised response.

        Returns:
            ``{"type": "text", "content": str}`` or
            ``{"type": "tool_calls", "tool_calls": [...], "raw": None}``
        """
        openai_tools = self._convert_tools(tools)

        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.extend(self._convert_message(msg))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": 0.2,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": arguments,
                    }
                )
            return {"type": "tool_calls", "tool_calls": tool_calls, "raw": None}

        content = (choice.message.content or "").strip()
        return {"type": "text", "content": content}

    # ------------------------------------------------------------------
    # Private translation helpers
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tool definitions to OpenAI format."""
        result = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return result

    def _convert_message(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate a single message from TAORLoop format to OpenAI API format.

        TAORLoop produces two Anthropic-specific message shapes that need
        translation:

        1. Assistant turn after tool calls::

               {"role": "assistant", "content": [{"id": ..., "name": ..., "input": ...}, ...]}

        2. Tool-result turn::

               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}, ...]}
        """
        role = msg["role"]
        content = msg["content"]

        # Assistant message carrying tool calls (list of our normalised dicts).
        if (
            role == "assistant"
            and isinstance(content, list)
            and content
            and isinstance(content[0], dict)
            and "name" in content[0]
            and "input" in content[0]
        ):
            tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"], ensure_ascii=False),
                    },
                }
                for tc in content
            ]
            return [{"role": "assistant", "content": None, "tool_calls": tool_calls}]

        # User message containing tool results.
        if (
            role == "user"
            and isinstance(content, list)
            and content
            and isinstance(content[0], dict)
            and content[0].get("type") == "tool_result"
        ):
            return [
                {
                    "role": "tool",
                    "tool_call_id": item["tool_use_id"],
                    "content": item["content"],
                }
                for item in content
            ]

        return [msg]
