"""LLM client wrapping any OpenAI-compatible API (Groq, Gemini, Ollama, OpenRouter, …).

Implements :class:`core.llm.protocol.LLMClientProtocol`.

Because TAORLoop uses **OpenAI-compatible** messages natively, this adapter
requires no message translation – it passes the history straight through to
the OpenAI ``chat.completions.create`` endpoint.

Tool definitions are already in OpenAI ``function`` schema format (the
``parameters`` key), so those are passed through unchanged too.
"""

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Thin wrapper around any OpenAI-compatible chat-completion endpoint.

    Implements :class:`~core.llm.protocol.LLMClientProtocol`.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        """Initialise the client.

        Args:
            api_key: Provider API key (pass ``""`` for Ollama local).
            base_url: Provider base URL, e.g. ``https://api.groq.com/openai/v1``.
            model: Model identifier used for all requests.
        """
        self.model = model
        self._client = OpenAI(api_key=api_key or "none", base_url=base_url)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = "",
    ) -> dict[str, Any]:
        """Send *messages* to the OpenAI-compatible endpoint.

        Args:
            messages: Conversation history in OpenAI message format (native).
            tools: Tool definitions in OpenAI function-calling schema (native).
            system: Optional system prompt string.

        Returns:
            ``{"type": "text", "content": str}`` or
            ``{"type": "tool_calls", "tool_calls": [{"id", "name", "input"}, …]}``
        """
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": arguments,
                    }
                )
            return {"type": "tool_calls", "tool_calls": tool_calls}

        content = (choice.message.content or "").strip()
        return {"type": "text", "content": content}
