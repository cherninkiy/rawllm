"""LLM client wrapping the Anthropic API with tool-use support.

Implements :class:`core.llm_protocol.LLMClientProtocol`.

Message translation
-------------------
TAORLoop uses **OpenAI-compatible** messages internally.  This adapter
converts them to Anthropic format before every API call and returns the
normalised response dict defined by the protocol.

OpenAI → Anthropic conversions performed here:

* ``role: assistant`` with ``tool_calls`` list →
  ``role: assistant`` with ``content: [tool_use, …]`` blocks.
* Consecutive ``role: tool`` messages →
  single ``role: user`` with ``content: [tool_result, …]`` blocks.
* Tool definitions: ``parameters`` key → ``input_schema`` key.
"""

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """Anthropic adapter implementing :class:`~core.llm_protocol.LLMClientProtocol`."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        """Initialise the client.

        Args:
            api_key: Anthropic API key.
            model: Model identifier to use for all requests.
        """
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = "",
    ) -> dict[str, Any]:
        """Send *messages* to Anthropic and return a normalised response.

        Args:
            messages: Conversation history in **OpenAI** message format.
            tools: Tool definitions in **OpenAI** function-calling schema.
            system: Optional system prompt string.

        Returns:
            ``{"type": "text", "content": str}`` or
            ``{"type": "tool_calls", "tool_calls": [{"id", "name", "input"}, …]}``
        """
        anthropic_tools = self._convert_tools(tools)
        anthropic_messages = self._convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise

        if response.stop_reason == "tool_use":
            tool_calls = [
                {"id": block.id, "name": block.name, "input": block.input}
                for block in response.content
                if block.type == "tool_use"
            ]
            return {"type": "tool_calls", "tool_calls": tool_calls}

        # "end_turn" or any other stop reason → extract text
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        return {"type": "text", "content": "\n".join(text_parts).strip()}

    # ------------------------------------------------------------------
    # Private translation helpers
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format.

        Input (OpenAI)::

            {"type": "function", "function": {"name": …, "description": …, "parameters": …}}

        Output (Anthropic)::

            {"name": …, "description": …, "input_schema": …}
        """
        result = []
        for tool in tools:
            fn = tool.get("function", {})
            result.append(
                {
                    "name": fn.get("name", tool.get("name", "")),
                    "description": fn.get("description", tool.get("description", "")),
                    "input_schema": fn.get("parameters", tool.get("input_schema", {})),
                }
            )
        return result

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate OpenAI-format message history to Anthropic API format.

        Handled cases:

        1. ``role: assistant`` with ``tool_calls`` →
           ``role: assistant`` with ``content: [tool_use …]``.
        2. One or more consecutive ``role: tool`` messages →
           single ``role: user`` with ``content: [tool_result …]``.
        3. ``role: system`` → skipped (passed as *system* parameter instead).
        4. All other messages pass through unchanged.
        """
        result: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                # Handled via the dedicated ``system`` parameter.
                i += 1
                continue

            if role == "assistant" and msg.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        args = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args,
                        }
                    )
                result.append({"role": "assistant", "content": content})
                i += 1
                continue

            if role == "tool":
                # Group all consecutive tool-result messages into one user turn.
                tool_results: list[dict[str, Any]] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": messages[i]["tool_call_id"],
                            "content": messages[i]["content"],
                        }
                    )
                    i += 1
                result.append({"role": "user", "content": tool_results})
                continue

            # Regular user / plain-text assistant message.
            result.append(msg)
            i += 1

        return result
