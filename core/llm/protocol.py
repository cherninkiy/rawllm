"""Structural protocol (interface) that every LLM adapter must satisfy.

Internal message format
-----------------------
TAORLoop and all adapters speak **OpenAI-compatible** messages natively.

Tool-call assistant turn::

    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "<str>",
                "type": "function",
                "function": {"name": "<str>", "arguments": "<json-str>"},
            },
            ...
        ],
    }

Tool-result turn (one message per result)::

    {"role": "tool", "tool_call_id": "<str>", "content": "<json-str>"}

Tool definitions use the OpenAI schema (``parameters`` key).  Adapters for
non-OpenAI providers (e.g. Anthropic) convert internally.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Structural interface for LLM chat clients with tool-use support.

    Any object with a ``model`` attribute and a matching ``chat()`` signature
    satisfies this protocol – no inheritance required.
    """

    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = "",
    ) -> dict[str, Any]:
        """Send *messages* to the LLM and return a normalised response.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Tool definitions in OpenAI function-calling schema.
            system: Optional system prompt string.

        Returns:
            One of::

                {"type": "text",       "content": str}
                {"type": "tool_calls", "tool_calls": [
                    {"id": str, "name": str, "input": dict}, ...
                ]}
        """
        ...
