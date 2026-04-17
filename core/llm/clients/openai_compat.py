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

import httpx

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
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

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

        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=kwargs,
            timeout=120,
            trust_env=False,
        )
        if response.status_code >= 400:
            body = response.text[:500]
            raise RuntimeError(f"OpenAI-compatible API error {response.status_code}: {body}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            body = response.text[:500]
            raise RuntimeError(
                "OpenAI-compatible API returned invalid JSON "
                f"for status {response.status_code}: {body}"
            ) from exc
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible API returned no choices")

        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        tool_calls_raw = message.get("tool_calls") or []

        if finish_reason == "tool_calls" and tool_calls_raw:
            tool_calls = []
            for tc in tool_calls_raw:
                fn = tc.get("function") or {}
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": arguments,
                    }
                )
            return {"type": "tool_calls", "tool_calls": tool_calls}

        content = (message.get("content") or "").strip()
        return {"type": "text", "content": content}
