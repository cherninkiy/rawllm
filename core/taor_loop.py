"""TAOR loop: Think → Act → Observe → Repeat.

Internal message format
-----------------------
All messages in the history are **OpenAI-compatible**.  The LLM adapter
(Anthropic or OpenAI) is responsible for translating to/from its native wire
format; TAORLoop itself is provider-agnostic.

Tool call assistant turn appended to history::

    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": <str>,
                "type": "function",
                "function": {"name": <str>, "arguments": <json-str>},
            },
            …
        ],
    }

Tool result turns (one per tool call)::

    {"role": "tool", "tool_call_id": <str>, "content": <json-str>}
"""

import asyncio
import json
import logging
from typing import Any

from core.llm.protocol import LLMClientProtocol
from core.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions – OpenAI function-calling schema
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_plugin",
            "description": (
                "Write a Python plugin to disk and hot-reload it. "
                "The code must define `run(input_data: dict) -> dict`. "
                "Optionally it may define `init(callback)` and `shutdown()`. "
                "If the plugin imports modules outside the default allow-list, list them in "
                "proposed_requirements; the orchestrator will gate execution until approved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Plugin name (no .py extension)."},
                    "code": {"type": "string", "description": "Full Python source code of the plugin."},
                    "proposed_requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of third-party modules the plugin needs.",
                    },
                },
                "required": ["name", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_plugin",
            "description": "Execute a loaded plugin by name and return its result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Plugin name."},
                    "input_data": {
                        "type": "object",
                        "description": "Arbitrary JSON-serialisable dict passed to the plugin's run().",
                    },
                },
                "required": ["name", "input_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unload_plugin",
            "description": "Gracefully shut down and remove a plugin. Cannot unload 'http'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Plugin name to unload."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_plugins_parallel",
            "description": (
                "Execute multiple plugins concurrently in a single Act and return all results. "
                "Use this instead of sequential run_plugin calls when the plugins are independent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "description": "List of [plugin_name, input_data] pairs to run in parallel.",
                        "items": {
                            "type": "array",
                            "prefixItems": [
                                {"type": "string", "description": "Plugin name."},
                                {"type": "object", "description": "Input dict for the plugin."},
                            ],
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "required": ["calls"],
            },
        },
    },
]


class TAORLoop:
    """Runs the Think–Act–Observe–Repeat cycle between the LLM and tool executor."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        tool_executor: ToolExecutor,
        system_prompt: str,
        max_iterations: int = 10,
    ) -> None:
        """Initialise the loop.

        Args:
            llm_client: Any object satisfying :class:`~core.llm.protocol.LLMClientProtocol`.
            tool_executor: Executes tool calls returned by the LLM.
            system_prompt: System prompt injected on every request.
            max_iterations: Hard limit on LLM↔tool turns per request.
        """
        self._llm = llm_client
        self._executor = tool_executor
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

    def process_request(
        self, user_prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Synchronous wrapper around :meth:`process_request_async` for backward compatibility."""
        return asyncio.run(self.process_request_async(user_prompt, context))

    async def process_request_async(
        self, user_prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Process a user request through the TAOR cycle and return the final answer.

        Args:
            user_prompt: The raw user message.
            context: Optional additional context merged into the first user message.

        Returns:
            The LLM's final text response.
        """
        user_content = user_prompt
        if context:
            user_content = f"{user_prompt}\n\nContext:\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]

        for iteration in range(self._max_iterations):
            logger.debug("TAOR iteration %d/%d", iteration + 1, self._max_iterations)

            response = await self._llm.chat_async(
                messages=messages,
                tools=TOOLS,
                system=self._system_prompt,
            )

            if response["type"] == "text":
                return response["content"]

            # tool_calls branch
            tool_calls = response["tool_calls"]

            # Append the assistant's tool-call turn in OpenAI format.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"], ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute all tool calls in parallel and append individual tool-result messages.
            results = await asyncio.gather(
                *[self._dispatch_async(call["name"], call["input"]) for call in tool_calls]
            )
            for call, result in zip(tool_calls, results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return (
            f"Reached maximum iterations ({self._max_iterations}) without a final answer. "
            "Please try again with a simpler request."
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Route a tool call to the executor and return the result.

        Note: kept in sync with :meth:`_dispatch_async`.  Direct callers
        should prefer the async path; this synchronous variant is retained
        for test coverage and potential future use.
        """
        if tool_name == "add_plugin":
            return self._executor.add_plugin(
                name=tool_input["name"],
                code=tool_input["code"],
            )
        if tool_name == "run_plugin":
            return self._executor.run_plugin(
                name=tool_input["name"],
                input_data=tool_input.get("input_data", {}),
            )
        if tool_name == "unload_plugin":
            return self._executor.unload_plugin(name=tool_input["name"])
        if tool_name == "run_plugins_parallel":
            raw_calls = tool_input.get("calls", [])
            calls = [(c[0], c[1]) for c in raw_calls]
            results = [
                self._executor.run_plugin(name=name, input_data=inp)
                for name, inp in calls
            ]
            return {"results": results}

        return {"error": f"Unknown tool: {tool_name!r}"}

    async def _dispatch_async(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Async routing of a tool call to the executor."""
        if tool_name == "add_plugin":
            return await self._executor.add_plugin_async(
                name=tool_input["name"],
                code=tool_input["code"],
            )
        if tool_name == "run_plugin":
            return await self._executor.run_plugin_async(
                name=tool_input["name"],
                input_data=tool_input.get("input_data", {}),
            )
        if tool_name == "unload_plugin":
            return await self._executor.unload_plugin_async(name=tool_input["name"])
        if tool_name == "run_plugins_parallel":
            raw_calls = tool_input.get("calls", [])
            calls = [(c[0], c[1]) for c in raw_calls]
            results = await self._executor.run_plugins_parallel(calls)
            return {"results": results}

        return {"error": f"Unknown tool: {tool_name!r}"}
