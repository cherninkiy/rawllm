"""TAOR loop: Think → Act → Observe → Repeat."""

import json
import logging
from typing import Any

from core.llm_client import LLMClient
from core.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic tool definitions (schema passed to the LLM on every request)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "add_plugin",
        "description": (
            "Write a Python plugin to disk and hot-reload it. "
            "The code must define `run(input_data: dict) -> dict`. "
            "Optionally it may define `init(callback)` and `shutdown()`. "
            "If the plugin imports modules outside the default allow-list, list them in "
            "proposed_requirements; the orchestrator will gate execution until approved."
        ),
        "input_schema": {
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
    {
        "name": "run_plugin",
        "description": "Execute a loaded plugin by name and return its result.",
        "input_schema": {
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
    {
        "name": "unload_plugin",
        "description": "Gracefully shut down and remove a plugin. Cannot unload 'http'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Plugin name to unload."},
            },
            "required": ["name"],
        },
    },
]


class TAORLoop:
    """Runs the Think–Act–Observe–Repeat cycle between the LLM and tool executor."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        system_prompt: str,
        max_iterations: int = 10,
    ) -> None:
        """Initialise the loop.

        Args:
            llm_client: The LLM wrapper used to send messages.
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

            response = self._llm.chat(
                messages=messages,
                tools=TOOLS,
                system=self._system_prompt,
            )

            if response["type"] == "text":
                return response["content"]

            # tool_calls branch
            tool_calls = response["tool_calls"]
            raw_response = response.get("raw")

            # Append the assistant's tool-use turn
            if raw_response is not None:
                messages.append({"role": "assistant", "content": raw_response.content})
            else:
                messages.append({"role": "assistant", "content": tool_calls})

            # Execute each tool and collect results
            tool_results = []
            for call in tool_calls:
                result = self._dispatch(call["name"], call["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return (
            f"Reached maximum iterations ({self._max_iterations}) without a final answer. "
            "Please try again with a simpler request."
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Route a tool call to the executor and return the result."""
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

        return {"error": f"Unknown tool: {tool_name!r}"}
