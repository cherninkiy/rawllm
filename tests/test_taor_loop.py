"""Tests for TAORLoop."""

import json
from unittest.mock import MagicMock, call

import pytest

from core.taor_loop import TAORLoop


@pytest.fixture()
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_executor() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def loop(mock_llm: MagicMock, mock_executor: MagicMock) -> TAORLoop:
    return TAORLoop(mock_llm, mock_executor, system_prompt="You are helpful.", max_iterations=5)


# ---------------------------------------------------------------------------


def test_process_request_returns_text_on_first_response(
    loop: TAORLoop, mock_llm: MagicMock
) -> None:
    mock_llm.chat.return_value = {"type": "text", "content": "Hello, world!"}
    result = loop.process_request("Hi")
    assert result == "Hello, world!"
    assert mock_llm.chat.call_count == 1


def test_process_request_with_context_includes_context(
    loop: TAORLoop, mock_llm: MagicMock
) -> None:
    mock_llm.chat.return_value = {"type": "text", "content": "OK"}
    loop.process_request("Hello", context={"key": "value"})

    _, kwargs = mock_llm.chat.call_args
    messages = kwargs.get("messages") or mock_llm.chat.call_args[0][0]
    user_content = messages[0]["content"]
    assert "key" in user_content


def test_process_request_executes_tool_calls(
    loop: TAORLoop, mock_llm: MagicMock, mock_executor: MagicMock
) -> None:
    tool_call = {"id": "tc1", "name": "run_plugin", "input": {"name": "echo", "input_data": {"x": 1}}}

    mock_llm.chat.side_effect = [
        {"type": "tool_calls", "tool_calls": [tool_call]},
        {"type": "text", "content": "Done"},
    ]
    mock_executor.run_plugin.return_value = {"result": "ok"}

    result = loop.process_request("Run echo plugin")

    assert result == "Done"
    mock_executor.run_plugin.assert_called_once_with(name="echo", input_data={"x": 1})
    assert mock_llm.chat.call_count == 2


def test_process_request_tool_call_appends_openai_format_messages(
    loop: TAORLoop, mock_llm: MagicMock, mock_executor: MagicMock
) -> None:
    """After a tool call, the history must use OpenAI-native message format."""
    tool_call = {"id": "tc1", "name": "run_plugin", "input": {"name": "echo", "input_data": {}}}

    mock_llm.chat.side_effect = [
        {"type": "tool_calls", "tool_calls": [tool_call]},
        {"type": "text", "content": "Done"},
    ]
    mock_executor.run_plugin.return_value = {"result": "ok"}

    loop.process_request("Run echo plugin")

    # The second call receives the updated history.
    second_call_kwargs = mock_llm.chat.call_args_list[1]
    messages = second_call_kwargs[1].get("messages") or second_call_kwargs[0][0]

    # messages[0] = original user turn
    assert messages[0]["role"] == "user"

    # messages[1] = assistant turn with tool_calls list (OpenAI format)
    asst = messages[1]
    assert asst["role"] == "assistant"
    assert asst["content"] is None
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["id"] == "tc1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "run_plugin"

    # messages[2] = tool result (role: tool)
    tool_result = messages[2]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "tc1"
    assert json.loads(tool_result["content"]) == {"result": "ok"}


def test_process_request_dispatches_add_plugin(
    loop: TAORLoop, mock_llm: MagicMock, mock_executor: MagicMock
) -> None:
    tool_call = {
        "id": "tc2",
        "name": "add_plugin",
        "input": {"name": "myplugin", "code": "def run(d): return d"},
    }

    mock_llm.chat.side_effect = [
        {"type": "tool_calls", "tool_calls": [tool_call]},
        {"type": "text", "content": "Plugin added"},
    ]
    mock_executor.add_plugin.return_value = {"status": "ok"}

    result = loop.process_request("Add a plugin")
    assert result == "Plugin added"
    mock_executor.add_plugin.assert_called_once_with(name="myplugin", code="def run(d): return d")


def test_process_request_dispatches_unload_plugin(
    loop: TAORLoop, mock_llm: MagicMock, mock_executor: MagicMock
) -> None:
    tool_call = {"id": "tc3", "name": "unload_plugin", "input": {"name": "old"}}

    mock_llm.chat.side_effect = [
        {"type": "tool_calls", "tool_calls": [tool_call]},
        {"type": "text", "content": "Unloaded"},
    ]
    mock_executor.unload_plugin.return_value = {"status": "ok"}

    result = loop.process_request("Unload old")
    assert result == "Unloaded"
    mock_executor.unload_plugin.assert_called_once_with(name="old")


def test_process_request_unknown_tool_returns_error(
    loop: TAORLoop, mock_llm: MagicMock
) -> None:
    tool_call = {"id": "tc4", "name": "fly_to_moon", "input": {}}

    mock_llm.chat.side_effect = [
        {"type": "tool_calls", "tool_calls": [tool_call]},
        {"type": "text", "content": "Cannot do that"},
    ]

    result = loop.process_request("Do something impossible")
    # The second LLM call receives the error as a tool result; we just verify it completes.
    assert result == "Cannot do that"


def test_process_request_max_iterations(
    loop: TAORLoop, mock_llm: MagicMock, mock_executor: MagicMock
) -> None:
    tool_call = {"id": "tc5", "name": "run_plugin", "input": {"name": "loop", "input_data": {}}}
    mock_llm.chat.return_value = {"type": "tool_calls", "tool_calls": [tool_call]}
    mock_executor.run_plugin.return_value = {"result": "still running"}

    result = loop.process_request("Run forever")
    assert "maximum iterations" in result
    assert mock_llm.chat.call_count == 5  # max_iterations=5

