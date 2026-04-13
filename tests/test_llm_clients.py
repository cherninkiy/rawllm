"""Tests for core.llm.clients – AnthropicClient and OpenAICompatibleClient."""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm.clients.anthropic import AnthropicClient
from core.llm.clients.openai_compat import OpenAICompatibleClient
from core.llm.protocol import LLMClientProtocol


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_anthropic_client_satisfies_protocol() -> None:
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = "test-model"
    # Protocol is structural – checking isinstance is enough.
    assert isinstance(client, LLMClientProtocol)


def test_openai_compat_client_satisfies_protocol() -> None:
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    assert isinstance(client, LLMClientProtocol)


# ---------------------------------------------------------------------------
# AnthropicClient._convert_tools
# ---------------------------------------------------------------------------


@pytest.fixture()
def anthropic_client() -> AnthropicClient:
    client = AnthropicClient.__new__(AnthropicClient)
    client.model = "claude-test"
    client._client = MagicMock()
    return client


def test_convert_tools_openai_format(anthropic_client: AnthropicClient) -> None:
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "description": "Does a thing",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = anthropic_client._convert_tools(openai_tools)
    assert len(result) == 1
    assert result[0]["name"] == "do_thing"
    assert result[0]["description"] == "Does a thing"
    assert "input_schema" in result[0]
    assert result[0]["input_schema"] == {"type": "object", "properties": {}}


def test_convert_tools_already_anthropic_format(anthropic_client: AnthropicClient) -> None:
    """Handles tools that already use the Anthropic-style 'name'/'input_schema' keys."""
    anthropic_tools = [
        {
            "name": "my_tool",
            "description": "desc",
            "input_schema": {"type": "object"},
        }
    ]
    result = anthropic_client._convert_tools(anthropic_tools)
    assert result[0]["name"] == "my_tool"
    assert result[0]["input_schema"] == {"type": "object"}


def test_convert_tools_empty_list(anthropic_client: AnthropicClient) -> None:
    assert anthropic_client._convert_tools([]) == []


# ---------------------------------------------------------------------------
# AnthropicClient._convert_messages
# ---------------------------------------------------------------------------


def test_convert_messages_user_passthrough(anthropic_client: AnthropicClient) -> None:
    messages = [{"role": "user", "content": "Hello"}]
    result = anthropic_client._convert_messages(messages)
    assert result == messages


def test_convert_messages_skips_system(anthropic_client: AnthropicClient) -> None:
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hi"},
    ]
    result = anthropic_client._convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_convert_messages_assistant_with_tool_calls(anthropic_client: AnthropicClient) -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "run_plugin", "arguments": '{"name": "echo"}'},
                }
            ],
        }
    ]
    result = anthropic_client._convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    content = result[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "tool_use"
    assert content[0]["id"] == "tc1"
    assert content[0]["name"] == "run_plugin"
    assert content[0]["input"] == {"name": "echo"}


def test_convert_messages_assistant_with_text_and_tool_calls(anthropic_client: AnthropicClient) -> None:
    """Text content before tool_calls should be preserved as a text block."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me run that",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "run_plugin", "arguments": "{}"},
                }
            ],
        }
    ]
    result = anthropic_client._convert_messages(messages)
    content = result[0]["content"]
    assert content[0] == {"type": "text", "text": "Let me run that"}
    assert content[1]["type"] == "tool_use"


def test_convert_messages_tool_results_grouped(anthropic_client: AnthropicClient) -> None:
    """Multiple consecutive role:tool messages → single role:user with tool_result blocks."""
    messages = [
        {"role": "tool", "tool_call_id": "tc1", "content": '{"result": 1}'},
        {"role": "tool", "tool_call_id": "tc2", "content": '{"result": 2}'},
    ]
    result = anthropic_client._convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "tool_result", "tool_use_id": "tc1", "content": '{"result": 1}'}
    assert content[1] == {"type": "tool_result", "tool_use_id": "tc2", "content": '{"result": 2}'}


def test_convert_messages_malformed_tool_call_arguments(anthropic_client: AnthropicClient) -> None:
    """Malformed JSON in tool call arguments should not raise."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "run_plugin", "arguments": "NOT JSON"},
                }
            ],
        }
    ]
    result = anthropic_client._convert_messages(messages)
    assert result[0]["content"][0]["input"] == {}


# ---------------------------------------------------------------------------
# AnthropicClient.chat – mocked API
# ---------------------------------------------------------------------------


def test_anthropic_client_chat_returns_text(anthropic_client: AnthropicClient) -> None:
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_text_block = MagicMock()
    mock_text_block.text = "Hello, world!"
    mock_response.content = [mock_text_block]
    anthropic_client._client.messages.create.return_value = mock_response

    result = anthropic_client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        system="Be helpful",
    )
    assert result == {"type": "text", "content": "Hello, world!"}


def test_anthropic_client_chat_returns_tool_calls(anthropic_client: AnthropicClient) -> None:
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "tc1"
    mock_tool_block.name = "run_plugin"
    mock_tool_block.input = {"name": "echo"}
    mock_response.content = [mock_tool_block]
    anthropic_client._client.messages.create.return_value = mock_response

    result = anthropic_client.chat(
        messages=[{"role": "user", "content": "Run echo"}],
        tools=[],
    )
    assert result["type"] == "tool_calls"
    assert result["tool_calls"][0] == {"id": "tc1", "name": "run_plugin", "input": {"name": "echo"}}


def test_anthropic_client_chat_raises_on_api_error(anthropic_client: AnthropicClient) -> None:
    import anthropic as anthropic_sdk
    anthropic_client._client.messages.create.side_effect = anthropic_sdk.APIError(
        message="rate limit", request=MagicMock(), body=None
    )
    with pytest.raises(anthropic_sdk.APIError):
        anthropic_client.chat(messages=[{"role": "user", "content": "Hi"}], tools=[])


# ---------------------------------------------------------------------------
# OpenAICompatibleClient.chat – mocked API
# ---------------------------------------------------------------------------


@pytest.fixture()
def openai_client() -> OpenAICompatibleClient:
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.model = "llama3"
    client._base_url = "https://example.test/v1"
    client._headers = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}
    return client


def test_openai_client_chat_returns_text(openai_client: OpenAICompatibleClient) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "Hello from Llama!", "tool_calls": None},
            }
        ]
    }
    with patch("core.llm.clients.openai_compat.httpx.post", return_value=mock_response):
        result = openai_client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            system="Be helpful",
        )
    assert result == {"type": "text", "content": "Hello from Llama!"}


def test_openai_client_chat_returns_tool_calls(openai_client: OpenAICompatibleClient) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "run_plugin", "arguments": '{"name": "echo"}'},
                        }
                    ],
                },
            }
        ]
    }
    with patch("core.llm.clients.openai_compat.httpx.post", return_value=mock_response):
        result = openai_client.chat(
            messages=[{"role": "user", "content": "Run echo"}],
            tools=[{"type": "function", "function": {"name": "run_plugin"}}],
        )
    assert result["type"] == "tool_calls"
    assert result["tool_calls"][0] == {"id": "tc1", "name": "run_plugin", "input": {"name": "echo"}}


def test_openai_client_chat_injects_system_as_first_message(openai_client: OpenAICompatibleClient) -> None:
    """The system prompt should be prepended as a system message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "OK", "tool_calls": None}}]
    }
    with patch("core.llm.clients.openai_compat.httpx.post", return_value=mock_response) as mock_post:
        openai_client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
            system="You are a helpful assistant",
        )

    call_kwargs = mock_post.call_args[1]["json"]
    messages_sent = call_kwargs["messages"]
    assert messages_sent[0] == {"role": "system", "content": "You are a helpful assistant"}
    assert messages_sent[1] == {"role": "user", "content": "Hello"}


def test_openai_client_chat_malformed_tool_call_arguments(openai_client: OpenAICompatibleClient) -> None:
    """Malformed JSON in tool call arguments should not raise."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "tool", "arguments": "INVALID JSON"},
                        }
                    ]
                },
            }
        ]
    }
    with patch("core.llm.clients.openai_compat.httpx.post", return_value=mock_response):
        result = openai_client.chat(messages=[], tools=[])
    assert result["tool_calls"][0]["input"] == {}


# ---------------------------------------------------------------------------
# core.llm.factory
# ---------------------------------------------------------------------------


def test_factory_unknown_provider_raises() -> None:
    from core.llm.factory import get_llm_client
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        get_llm_client("unknown_provider_xyz")


def test_factory_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm_client("anthropic")


def test_factory_returns_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("anthropic.Anthropic"):
        client = get_llm_client("anthropic")
    assert isinstance(client, AnthropicClient)


def test_factory_returns_openai_compat_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = get_llm_client("groq")
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_reads_llm_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_overrides_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "claude-custom")
    with patch("anthropic.Anthropic"):
        client = get_llm_client("anthropic")
    assert client.model == "claude-custom"


def test_factory_ollama_no_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client
    client = get_llm_client("ollama")
    assert isinstance(client, OpenAICompatibleClient)


def test_factory_ollama_qwen_coder_no_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import get_llm_client

    client = get_llm_client("ollama-qwen-coder")

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "qwen2.5-coder:7b"
