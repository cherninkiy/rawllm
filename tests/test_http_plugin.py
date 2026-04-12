"""Tests for the HTTP plugin."""

import json
import threading
import time
import urllib.request
import urllib.error
from typing import Iterator

import pytest

# Import the plugin module directly (it must not import from core)
import importlib.util
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parent.parent / "plugins" / "http.py"

spec = importlib.util.spec_from_file_location("plugins.http", PLUGIN_PATH)
http_plugin = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(http_plugin)  # type: ignore[union-attr]


TEST_PORT = 18080  # Use a different port to avoid conflicts with production server


def _patch_port(port: int) -> None:
    """Override the plugin's PORT constant for testing."""
    http_plugin.PORT = port


@pytest.fixture(autouse=True)
def reset_plugin() -> Iterator[None]:
    """Ensure plugin state is reset between tests."""
    _patch_port(TEST_PORT)
    yield
    http_plugin.shutdown()
    http_plugin._callback = None
    http_plugin._server = None
    http_plugin._server_thread = None


def _make_callback(response: str = "test answer") -> threading.Event:
    """Return (callback, called_event) where called_event is set when callback runs."""
    called = threading.Event()

    def callback(prompt: str, context: dict) -> str:
        called.set()
        return response

    http_plugin._callback = callback
    return called


def _wait_for_server(port: int, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5)
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server on port {port} did not start within {timeout}s")


# ---------------------------------------------------------------------------


def test_init_starts_server() -> None:
    called = _make_callback("hello")
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)
    assert http_plugin._server is not None


def test_get_request_returns_html() -> None:
    _make_callback()
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)

    response = urllib.request.urlopen(f"http://127.0.0.1:{TEST_PORT}/")
    assert response.status == 200
    body = response.read().decode()
    assert "RawLLM" in body


def test_post_request_calls_callback() -> None:
    called = threading.Event()

    def callback(prompt: str, context: dict) -> str:
        called.set()
        return "pong"

    http_plugin.init(callback=callback)
    _wait_for_server(TEST_PORT)

    payload = json.dumps({"prompt": "ping", "context": {}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{TEST_PORT}/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())

    assert called.is_set()
    assert body == {"answer": "pong"}


def test_post_missing_prompt_returns_400() -> None:
    _make_callback()
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)

    payload = json.dumps({"context": {}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{TEST_PORT}/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        pytest.fail("Expected HTTPError 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        body = json.loads(exc.read())
        assert "error" in body


def test_post_invalid_json_returns_400() -> None:
    _make_callback()
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)

    req = urllib.request.Request(
        f"http://127.0.0.1:{TEST_PORT}/",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        pytest.fail("Expected HTTPError 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_run_returns_status() -> None:
    _make_callback()
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)

    result = http_plugin.run({"action": "status"})
    assert result["status"] == "running"
    assert result["port"] == TEST_PORT


def test_run_stop_shuts_down_server() -> None:
    _make_callback()
    http_plugin.init(callback=http_plugin._callback)
    _wait_for_server(TEST_PORT)

    result = http_plugin.run({"action": "stop"})
    assert result["status"] == "stopped"
    assert http_plugin._server is None


def test_context_is_passed_to_callback() -> None:
    received: dict = {}

    def callback(prompt: str, context: dict) -> str:
        received["prompt"] = prompt
        received["context"] = context
        return "ok"

    http_plugin.init(callback=callback)
    _wait_for_server(TEST_PORT)

    payload = json.dumps({"prompt": "hello", "context": {"user": "alice"}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{TEST_PORT}/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req)

    assert received["prompt"] == "hello"
    assert received["context"] == {"user": "alice"}
