"""Prompt for RawLLM: act as an HTTP transport bridge.

Role:
1. Expose HTTP API for incoming user prompts.
2. Forward each valid request to the orchestrator callback.
3. Return callback output as JSON without mutating business logic.

Contract:
- POST / accepts JSON: {"prompt": str, "context": dict}
- Successful response: {"answer": str}
- Error response: {"error": str} with proper HTTP status code
- GET / returns a lightweight status page for health checks

Operational constraints:
- Keep server in a daemon thread so orchestrator remains responsive.
- Do not import from core modules directly.
- Handle malformed input defensively and never crash server loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

logger = logging.getLogger(__name__)

_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None
_callback: Callable[[str, dict], str] | None = None

PORT = int(os.environ.get("HTTP_PORT", "8080"))


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Handle individual HTTP requests."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("HTTP %s", format % args)

    def do_GET(self) -> None:  # noqa: N802
        """Return a simple HTML status page."""
        body = (
            b"<html><body>"
            b"<h1>RawLLM - HTTP Plugin</h1>"
            b"<p>POST / with JSON <code>{\"prompt\": \"...\", \"context\": {}}</code></p>"
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        """Handle an inference request."""
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        prompt: str = payload.get("prompt", "")
        context: dict = payload.get("context", {})

        if not prompt:
            self._send_error(400, "Missing 'prompt' field.")
            return

        if _callback is None:
            self._send_error(503, "Orchestrator callback not set.")
            return

        try:
            answer = _callback(prompt, context)
        except Exception as exc:
            logger.exception("Callback raised an exception")
            self._send_error(500, str(exc))
            return

        response_body = json.dumps({"answer": answer}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------


def init(callback: Callable[[str, dict], str]) -> None:
    """Start the HTTP server and store the *callback* for routing requests.

    Args:
        callback: A function ``(prompt: str, context: dict) -> str`` that
                  processes a user request and returns the answer.
    """
    global _server, _server_thread, _callback

    _callback = callback

    _server = HTTPServer(("0.0.0.0", PORT), _Handler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    logger.info("HTTP plugin listening on port %d", PORT)


def shutdown() -> None:
    """Stop the HTTP server gracefully."""
    global _server, _server_thread

    if _server is not None:
        _server.shutdown()
        _server = None
    if _server_thread is not None:
        _server_thread.join(timeout=5)
        _server_thread = None
    logger.info("HTTP plugin stopped.")


def run(input_data: dict) -> dict:
    """Control endpoint for the HTTP plugin.

    Supported actions:
        ``{"action": "status"}`` – returns the current server status.
        ``{"action": "stop"}``   – stops the server (same as shutdown()).

    Returns:
        A dict with ``{"status": "running"}`` or ``{"status": "stopped"}``.
    """
    action = input_data.get("action", "status")

    if action == "stop":
        shutdown()
        return {"status": "stopped"}

    return {"status": "running" if _server is not None else "stopped", "port": PORT}
