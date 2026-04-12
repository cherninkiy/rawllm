"""Runtime configuration for the orchestrator."""

import os
from typing import Any

# Plugins that run in-process (trusted). All others run sandboxed via subprocess.
# Comma-separated list, e.g. TRUSTED_PLUGINS="http,logger"
TRUSTED_PLUGINS: list[str] = [
    p.strip() for p in os.environ.get("TRUSTED_PLUGINS", "").split(",") if p.strip()
]

# Modules allowed without human approval.  Anything outside this list is
# written to pending_requirements.txt and the plugin is held back.
_DEFAULT_ALLOWED = "json,datetime,math,re,collections,itertools,typing"
ALLOWED_REQUIREMENTS: list[str] = [
    r.strip()
    for r in os.environ.get("ALLOWED_REQUIREMENTS", _DEFAULT_ALLOWED).split(",")
    if r.strip()
]

# Seconds before a sandboxed subprocess is killed.
SANDBOX_TIMEOUT: int = int(os.environ.get("SANDBOX_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# LLM provider registry
# ---------------------------------------------------------------------------
# Each entry describes how to construct an LLM client for that provider.
# Keys:
#   api_key_env  – environment variable name for the API key (None = no key needed)
#   base_url     – base URL for the OpenAI-compatible endpoint (absent for Anthropic)
#   model        – default model identifier

LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-3-5-sonnet-20241022",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama3-70b-8192",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "mistralai/mistral-7b-instruct:free",
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
    },
}
