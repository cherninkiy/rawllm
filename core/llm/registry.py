"""LLM provider registry: maps provider names to connection settings."""

from typing import Any

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
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-coder:free",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
    },
    "ollama-qwen-coder": {
        "api_key_env": None,
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5-coder:7b",
    },
}
