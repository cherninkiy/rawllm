"""Factory for constructing LLM clients from environment/config.

Usage::

    from core.llm.factory import get_llm_client
    llm = get_llm_client("anthropic")  # reads ANTHROPIC_API_KEY from env
"""

import os
from typing import Any

from core.llm.protocol import LLMClientProtocol
from core.llm.registry import LLM_PROVIDERS


def get_llm_client(provider: str | None = None) -> LLMClientProtocol:
    """Construct and return the appropriate LLM client for *provider*.

    If *provider* is ``None`` the value of the ``LLM_PROVIDER`` environment
    variable is used (default: ``"anthropic"``).

    Args:
        provider: One of the keys in :data:`core.llm.registry.LLM_PROVIDERS`.

    Raises:
        RuntimeError: If *provider* is unknown or its API key is missing.
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    cfg: dict[str, Any] | None = LLM_PROVIDERS.get(provider)
    if cfg is None:
        raise RuntimeError(
            f"Unknown LLM provider {provider!r}. "
            f"Choose from: {sorted(LLM_PROVIDERS)}"
        )

    api_key_env: str | None = cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set. "
                "Add it to your .env file or export it before running."
            )
    else:
        api_key = ""

    model: str = os.environ.get("LLM_MODEL", cfg["model"])

    if provider == "anthropic":
        from core.llm.clients.anthropic import AnthropicClient

        return AnthropicClient(api_key=api_key, model=model)

    # All other providers use an OpenAI-compatible endpoint.
    from core.llm.clients.openai_compat import OpenAICompatibleClient

    base_url: str = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)
