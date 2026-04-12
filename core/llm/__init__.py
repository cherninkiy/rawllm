"""LLM abstraction layer.

Public API::

    from core.llm import LLMClientProtocol, get_llm_client

    client = get_llm_client("anthropic")   # or omit for LLM_PROVIDER env var
"""

from core.llm.factory import get_llm_client
from core.llm.protocol import LLMClientProtocol

__all__ = ["LLMClientProtocol", "get_llm_client"]
