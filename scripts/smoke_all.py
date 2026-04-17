#!/usr/bin/env python3
"""Smoke-test all LLM providers defined in .env.

Usage:
    PYTHONPATH=. python scripts/smoke_all.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from core.llm.factory import get_llm_client
from core.llm.registry import LLM_PROVIDERS

PROMPT = [{"role": "user", "content": "Reply with TEST_OK only"}]

# Providers to test and their key env var (None = no key needed, just check service)
PROVIDERS_TO_TEST = [
    "openrouter",
    "ollama-qwen-coder",
]


def test_provider(provider: str) -> tuple[bool | None, str]:
    cfg = LLM_PROVIDERS.get(provider, {})
    key_var = cfg.get("api_key_env")

    if key_var and not os.environ.get(key_var):
        return None, f"{key_var} not set — skipped"

    try:
        client = get_llm_client(provider)
        resp = client.chat(messages=PROMPT, tools=[])
        text = resp if isinstance(resp, str) else str(resp)
        return True, text[:120]
    except Exception as exc:
        return False, str(exc)[:300]


def main() -> None:
    print("=== LLM provider smoke tests ===\n")

    passed = failed = skipped = 0

    for provider in PROVIDERS_TO_TEST:
        print(f"  testing {provider} ...", flush=True)
        ok, msg = test_provider(provider)

        if ok is None:
            print(f"  [ SKIP ] {provider}: {msg}\n")
            skipped += 1
        elif ok:
            print(f"  [  OK  ] {provider}: {msg}\n")
            passed += 1
        else:
            print(f"  [ FAIL ] {provider}: {msg}\n")
            failed += 1

    print(f"=== Results: {passed} passed, {failed} failed, {skipped} skipped ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
