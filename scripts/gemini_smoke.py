from __future__ import annotations

import os
import faulthandler

from dotenv import load_dotenv

from core.llm.factory import get_llm_client


def main() -> None:
    faulthandler.dump_traceback_later(8, repeat=False)
    print("step:load_env", flush=True)
    load_dotenv(".env")
    os.environ["LLM_PROVIDER"] = "gemini"
    print("step:create_client", flush=True)
    client = get_llm_client("gemini")
    print("step:chat", flush=True)
    response = client.chat(
        messages=[{"role": "user", "content": "Reply with TEST_OK only"}],
        tools=[],
    )
    print("step:done", flush=True)
    print(response)


if __name__ == "__main__":
    main()