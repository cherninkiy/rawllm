from pathlib import Path

from core.prompt_builder import build_startup_prompt


def test_build_startup_prompt_includes_available_resources() -> None:
    prompt = build_startup_prompt(
        {
            "ports": [8000, 8001, 8002],
            "workspace": Path("/workspace"),
            "services": {"postgres": "postgresql://db", "redis": "redis://cache"},
        },
        "create an HTTP interface",
    )

    assert "8000-8002" in prompt
    assert "/workspace" in prompt
    assert "postgres, redis" in prompt
    assert "create an HTTP interface" in prompt


def test_build_startup_prompt_uses_default_task() -> None:
    prompt = build_startup_prompt({"ports": [], "workspace": None, "services": {}}, None)
    assert "create an interface for user interaction using these resources." in prompt
