from pathlib import Path

import core.config as config
import run


def test_parse_args_reads_resource_flags() -> None:
    args = run._parse_args(
        ["--ports", "8000-8002", "--workspace", "/tmp/workspace", "--services", "postgres:postgresql://db", "--prompt", "build http"]
    )

    assert args.ports == "8000-8002"
    assert args.workspace == "/tmp/workspace"
    assert args.services == "postgres:postgresql://db"
    assert args.prompt == "build http"


def test_load_or_default_system_prompt_reads_file(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_PATH", str(prompt_path))

    assert run._load_or_default_system_prompt() == "system prompt"


def test_load_or_default_system_prompt_falls_back_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(config, "SYSTEM_PROMPT_PATH", "/tmp/does-not-exist-system-prompt.txt")

    assert run._load_or_default_system_prompt() == run.DEFAULT_SYSTEM_PROMPT
