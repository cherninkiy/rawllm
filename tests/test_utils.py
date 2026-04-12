"""Tests for core.utils helper functions."""

import os
from pathlib import Path

import pytest

from core.utils import (
    configure_logging,
    ensure_dir,
    extract_imports,
    get_api_key,
    load_env,
    read_system_prompt,
)


def test_extract_imports_simple() -> None:
    code = "import os\nimport sys\n\ndef run(d): return {}\n"
    result = extract_imports(code)
    assert "os" in result
    assert "sys" in result


def test_extract_imports_from_style() -> None:
    code = "from pathlib import Path\nfrom collections import defaultdict\n\ndef run(d): return {}\n"
    result = extract_imports(code)
    assert "pathlib" in result
    assert "collections" in result


def test_extract_imports_dotted() -> None:
    code = "import os.path\n\ndef run(d): return {}\n"
    result = extract_imports(code)
    assert "os" in result


def test_extract_imports_invalid_syntax_returns_empty() -> None:
    result = extract_imports("this is not python code !!@#")
    assert result == []


def test_extract_imports_empty_code() -> None:
    result = extract_imports("")
    assert result == []


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    new_dir = tmp_path / "sub" / "deep"
    result = ensure_dir(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    ensure_dir(tmp_path)  # already exists – should not raise
    assert tmp_path.is_dir()


def test_read_system_prompt(tmp_path: Path) -> None:
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text("  You are helpful.  \n", encoding="utf-8")
    result = read_system_prompt(prompt_file)
    assert result == "You are helpful."


def test_read_system_prompt_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_system_prompt(tmp_path / "missing.txt")


def test_load_env_does_not_raise_when_file_missing(tmp_path: Path) -> None:
    # Should silently succeed even if .env doesn't exist.
    load_env(tmp_path / ".env")


def test_load_env_reads_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_LOAD_ENV_VAR=hello123\n", encoding="utf-8")
    # Only override=False, so existing env vars win; but this key shouldn't exist yet.
    os.environ.pop("TEST_LOAD_ENV_VAR", None)
    load_env(env_file)
    assert os.environ.get("TEST_LOAD_ENV_VAR") == "hello123"
    del os.environ["TEST_LOAD_ENV_VAR"]


def test_get_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret123")
    result = get_api_key("MY_API_KEY")
    assert result == "secret123"


def test_get_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_API_KEY_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="MY_API_KEY_MISSING"):
        get_api_key("MY_API_KEY_MISSING")


def test_configure_logging_does_not_raise() -> None:
    configure_logging()  # Should not raise even if called multiple times.
    configure_logging()
