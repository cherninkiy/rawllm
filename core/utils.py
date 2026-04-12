"""Utility helpers for the orchestrator."""

import ast
import logging
import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_file: str | Path = ".env") -> None:
    """Load environment variables from *env_file* (defaults to ``.env`` in CWD).

    Does nothing if the file does not exist – environment variables already set
    in the process (e.g. from CI/Docker secrets) take precedence.
    """
    load_dotenv(dotenv_path=env_file, override=False)


def read_system_prompt(path: str | Path = "system_prompt.txt") -> str:
    """Read and return the system prompt from *path*.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    return Path(path).read_text(encoding="utf-8").strip()


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any parents) if it does not already exist.

    Returns the resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_api_key(env_var: str = "ANTHROPIC_API_KEY") -> str:
    """Return the API key from *env_var*, raising ``RuntimeError`` if absent."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var!r} is not set. "
            "Add it to your .env file or export it before running."
        )
    return key


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a sensible format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


def extract_imports(code: str) -> list[str]:
    """Return a list of top-level module names imported by *code*.

    Uses AST parsing so only syntactically valid code produces results.
    Returns an empty list if parsing fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names
