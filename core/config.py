"""Runtime configuration for the orchestrator."""

import os
from pathlib import Path

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

# File paths – all configurable via environment variables.
METRICS_FILE: Path = Path(os.environ.get("METRICS_FILE", "metrics.jsonl"))
PENDING_REQUIREMENTS_FILE: Path = Path(
    os.environ.get("PENDING_REQUIREMENTS_FILE", "pending_requirements.txt")
)
PLUGINS_DIR: str = os.environ.get("PLUGINS_DIR", "plugins")
SYSTEM_PROMPT_PATH: str = os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")
