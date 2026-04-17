"""Runtime configuration for the orchestrator."""

import os
import tempfile
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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

# Sandbox backend: "subprocess" (legacy) or "docker" (isolated FS/users).
SANDBOX_BACKEND: str = os.environ.get("SANDBOX_BACKEND", "subprocess").strip().lower()

# User separation policy.
RAWLLM_CORE_USER: str = os.environ.get("RAWLLM_CORE_USER", "rawllm-core")
SANDBOX_PLUGIN_USER: str = os.environ.get("SANDBOX_PLUGIN_USER", "rawllm-plugin")

# Docker image and sync helper image.
SANDBOX_DOCKER_IMAGE: str = os.environ.get("SANDBOX_DOCKER_IMAGE", "rawllm/plugin-sandbox:latest")
SANDBOX_DOCKER_SYNC_IMAGE: str = os.environ.get("SANDBOX_DOCKER_SYNC_IMAGE", "alpine:3.20")

# Isolated docker volumes and mount points.
SANDBOX_WORKSPACE_VOLUME: str = os.environ.get("SANDBOX_WORKSPACE_VOLUME", "rawllm_workspace")
SANDBOX_CORE_REPO_VOLUME: str = os.environ.get("SANDBOX_CORE_REPO_VOLUME", "rawllm_core_repo")
SANDBOX_PLUGIN_STORE_VOLUME: str = os.environ.get("SANDBOX_PLUGIN_STORE_VOLUME", "rawllm_plugin_store")

SANDBOX_WORKSPACE_MOUNT: str = os.environ.get("SANDBOX_WORKSPACE_MOUNT", "/workspace")
SANDBOX_CORE_REPO_MOUNT: str = os.environ.get("SANDBOX_CORE_REPO_MOUNT", "/core_repo")
SANDBOX_PLUGIN_STORE_MOUNT: str = os.environ.get("SANDBOX_PLUGIN_STORE_MOUNT", "/plugin_store")

# Source directories copied into read-only volumes before plugin execution.
SANDBOX_CORE_REPO_SOURCE: str = os.environ.get("SANDBOX_CORE_REPO_SOURCE", ".")
SANDBOX_PLUGIN_STORE_SOURCE: str = os.environ.get("SANDBOX_PLUGIN_STORE_SOURCE", "plugins")
SANDBOX_STAGING_DIR: str = os.environ.get(
    "SANDBOX_STAGING_DIR",
    str(Path(tempfile.gettempdir()) / "rawllm_staging"),
)

# Hard fail when docker backend is requested but not available.
SANDBOX_DOCKER_REQUIRED: bool = _env_bool("SANDBOX_DOCKER_REQUIRED", True)

# File paths – all configurable via environment variables.
METRICS_FILE: Path = Path(os.environ.get("METRICS_FILE", "metrics.jsonl"))
PENDING_REQUIREMENTS_FILE: Path = Path(
    os.environ.get("PENDING_REQUIREMENTS_FILE", "pending_requirements.txt")
)
PLUGINS_DIR: str = os.environ.get("PLUGINS_DIR", "plugins")
SYSTEM_PROMPT_PATH: str = os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")
