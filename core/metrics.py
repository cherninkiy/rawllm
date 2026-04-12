"""Append-only JSON-lines metrics log for plugin events."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS_FILE = Path("metrics.jsonl")

logger = logging.getLogger(__name__)


def log_event(event_type: str, data: dict[str, Any]) -> None:
    """Append a single JSON line to *metrics.jsonl*."""
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    try:
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write metrics entry")


def log_execution(
    plugin_name: str,
    version: str,
    execution_time_ms: float,
    success: bool,
    error_type: str | None = None,
    traceback_str: str | None = None,
    import_risk_score: int = 0,
) -> None:
    """Log a single plugin execution."""
    log_event(
        "plugin_execution",
        {
            "plugin_name": plugin_name,
            "version": version,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error_type": error_type,
            "traceback": traceback_str,
            "import_risk_score": import_risk_score,
        },
    )


def log_version_change(plugin_name: str, old_version: str, new_version: str) -> None:
    log_event("version_change", {"plugin_name": plugin_name, "old_version": old_version, "new_version": new_version})


def log_rollback(plugin_name: str, from_version: str, to_version: str) -> None:
    log_event("rollback", {"plugin_name": plugin_name, "from_version": from_version, "to_version": to_version})


def log_dependency_request(plugin_name: str, requested: list[str], pending: list[str]) -> None:
    log_event("dependency_request", {"plugin_name": plugin_name, "requested": requested, "pending": pending})
