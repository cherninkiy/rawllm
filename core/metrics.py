"""Append-only JSON-lines metrics log for plugin events."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import METRICS_FILE

logger = logging.getLogger(__name__)


def log_event(event_type: str, data: dict[str, Any], metrics_file: Path | None = None) -> None:
    """Append a single JSON line to *metrics.jsonl*."""
    path = metrics_file if metrics_file is not None else METRICS_FILE
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except (OSError, PermissionError):  # noqa: BLE001
        logger.exception("Failed to write metrics entry")


def log_execution(
    plugin_name: str,
    version: str,
    execution_time_ms: float,
    success: bool,
    error_type: str | None = None,
    traceback_str: str | None = None,
    import_risk_score: int = 0,
    metrics_file: Path | None = None,
    success_score: float | None = None,
    trajectory_id: str | None = None,
    step_number: int | None = None,
) -> None:
    """Log a single plugin execution.
    
    Args:
        plugin_name: Name of the executed plugin.
        version: Plugin version.
        execution_time_ms: Execution time in milliseconds.
        success: Whether execution was successful.
        error_type: Type of error if failed.
        traceback_str: Full traceback string if failed.
        import_risk_score: Risk score from import analysis.
        success_score: Float score 0.0-1.0 indicating degree of success.
        trajectory_id: Unique ID for multi-step operation sequence.
        step_number: Position in the trajectory (1-indexed).
    """
    data: dict[str, Any] = {
        "plugin_name": plugin_name,
        "version": version,
        "execution_time_ms": execution_time_ms,
        "success": success,
        "error_type": error_type,
        "traceback": traceback_str,
        "import_risk_score": import_risk_score,
    }
    
    # Add extended tracking fields
    if success_score is not None:
        data["success_score"] = max(0.0, min(1.0, success_score))
    if trajectory_id is not None:
        data["trajectory_id"] = trajectory_id
    if step_number is not None:
        data["step_number"] = step_number
    
    log_event("plugin_execution", data, metrics_file=metrics_file)


def log_version_change(
    plugin_name: str, old_version: str, new_version: str, metrics_file: Path | None = None
) -> None:
    log_event(
        "version_change",
        {"plugin_name": plugin_name, "old_version": old_version, "new_version": new_version},
        metrics_file=metrics_file,
    )


def log_rollback(plugin_name: str, from_version: str, to_version: str, metrics_file: Path | None = None) -> None:
    log_event(
        "rollback",
        {"plugin_name": plugin_name, "from_version": from_version, "to_version": to_version},
        metrics_file=metrics_file,
    )


def log_dependency_request(
    plugin_name: str, requested: list[str], pending: list[str], metrics_file: Path | None = None
) -> None:
    log_event(
        "dependency_request",
        {"plugin_name": plugin_name, "requested": requested, "pending": pending},
        metrics_file=metrics_file,
    )


# ---------------------------------------------------------------------------
# History retrieval and aggregation
# ---------------------------------------------------------------------------


def get_events(
    plugin_name: str | None = None,
    event_type: str | None = None,
    metrics_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Read and optionally filter entries from the metrics log.

    Args:
        plugin_name: If provided, only events for this plugin are returned.
        event_type: If provided, only events of this type are returned.
        metrics_file: Override the default ``metrics.jsonl`` path (for testing).

    Returns:
        List of event dicts, in chronological order.  Returns an empty list
        if the file does not exist or contains no matching entries.
    """
    path = metrics_file if metrics_file is not None else METRICS_FILE
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if plugin_name is not None and entry.get("plugin_name") != plugin_name:
                continue
            if event_type is not None and entry.get("event") != event_type:
                continue
            events.append(entry)
    except (OSError, PermissionError):
        logger.exception("Failed to read metrics file")

    return events


def aggregate_by_plugin(
    plugin_name: str | None = None,
    metrics_file: Path | None = None,
) -> dict[str, Any]:
    """Aggregate execution statistics per plugin (or for one plugin).

    Args:
        plugin_name: If provided, return stats only for that plugin.
        metrics_file: Override the default ``metrics.jsonl`` path (for testing).

    Returns:
        A dict keyed by plugin name, each value containing::

            {
                "total_executions": int,
                "successful": int,
                "failed": int,
                "avg_exec_ms": float,
                "version_changes": int,
                "rollbacks": int,
                "dependency_requests": int,
                "import_risk_score": int,  # latest recorded score
                "avg_success_score": float,  # average success_score (0.0-1.0)
                "trajectory_count": int,     # number of unique trajectories
            }
    """
    events = get_events(plugin_name=plugin_name, metrics_file=metrics_file)

    stats: dict[str, dict[str, Any]] = {}

    def _plugin_stats(name: str) -> dict[str, Any]:
        return {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "total_exec_ms": 0.0,
            "avg_exec_ms": 0.0,
            "version_changes": 0,
            "rollbacks": 0,
            "dependency_requests": 0,
            "import_risk_score": 0,
            "total_success_score": 0.0,
            "success_score_count": 0,
            "trajectories": set(),
        }

    for entry in events:
        name = entry.get("plugin_name", "unknown")
        if name not in stats:
            stats[name] = _plugin_stats(name)
        s = stats[name]
        etype = entry.get("event")

        if etype == "plugin_execution":
            s["total_executions"] += 1
            if entry.get("success"):
                s["successful"] += 1
            else:
                s["failed"] += 1
            s["total_exec_ms"] += entry.get("execution_time_ms", 0.0)
            s["import_risk_score"] = entry.get("import_risk_score", s["import_risk_score"])
            
            # Extended success_score tracking
            if "success_score" in entry:
                s["total_success_score"] += entry["success_score"]
                s["success_score_count"] += 1
            
            # Extended trajectory tracking
            if "trajectory_id" in entry:
                s["trajectories"].add(entry["trajectory_id"])
                
        elif etype == "version_change":
            s["version_changes"] += 1
        elif etype == "rollback":
            s["rollbacks"] += 1
        elif etype == "dependency_request":
            s["dependency_requests"] += 1

    # Compute derived fields and remove internal accumulators.
    for s in stats.values():
        total = s["total_executions"]
        s["avg_exec_ms"] = s["total_exec_ms"] / total if total > 0 else 0.0
        
        # Compute avg_success_score
        if s["success_score_count"] > 0:
            s["avg_success_score"] = s["total_success_score"] / s["success_score_count"]
        else:
            s["avg_success_score"] = 0.0
        
        # Count unique trajectories
        s["trajectory_count"] = len(s["trajectories"])
        
        del s["total_exec_ms"]
        del s["total_success_score"]
        del s["success_score_count"]
        del s["trajectories"]

    return stats
