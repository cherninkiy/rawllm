"""Tests for core.metrics – event logging and aggregation."""

import json
from pathlib import Path

import pytest

import core.metrics as metrics


@pytest.fixture()
def metrics_file(tmp_path: Path) -> Path:
    return tmp_path / "test_metrics.jsonl"


# ---------------------------------------------------------------------------
# log_event / write helpers
# ---------------------------------------------------------------------------


def test_log_event_creates_file(metrics_file: Path) -> None:
    metrics.log_event("test_event", {"key": "value"}, metrics_file=metrics_file)
    assert metrics_file.exists()


def test_log_event_writes_valid_json(metrics_file: Path) -> None:
    metrics.log_event("test_event", {"key": "value"}, metrics_file=metrics_file)
    line = metrics_file.read_text(encoding="utf-8").strip()
    entry = json.loads(line)
    assert entry["event"] == "test_event"
    assert entry["key"] == "value"
    assert "timestamp" in entry


def test_log_execution(metrics_file: Path) -> None:
    metrics.log_execution(
        plugin_name="my_plugin",
        version="v1",
        execution_time_ms=42.5,
        success=True,
        import_risk_score=0,
        metrics_file=metrics_file,
    )
    entries = metrics.get_events(metrics_file=metrics_file)
    assert len(entries) == 1
    e = entries[0]
    assert e["event"] == "plugin_execution"
    assert e["plugin_name"] == "my_plugin"
    assert e["version"] == "v1"
    assert e["execution_time_ms"] == 42.5
    assert e["success"] is True


def test_log_execution_failure(metrics_file: Path) -> None:
    metrics.log_execution(
        plugin_name="bad_plugin",
        version="v1",
        execution_time_ms=5.0,
        success=False,
        error_type="ValueError",
        traceback_str="Traceback...\nValueError: oops",
        metrics_file=metrics_file,
    )
    entries = metrics.get_events(metrics_file=metrics_file)
    e = entries[0]
    assert e["success"] is False
    assert e["error_type"] == "ValueError"
    assert "ValueError" in e["traceback"]


def test_log_version_change(metrics_file: Path) -> None:
    metrics.log_version_change("plugin_a", "v1", "v2", metrics_file=metrics_file)
    entries = metrics.get_events(metrics_file=metrics_file)
    assert entries[0]["event"] == "version_change"
    assert entries[0]["old_version"] == "v1"
    assert entries[0]["new_version"] == "v2"


def test_log_rollback(metrics_file: Path) -> None:
    metrics.log_rollback("plugin_a", "v3", "v2", metrics_file=metrics_file)
    entries = metrics.get_events(metrics_file=metrics_file)
    assert entries[0]["event"] == "rollback"
    assert entries[0]["from_version"] == "v3"
    assert entries[0]["to_version"] == "v2"


def test_log_dependency_request(metrics_file: Path) -> None:
    metrics.log_dependency_request("plugin_a", ["requests", "os"], ["requests"], metrics_file=metrics_file)
    entries = metrics.get_events(metrics_file=metrics_file)
    assert entries[0]["event"] == "dependency_request"
    assert "requests" in entries[0]["requested"]
    assert "requests" in entries[0]["pending"]


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


def test_get_events_returns_empty_for_missing_file(tmp_path: Path) -> None:
    result = metrics.get_events(metrics_file=tmp_path / "nonexistent.jsonl")
    assert result == []


def test_get_events_filters_by_plugin_name(metrics_file: Path) -> None:
    metrics.log_execution("alpha", "v1", 10.0, True, metrics_file=metrics_file)
    metrics.log_execution("beta", "v1", 20.0, False, metrics_file=metrics_file)
    events = metrics.get_events(plugin_name="alpha", metrics_file=metrics_file)
    assert len(events) == 1
    assert events[0]["plugin_name"] == "alpha"


def test_get_events_filters_by_event_type(metrics_file: Path) -> None:
    metrics.log_execution("alpha", "v1", 10.0, True, metrics_file=metrics_file)
    metrics.log_version_change("alpha", "v1", "v2", metrics_file=metrics_file)
    events = metrics.get_events(event_type="version_change", metrics_file=metrics_file)
    assert len(events) == 1
    assert events[0]["event"] == "version_change"


def test_get_events_filters_by_both(metrics_file: Path) -> None:
    metrics.log_execution("alpha", "v1", 10.0, True, metrics_file=metrics_file)
    metrics.log_execution("beta", "v1", 20.0, False, metrics_file=metrics_file)
    metrics.log_version_change("alpha", "v1", "v2", metrics_file=metrics_file)
    events = metrics.get_events(plugin_name="alpha", event_type="plugin_execution", metrics_file=metrics_file)
    assert len(events) == 1
    assert events[0]["plugin_name"] == "alpha"
    assert events[0]["event"] == "plugin_execution"


def test_get_events_skips_malformed_lines(metrics_file: Path) -> None:
    metrics_file.write_text('NOT JSON\n{"event":"ok","plugin_name":"p","timestamp":"t"}\n', encoding="utf-8")
    events = metrics.get_events(metrics_file=metrics_file)
    assert len(events) == 1
    assert events[0]["event"] == "ok"


# ---------------------------------------------------------------------------
# aggregate_by_plugin
# ---------------------------------------------------------------------------


def test_aggregate_empty_returns_empty(tmp_path: Path) -> None:
    result = metrics.aggregate_by_plugin(metrics_file=tmp_path / "missing.jsonl")
    assert result == {}


def test_aggregate_counts_executions(metrics_file: Path) -> None:
    metrics.log_execution("calc", "v1", 10.0, True, metrics_file=metrics_file)
    metrics.log_execution("calc", "v1", 20.0, True, metrics_file=metrics_file)
    metrics.log_execution("calc", "v1", 30.0, False, error_type="ValueError", metrics_file=metrics_file)

    result = metrics.aggregate_by_plugin("calc", metrics_file=metrics_file)
    assert "calc" in result
    stats = result["calc"]
    assert stats["total_executions"] == 3
    assert stats["successful"] == 2
    assert stats["failed"] == 1
    assert abs(stats["avg_exec_ms"] - 20.0) < 0.01


def test_aggregate_counts_version_events(metrics_file: Path) -> None:
    metrics.log_version_change("myplugin", "v1", "v2", metrics_file=metrics_file)
    metrics.log_version_change("myplugin", "v2", "v3", metrics_file=metrics_file)
    metrics.log_rollback("myplugin", "v3", "v2", metrics_file=metrics_file)
    metrics.log_dependency_request("myplugin", ["requests"], ["requests"], metrics_file=metrics_file)

    result = metrics.aggregate_by_plugin("myplugin", metrics_file=metrics_file)
    stats = result["myplugin"]
    assert stats["version_changes"] == 2
    assert stats["rollbacks"] == 1
    assert stats["dependency_requests"] == 1


def test_aggregate_all_plugins(metrics_file: Path) -> None:
    metrics.log_execution("plugin_a", "v1", 10.0, True, metrics_file=metrics_file)
    metrics.log_execution("plugin_b", "v1", 20.0, False, metrics_file=metrics_file)

    result = metrics.aggregate_by_plugin(metrics_file=metrics_file)
    assert "plugin_a" in result
    assert "plugin_b" in result
    assert result["plugin_a"]["successful"] == 1
    assert result["plugin_b"]["failed"] == 1
