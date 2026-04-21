"""Helpers for assembling the runtime startup prompt."""

from __future__ import annotations

import core.config as config


def _format_port_ranges(ports: list[int]) -> str:
    """Format a list of ports as compact range notation (e.g. 8000-8002, 8080)."""
    sorted_ports = sorted(ports)
    ranges: list[str] = []
    start = end = sorted_ports[0]
    for p in sorted_ports[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if end > start else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if end > start else str(start))
    return ", ".join(ranges)


def build_startup_prompt(
    available_resources: dict[str, object] | None = None,
    user_task: str | None = None,
) -> str:
    resources = available_resources or {
        "ports": list(config.AVAILABLE_PORTS),
        "workspace": config.WORKSPACE_PATH,
        "services": dict(config.AVAILABLE_SERVICES),
    }

    lines = ["Available resources in this session:"]

    raw_ports = resources.get("ports")
    if isinstance(raw_ports, list) and raw_ports:
        lines.append(f"- Network ports: {_format_port_ranges(raw_ports)}")

    workspace = resources.get("workspace")
    if workspace:
        lines.append(f"- Workspace directory: {workspace} (read/write)")

    raw_services = resources.get("services")
    if isinstance(raw_services, dict) and raw_services:
        lines.append(f"- Services: {', '.join(sorted(raw_services))}")

    task = user_task or "create an interface for user interaction using these resources."
    lines.append("")
    lines.append(f"Your task: {task}")
    return "\n".join(lines)
