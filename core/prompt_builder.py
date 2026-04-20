"""Helpers for assembling the runtime startup prompt."""

from __future__ import annotations

import core.config as config


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
        port_list = raw_ports
        lines.append(f"- Network ports: {port_list[0]}-{port_list[-1]}")

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
