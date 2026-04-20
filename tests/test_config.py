from pathlib import Path

import pytest

import core.config as config


def test_parse_ports_supports_ranges_and_lists() -> None:
    assert config._parse_ports("8000-8002,8080") == [8000, 8001, 8002, 8080]


def test_parse_ports_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid port value"):
        config._parse_ports("abc")


def test_parse_ports_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="Invalid port value"):
        config._parse_ports("8002-8000")


def test_parse_services_parses_name_uri_pairs() -> None:
    assert config._parse_services("postgres:postgresql://db,redis:redis://cache") == {
        "postgres": "postgresql://db",
        "redis": "redis://cache",
    }


def test_parse_services_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid service definition"):
        config._parse_services("postgres")


def test_configure_runtime_resources_updates_globals() -> None:
    original_ports = config.AVAILABLE_PORTS
    original_workspace = config.WORKSPACE_PATH
    original_services = config.AVAILABLE_SERVICES
    try:
        config.configure_runtime_resources(
            ports=[9000],
            workspace_path=Path("/tmp/workspace"),
            services={"postgres": "postgresql://db"},
        )
        assert config.AVAILABLE_PORTS == [9000]
        assert config.WORKSPACE_PATH == Path("/tmp/workspace")
        assert config.AVAILABLE_SERVICES == {"postgres": "postgresql://db"}
    finally:
        config.configure_runtime_resources(
            ports=original_ports,
            workspace_path=original_workspace,
            services=original_services,
        )
