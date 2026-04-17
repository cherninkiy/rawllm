from __future__ import annotations

import pytest

import core


def test_core_lazy_exports_resolve_symbols() -> None:
    assert core.AnthropicClient is not None
    assert core.PluginManager is not None
    assert core.ToolExecutor is not None
    assert core.TAORLoop is not None


def test_core_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = core.NotExistingSymbol
