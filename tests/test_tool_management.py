"""Tests for core.tool_management - reranking and rejection."""

import pytest

from core.tool_management import (
    RejectionReason,
    RejectionResult,
    ToolCallScore,
    ToolRejectionHandler,
    ToolReranker,
)


# ---------------------------------------------------------------------------
# ToolCallScore tests
# ---------------------------------------------------------------------------


def test_tool_call_score_to_dict() -> None:
    score = ToolCallScore(
        tool_name="test_tool",
        original_rank=0,
        reranked_score=0.85,
        confidence=0.75,
        factors={"success_rate": 0.9, "relevance": 0.8},
    )
    result = score.to_dict()
    assert result["tool_name"] == "test_tool"
    assert result["reranked_score"] == 0.85
    assert result["confidence"] == 0.75
    assert "success_rate" in result["factors"]


# ---------------------------------------------------------------------------
# RejectionResult tests
# ---------------------------------------------------------------------------


def test_rejection_result_accepted() -> None:
    result = RejectionResult(rejected=False)
    d = result.to_dict()
    assert d["rejected"] is False
    assert d["reason"] is None


def test_rejection_result_rejected() -> None:
    result = RejectionResult(
        rejected=True,
        reason=RejectionReason.LOW_CONFIDENCE,
        explanation="Confidence too low",
        alternative_suggestion="Try again",
    )
    d = result.to_dict()
    assert d["rejected"] is True
    assert d["reason"] == "LOW_CONFIDENCE"
    assert d["explanation"] == "Confidence too low"
    assert d["alternative_suggestion"] == "Try again"


# ---------------------------------------------------------------------------
# ToolReranker tests
# ---------------------------------------------------------------------------


def test_reranker_empty_calls() -> None:
    reranker = ToolReranker()
    calls, scores = reranker.rerank_tools([])
    assert calls == []
    assert scores == []


def test_reranker_single_tool() -> None:
    reranker = ToolReranker(confidence_threshold=0.1)
    tool_calls = [{"name": "run_plugin", "input": {"name": "test"}}]
    
    calls, scores = reranker.rerank_tools(tool_calls)
    
    assert len(calls) == 1
    assert len(scores) == 1
    assert scores[0].tool_name == "run_plugin"
    assert 0.0 <= scores[0].reranked_score <= 1.0
    assert 0.0 <= scores[0].confidence <= 1.0


def test_reranker_filters_low_confidence() -> None:
    reranker = ToolReranker(confidence_threshold=0.9)  # Very high threshold
    tool_calls = [{"name": "unknown_tool", "input": {}}]
    
    calls, scores = reranker.rerank_tools(tool_calls)
    
    # Should be filtered out due to low confidence
    assert len(calls) == 0


def test_reranker_updates_success_rate() -> None:
    reranker = ToolReranker(confidence_threshold=0.1)
    
    # Simulate successful calls
    for _ in range(10):
        reranker.update_success_rate("good_tool", True)
    
    # Simulate failed calls
    for _ in range(10):
        reranker.update_success_rate("bad_tool", False)
    
    # good_tool should have higher success rate
    assert reranker._success_rates["good_tool"] > 0.7
    assert reranker._success_rates["bad_tool"] < 0.3


def test_reranker_sorts_by_score() -> None:
    reranker = ToolReranker(confidence_threshold=0.1)
    
    # Create tools with different names to get different relevance scores
    tool_calls = [
        {"name": "http_request", "input": {}},
        {"name": "run_code", "input": {}},
        {"name": "read_file", "input": {}},
    ]
    
    context = {"code": "programming task"}
    calls, scores = reranker.rerank_tools(tool_calls, context=context)
    
    # run_code should be ranked highest due to code context
    assert len(calls) == 3
    assert scores[0].tool_name == "run_code"


def test_reranker_records_calls_for_recency() -> None:
    reranker = ToolReranker()
    
    for _ in range(150):  # More than max_recent (100)
        reranker.record_call("frequent_tool")
    
    # Should cap at max_recent
    assert len(reranker._recent_calls) == reranker._max_recent
    
    # Recency score should be high
    score = reranker.compute_recency_score("frequent_tool")
    assert score > 0.5


def test_reranker_context_relevance_code() -> None:
    reranker = ToolReranker()
    
    # Code context should boost code-related tools
    score = reranker._compute_relevance("run_python", {"code": "test"})
    assert score > 0.7
    
    # Non-code tool should get neutral score
    score = reranker._compute_relevance("http_get", {"code": "test"})
    assert abs(score - 0.5) < 0.1


# ---------------------------------------------------------------------------
# ToolRejectionHandler tests
# ---------------------------------------------------------------------------


def test_rejection_handler_accepts_valid_call() -> None:
    handler = ToolRejectionHandler()
    tool_call = {"name": "run_plugin", "input": {"name": "test"}}
    
    result = handler.reject_tool_call(tool_call)
    
    assert result.rejected is False
    assert result.reason is None


def test_rejection_handler_blocks_tool() -> None:
    handler = ToolRejectionHandler()
    handler.block_tool("dangerous_tool")
    
    tool_call = {"name": "dangerous_tool", "input": {}}
    result = handler.reject_tool_call(tool_call)
    
    assert result.rejected is True
    assert result.reason == RejectionReason.POLICY_VIOLATION
    assert "blocked" in result.explanation.lower()


def test_rejection_handler_unblocks_tool() -> None:
    handler = ToolRejectionHandler()
    handler.block_tool("temp_blocked")
    handler.unblock_tool("temp_blocked")
    
    tool_call = {"name": "temp_blocked", "input": {}}
    result = handler.reject_tool_call(tool_call)
    
    assert result.rejected is False


def test_rejection_handler_detects_duplicate() -> None:
    handler = ToolRejectionHandler(max_duplicate_window=5)
    
    tool_call = {"name": "run_plugin", "input": {"name": "test"}}
    
    # First call should be accepted
    result1 = handler.reject_tool_call(tool_call)
    assert result1.rejected is False
    
    # Duplicate should be rejected
    result2 = handler.reject_tool_call(tool_call)
    assert result2.rejected is True
    assert result2.reason == RejectionReason.DUPLICATE_CALL


def test_rejection_handler_parameter_constraint() -> None:
    handler = ToolRejectionHandler()
    handler.add_parameter_constraint("run_plugin", ["dangerous_param"])
    
    tool_call = {"name": "run_plugin", "input": {"dangerous_param": "value"}}
    result = handler.reject_tool_call(tool_call)
    
    assert result.rejected is True
    assert result.reason == RejectionReason.UNSAFE_PARAMETERS
    assert "dangerous_param" in result.explanation


def test_rejection_handler_low_confidence_auto_reject() -> None:
    handler = ToolRejectionHandler(
        auto_reject_low_confidence=True,
        confidence_threshold=0.5,
    )
    
    tool_call = {"name": "test_tool", "input": {}}
    result = handler.reject_tool_call(tool_call, confidence=0.3)
    
    assert result.rejected is True
    assert result.reason == RejectionReason.LOW_CONFIDENCE


def test_rejection_handler_low_confidence_allowed() -> None:
    handler = ToolRejectionHandler(
        auto_reject_low_confidence=False,  # Disabled
        confidence_threshold=0.5,
    )
    
    tool_call = {"name": "test_tool", "input": {}}
    result = handler.reject_tool_call(tool_call, confidence=0.3)
    
    # Should be accepted since auto-reject is disabled
    assert result.rejected is False


def test_rejection_handler_alternative_suggestions() -> None:
    handler = ToolRejectionHandler()
    
    # Test duplicate suggestion
    handler.block_tool("dup_tool")
    result = handler.reject_tool_call({"name": "dup_tool", "input": {}})
    assert result.alternative_suggestion is not None
    
    # Test low confidence suggestion
    result = handler.reject_tool_call(
        {"name": "test", "input": {}},
        reason=RejectionReason.LOW_CONFIDENCE,
    )
    assert "breaking down" in result.alternative_suggestion.lower()


def test_rejection_handler_process_with_rejection() -> None:
    handler = ToolRejectionHandler(auto_reject_low_confidence=False)
    
    tool_calls = [
        {"name": "tool_a", "input": {}},
        {"name": "tool_b", "input": {}},
        {"name": "tool_c", "input": {}},
    ]
    
    scores = [
        ToolCallScore("tool_a", 0, 0.9, 0.8),
        ToolCallScore("tool_b", 1, 0.5, 0.4),
        ToolCallScore("tool_c", 2, 0.3, 0.2),
    ]
    
    # Block tool_b
    handler.block_tool("tool_b")
    
    accepted, results = handler.process_with_rejection(tool_calls, scores)
    
    # tool_a should be accepted, tool_b blocked, tool_c accepted (no auto-reject)
    assert len(accepted) == 2
    assert accepted[0]["name"] == "tool_a"
    assert accepted[1]["name"] == "tool_c"
    
    assert results[0].rejected is False
    assert results[1].rejected is True
    assert results[2].rejected is False


def test_rejection_handler_custom_reason() -> None:
    handler = ToolRejectionHandler()
    
    tool_call = {"name": "test_tool", "input": {}}
    result = handler.reject_tool_call(
        tool_call,
        reason=RejectionReason.MANUAL_REJECT,
        custom_explanation="Custom rejection message",
    )
    
    assert result.rejected is True
    assert result.reason == RejectionReason.MANUAL_REJECT
    assert result.explanation == "Custom rejection message"
