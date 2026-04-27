"""Tool management enhancements for multi-agent systems.

This module provides:
- Tool reranking based on context and confidence scoring
- Reject option for declining tool execution with explanations
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Reasons for rejecting a tool call."""

    LOW_CONFIDENCE = auto()
    """Tool selection confidence below threshold."""

    UNSAFE_PARAMETERS = auto()
    """Parameters detected as potentially harmful."""

    DUPLICATE_CALL = auto()
    """Same tool was recently called with same parameters."""

    RESOURCE_CONSTRAINT = auto()
    """System resources insufficient for this operation."""

    POLICY_VIOLATION = auto()
    """Call violates configured policies."""

    CONTEXT_MISMATCH = auto()
    """Tool not appropriate for current context."""

    MANUAL_REJECT = auto()
    """Explicit manual rejection by system/orchestrator."""


@dataclass
class ToolCallScore:
    """Score assigned to a tool call during reranking."""

    tool_name: str
    original_rank: int
    reranked_score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    factors: dict[str, float] = field(default_factory=dict)
    """Breakdown of scoring factors (e.g., relevance, recency, success_rate)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "original_rank": self.original_rank,
            "reranked_score": self.reranked_score,
            "confidence": self.confidence,
            "factors": self.factors,
        }


@dataclass
class RejectionResult:
    """Result of rejecting a tool call."""

    rejected: bool
    reason: RejectionReason | None = None
    explanation: str = ""
    alternative_suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejected": self.rejected,
            "reason": self.reason.name if self.reason else None,
            "explanation": self.explanation,
            "alternative_suggestion": self.alternative_suggestion,
        }


class ToolReranker:
    """Reranks tool calls based on context, history, and confidence.

    Provides intelligent reranking of tool candidates using multiple factors
    including success history, context relevance, and recency.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        success_history_weight: float = 0.4,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.2,
    ) -> None:
        """Initialize the reranker.

        Args:
            confidence_threshold: Minimum confidence to keep a tool call.
            success_history_weight: Weight for historical success rate.
            relevance_weight: Weight for contextual relevance.
            recency_weight: Weight for recent usage patterns.
        """
        self._confidence_threshold = confidence_threshold
        self._success_history_weight = success_history_weight
        self._relevance_weight = relevance_weight
        self._recency_weight = recency_weight

        # History tracking using deque for O(1) operations
        self._success_rates: dict[str, float] = {}
        self._recent_calls: deque[str] = deque(maxlen=100)
        self._max_recent = 100

    def update_success_rate(self, tool_name: str, success: bool) -> None:
        """Update the success rate for a tool.

        Args:
            tool_name: Name of the tool.
            success: Whether the execution was successful.
        """
        if tool_name not in self._success_rates:
            self._success_rates[tool_name] = 0.5  # Default prior

        # Exponential moving average
        alpha = 0.1
        current = self._success_rates[tool_name]
        new_value = 1.0 if success else 0.0
        self._success_rates[tool_name] = (1 - alpha) * current + alpha * new_value

    def record_call(self, tool_name: str) -> None:
        """Record a tool call for recency tracking.
        
        Deque with maxlen automatically handles size limits.
        """
        self._recent_calls.append(tool_name)

    def compute_recency_score(self, tool_name: str) -> float:
        """Compute recency score based on recent call frequency.

        Returns higher scores for tools called recently (assumes useful momentum).
        """
        if not self._recent_calls:
            return 0.5

        count = self._recent_calls.count(tool_name)
        # Normalize by window size
        return min(1.0, count / 10.0)  # Cap at 10 calls

    def rerank_tools(
        self,
        tool_calls: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[ToolCallScore]]:
        """Rerank tool calls based on context and history.

        Args:
            tool_calls: List of tool call dicts from LLM.
            context: Optional context for relevance scoring.

        Returns:
            Tuple of (reranked_tool_calls, scores).
            Tools below confidence threshold are filtered out.
        """
        if not tool_calls:
            return [], []

        scores: list[ToolCallScore] = []

        for idx, call in enumerate(tool_calls):
            tool_name = call.get("name", "unknown")

            # Factor 1: Historical success rate
            success_rate = self._success_rates.get(tool_name, 0.5)

            # Factor 2: Recency score
            recency_score = self.compute_recency_score(tool_name)

            # Factor 3: Contextual relevance (simplified - can be enhanced with embeddings)
            relevance_score = self._compute_relevance(tool_name, context or {})

            # Weighted combination
            final_score = (
                self._success_history_weight * success_rate
                + self._recency_weight * recency_score
                + self._relevance_weight * relevance_score
            )

            # Confidence is based on score magnitude and consistency
            confidence = min(1.0, final_score + 0.2)  # Small boost

            score = ToolCallScore(
                tool_name=tool_name,
                original_rank=idx,
                reranked_score=final_score,
                confidence=confidence,
                factors={
                    "success_rate": success_rate,
                    "recency": recency_score,
                    "relevance": relevance_score,
                },
            )
            scores.append(score)

        # Filter by confidence threshold
        accepted_calls = []
        accepted_scores = []

        for call, score in zip(tool_calls, scores):
            if score.confidence >= self._confidence_threshold:
                accepted_calls.append(call)
                accepted_scores.append(score)
            else:
                logger.info(
                    "Filtered tool call '%s': confidence %.2f < threshold %.2f",
                    score.tool_name,
                    score.confidence,
                    self._confidence_threshold,
                )

        # Sort by reranked score (descending)
        if accepted_calls:
            sorted_pairs = sorted(
                zip(accepted_calls, accepted_scores),
                key=lambda x: x[1].reranked_score,
                reverse=True,
            )
            accepted_calls = [c for c, _ in sorted_pairs]
            accepted_scores = [s for _, s in sorted_pairs]

        return accepted_calls, accepted_scores

    def _compute_relevance(self, tool_name: str, context: dict[str, Any]) -> float:
        """Compute contextual relevance score.

        Searches for keywords in both keys and values of the context dictionary.
        """
        # Basic heuristics based on tool name patterns
        tool_lower = tool_name.lower()

        # Convert context to searchable string (keys + values)
        context_text = " ".join(
            [str(k).lower() + " " + str(v).lower() for k, v in context.items()]
        )

        # Check if context hints match tool capabilities
        if "code" in context_text or "programming" in context_text:
            if any(kw in tool_lower for kw in ["code", "exec", "run", "python"]):
                return 0.9

        if "file" in context_text or "read" in context_text:
            if any(kw in tool_lower for kw in ["read", "load", "file"]):
                return 0.8

        if "network" in context_text or "http" in context_text:
            if any(kw in tool_lower for kw in ["http", "request", "fetch"]):
                return 0.9

        # Default neutral score
        return 0.5


class ToolRejectionHandler:
    """Handles rejection of tool calls with explanations.

    Provides automatic detection of problematic tool calls and generates
    helpful feedback with alternative suggestions.
    """

    def __init__(
        self,
        auto_reject_low_confidence: bool = True,
        confidence_threshold: float = 0.2,
        max_duplicate_window: int = 5,
    ) -> None:
        """Initialize the rejection handler.

        Args:
            auto_reject_low_confidence: Automatically reject low-confidence calls.
            confidence_threshold: Threshold for auto-rejection.
            max_duplicate_window: Number of recent calls to check for duplicates.
        """
        self._auto_reject = auto_reject_low_confidence
        self._confidence_threshold = confidence_threshold
        self._recent_calls: deque[dict[str, Any]] = deque(maxlen=max_duplicate_window)

        # Policy rules (can be extended)
        self._blocked_tools: set[str] = set()
        self._parameter_constraints: dict[str, list[str]] = {}

    def block_tool(self, tool_name: str) -> None:
        """Block a tool from being executed."""
        self._blocked_tools.add(tool_name)
        logger.warning("Tool '%s' has been blocked by policy", tool_name)

    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a previously blocked tool."""
        self._blocked_tools.discard(tool_name)

    def add_parameter_constraint(self, tool_name: str, forbidden_params: list[str]) -> None:
        """Add parameter constraints for a tool."""
        self._parameter_constraints[tool_name] = forbidden_params

    def check_duplicate(self, tool_call: dict[str, Any]) -> bool:
        """Check if this is a duplicate of a recent call.
        
        Deque automatically maintains window size via maxlen.
        """
        for recent in self._recent_calls:
            if (
                recent.get("name") == tool_call.get("name")
                and recent.get("input") == tool_call.get("input")
            ):
                return True
        return False

    def check_parameters(self, tool_call: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if parameters violate any constraints.

        Returns:
            Tuple of (is_safe, violation_description).
        """
        tool_name = tool_call.get("name", "")
        params = tool_call.get("input", {})

        if tool_name in self._parameter_constraints:
            forbidden = self._parameter_constraints[tool_name]
            for param in params:
                if param in forbidden:
                    return False, f"Parameter '{param}' is forbidden for tool '{tool_name}'"

        return True, None

    def reject_tool_call(
        self,
        tool_call: dict[str, Any],
        reason: RejectionReason | None = None,
        custom_explanation: str | None = None,
        confidence: float | None = None,
    ) -> RejectionResult:
        """Reject a tool call with explanation.

        Args:
            tool_call: The tool call to potentially reject.
            reason: Reason for rejection (auto-detected if None).
            custom_explanation: Custom explanation override.
            confidence: Confidence score for auto-rejection logic.

        Returns:
            RejectionResult indicating whether and why the call was rejected.
        """
        tool_name = tool_call.get("name", "unknown")

        # Auto-detect rejection reasons
        if reason is None:
            # Check if tool is blocked
            if tool_name in self._blocked_tools:
                reason = RejectionReason.POLICY_VIOLATION
                custom_explanation = f"Tool '{tool_name}' is blocked by administrator policy."

            # Check for duplicates
            elif self.check_duplicate(tool_call):
                reason = RejectionReason.DUPLICATE_CALL
                custom_explanation = (
                    f"This is a duplicate call to '{tool_name}' with identical parameters. "
                    "Consider using the previous result or modifying the input."
                )

            # Check parameter constraints
            is_safe, violation = self.check_parameters(tool_call)
            if not is_safe:
                reason = RejectionReason.UNSAFE_PARAMETERS
                custom_explanation = violation

            # Check confidence
            elif (
                self._auto_reject
                and confidence is not None
                and confidence < self._confidence_threshold
            ):
                reason = RejectionReason.LOW_CONFIDENCE
                custom_explanation = (
                    f"Tool selection confidence ({confidence:.2f}) is below "
                    f"threshold ({self._confidence_threshold}). "
                    "Consider reformulating the request or choosing a different approach."
                )

        # If no rejection reason, accept the call
        if reason is None:
            # Record for duplicate tracking (deque auto-manages size)
            self._recent_calls.append(tool_call.copy())

            return RejectionResult(rejected=False)

        # Build explanation
        explanation = custom_explanation or f"Tool call rejected: {reason.name}"

        # Generate alternative suggestion
        alternative = self._suggest_alternative(tool_call, reason)

        logger.info(
            "Rejected tool call '%s': %s",
            tool_name,
            explanation,
        )

        return RejectionResult(
            rejected=True,
            reason=reason,
            explanation=explanation,
            alternative_suggestion=alternative,
        )

    def _suggest_alternative(
        self,
        tool_call: dict[str, Any],
        reason: RejectionReason,
    ) -> str | None:
        """Suggest an alternative action when rejecting."""
        tool_name = tool_call.get("name", "")

        if reason == RejectionReason.DUPLICATE_CALL:
            return "Use the result from the previous identical call instead."

        elif reason == RejectionReason.LOW_CONFIDENCE:
            return (
                "Try breaking down your request into smaller steps, "
                "or explicitly specify which tool should be used."
            )

        elif reason == RejectionReason.UNSAFE_PARAMETERS:
            return "Review the tool documentation for allowed parameters."

        elif reason == RejectionReason.POLICY_VIOLATION:
            return f"Tool '{tool_name}' is not available. Consider alternative approaches."

        return None

    def process_with_rejection(
        self,
        tool_calls: list[dict[str, Any]],
        scores: list[ToolCallScore] | None = None,
    ) -> tuple[list[dict[str, Any]], list[RejectionResult]]:
        """Process tool calls through rejection logic.

        Args:
            tool_calls: List of tool calls to process.
            scores: Optional scores from reranker for confidence-based rejection.

        Returns:
            Tuple of (accepted_calls, rejection_results).
        """
        accepted = []
        results = []

        for idx, call in enumerate(tool_calls):
            confidence = scores[idx].confidence if scores and idx < len(scores) else None

            result = self.reject_tool_call(call, confidence=confidence)
            results.append(result)

            if not result.rejected:
                accepted.append(call)

        return accepted, results
