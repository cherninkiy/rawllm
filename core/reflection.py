"""Tool Reflection Cycle for automatic error analysis and correction.

This module provides:
- Error analysis for tool execution failures
- Automatic generation of corrected tool calls
- Reflection loop for iterative improvement
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of tool execution errors."""

    SYNTAX_ERROR = auto()
    """Code syntax or parsing error."""

    RUNTIME_ERROR = auto()
    """Runtime exception during execution."""

    TIMEOUT_ERROR = auto()
    """Execution exceeded time limit."""

    RESOURCE_ERROR = auto()
    """Insufficient resources (memory, disk, etc.)."""

    PERMISSION_ERROR = auto()
    """Access denied or permission issue."""

    NOT_FOUND_ERROR = auto()
    """Plugin, file, or resource not found."""

    VALIDATION_ERROR = auto()
    """Input validation failed."""

    NETWORK_ERROR = auto()
    """Network-related failure."""

    UNKNOWN_ERROR = auto()
    """Unclassified error."""


@dataclass
class ErrorAnalysis:
    """Result of analyzing a tool execution error."""

    error_category: ErrorCategory
    error_message: str
    error_type: str
    stack_trace: str | None = None
    root_cause: str = ""
    confidence: float = 0.5
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis to dictionary representation."""
        return {
            "error_category": self.error_category.name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


@dataclass
class CorrectionResult:
    """Result of generating a correction for a failed tool call."""

    success: bool
    corrected_call: dict[str, Any] | None = None
    explanation: str = ""
    confidence: float = 0.5
    alternative_approaches: list[str] = field(default_factory=list)
    requires_human_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert correction result to dictionary representation."""
        return {
            "success": self.success,
            "corrected_call": self.corrected_call,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "alternative_approaches": self.alternative_approaches,
            "requires_human_review": self.requires_human_review,
        }


class ErrorAnalyzer:
    """Analyzes tool execution errors to determine root cause and category."""

    def __init__(self) -> None:
        """Initialize the error analyzer."""
        # Error pattern mappings for categorization
        self._error_patterns: dict[ErrorCategory, list[str]] = {
            ErrorCategory.SYNTAX_ERROR: [
                "SyntaxError",
                "invalid syntax",
                "unexpected indent",
                "unexpected EOF",
                "missing parentheses",
            ],
            ErrorCategory.RUNTIME_ERROR: [
                "TypeError",
                "ValueError",
                "AttributeError",
                "KeyError",
                "IndexError",
                "RuntimeError",
                "AssertionError",
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "TimeoutError",
                "timeout",
                "timed out",
                "deadline exceeded",
            ],
            ErrorCategory.RESOURCE_ERROR: [
                "MemoryError",
                "ResourceWarning",
                "out of memory",
                "disk full",
                "no space left",
            ],
            ErrorCategory.PERMISSION_ERROR: [
                "PermissionError",
                "AccessDenied",
                "access denied",
                "permission denied",
                "not permitted",
            ],
            ErrorCategory.NOT_FOUND_ERROR: [
                "FileNotFoundError",
                "ModuleNotFoundError",
                "NotFound",
                "does not exist",
                "not found",
            ],
            ErrorCategory.VALIDATION_ERROR: [
                "ValidationError",
                "InvalidInput",
                "invalid value",
                "validation failed",
                "constraint violation",
            ],
            ErrorCategory.NETWORK_ERROR: [
                "ConnectionError",
                "ConnectionRefusedError",
                "ConnectionResetError",
                "BrokenPipeError",
                "network",
                "connection",
            ],
        }

    def analyze_error(
        self,
        tool_call: dict[str, Any],
        result: dict[str, Any],
        traceback_str: str | None = None,
    ) -> ErrorAnalysis:
        """Analyze an error from tool execution.

        Args:
            tool_call: The original tool call that failed.
            result: The result dict containing error information.
            traceback_str: Optional full traceback string.

        Returns:
            ErrorAnalysis with categorized error and suggestions.
        """
        error_message = result.get("error", result.get("error_message", "Unknown error"))
        error_type = result.get("error_type", "UnknownError")

        # Combine error message and traceback for analysis
        full_text = f"{error_type}: {error_message}"
        if traceback_str:
            full_text += f"\n{traceback_str}"

        full_text_lower = full_text.lower()

        # Categorize error based on patterns
        category = ErrorCategory.UNKNOWN_ERROR
        best_match_count = 0

        for err_category, patterns in self._error_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern.lower() in full_text_lower)
            if match_count > best_match_count:
                best_match_count = match_count
                category = err_category

        # Determine confidence based on pattern matches
        confidence = min(1.0, best_match_count / 3.0)  # Normalize to 0-1

        # Generate root cause hypothesis
        root_cause = self._hypothesize_root_cause(category, error_message, tool_call)

        # Generate suggestions
        suggestions = self._generate_suggestions(category, error_message, tool_call)

        return ErrorAnalysis(
            error_category=category,
            error_message=error_message,
            error_type=error_type,
            stack_trace=traceback_str,
            root_cause=root_cause,
            confidence=confidence,
            suggestions=suggestions,
            metadata={
                "tool_name": tool_call.get("name", "unknown"),
                "tool_input": tool_call.get("input", {}),
            },
        )

    def _hypothesize_root_cause(
        self,
        category: ErrorCategory,
        error_message: str,
        tool_call: dict[str, Any],
    ) -> str:
        """Generate a hypothesis about the root cause of the error."""
        tool_name = tool_call.get("name", "unknown")
        tool_input = tool_call.get("input", {})

        if category == ErrorCategory.SYNTAX_ERROR:
            return "The code contains syntax errors that prevent parsing."

        elif category == ErrorCategory.RUNTIME_ERROR:
            if "NoneType" in error_message:
                return "Attempting to access attributes or methods on a None value."
            elif "key" in error_message.lower():
                return "Accessing a dictionary key that doesn't exist."
            elif "index" in error_message.lower():
                return "Accessing a sequence index that's out of range."
            else:
                return "An exception occurred during code execution."

        elif category == ErrorCategory.NOT_FOUND_ERROR:
            if "ModuleNotFoundError" in error_message:
                return "Required module is not installed or not in the allow-list."
            elif "FileNotFoundError" in error_message:
                return "Referenced file does not exist at the specified path."
            else:
                return f"Plugin or resource '{tool_name}' was not found."

        elif category == ErrorCategory.PERMISSION_ERROR:
            return "Operation requires permissions that are not granted."

        elif category == ErrorCategory.TIMEOUT_ERROR:
            return "Execution took too long and exceeded the time limit."

        elif category == ErrorCategory.VALIDATION_ERROR:
            return "Input parameters do not meet validation requirements."

        return "Unable to determine specific root cause."

    def _generate_suggestions(
        self,
        category: ErrorCategory,
        error_message: str,
        tool_call: dict[str, Any],
    ) -> list[str]:
        """Generate actionable suggestions for fixing the error."""
        suggestions = []

        if category == ErrorCategory.SYNTAX_ERROR:
            suggestions.extend([
                "Check for missing colons, parentheses, or quotes.",
                "Verify proper indentation (Python uses 4 spaces).",
                "Use a linter or IDE to identify syntax issues.",
            ])

        elif category == ErrorCategory.RUNTIME_ERROR:
            suggestions.extend([
                "Add error handling with try-except blocks.",
                "Validate inputs before processing.",
                "Check for None values before accessing attributes.",
            ])

        elif category == ErrorCategory.NOT_FOUND_ERROR:
            suggestions.extend([
                "Verify the plugin name is spelled correctly.",
                "Ensure the plugin has been loaded successfully.",
                "Check if required dependencies are installed.",
            ])

        elif category == ErrorCategory.PERMISSION_ERROR:
            suggestions.extend([
                "Review file or resource permissions.",
                "Use appropriate authentication if required.",
                "Consider alternative approaches that don't require elevated permissions.",
            ])

        elif category == ErrorCategory.TIMEOUT_ERROR:
            suggestions.extend([
                "Optimize the code for better performance.",
                "Break the task into smaller chunks.",
                "Consider using asynchronous operations.",
            ])

        elif category == ErrorCategory.VALIDATION_ERROR:
            suggestions.extend([
                "Review the expected input format and types.",
                "Add input validation before calling the tool.",
                "Check documentation for parameter requirements.",
            ])

        return suggestions

    def categorize_error(self, error_type: str, error_message: str) -> ErrorCategory:
        """Quickly categorize an error without full analysis.

        Args:
            error_type: The exception type name.
            error_message: The error message.

        Returns:
            The ErrorCategory for this error.
        """
        full_text = f"{error_type}: {error_message}".lower()

        for category, patterns in self._error_patterns.items():
            if any(pattern.lower() in full_text for pattern in patterns):
                return category

        return ErrorCategory.UNKNOWN_ERROR


class CorrectionGenerator:
    """Generates corrected tool calls based on error analysis."""

    def __init__(self) -> None:
        """Initialize the correction generator."""
        self._analyzer = ErrorAnalyzer()

    def generate_correction(
        self,
        error_analysis: ErrorAnalysis,
        original_call: dict[str, Any],
    ) -> CorrectionResult:
        """Generate a corrected version of a failed tool call.

        Args:
            error_analysis: Analysis of what went wrong.
            original_call: The original tool call that failed.

        Returns:
            CorrectionResult with corrected call or explanation.
        """
        category = error_analysis.error_category
        tool_name = original_call.get("name", "")
        tool_input = original_call.get("input", {})

        # Try to generate corrections based on error category
        if category == ErrorCategory.SYNTAX_ERROR:
            return self._handle_syntax_error(original_call, error_analysis)

        elif category == ErrorCategory.RUNTIME_ERROR:
            return self._handle_runtime_error(original_call, error_analysis)

        elif category == ErrorCategory.NOT_FOUND_ERROR:
            return self._handle_not_found_error(original_call, error_analysis)

        elif category == ErrorCategory.VALIDATION_ERROR:
            return self._handle_validation_error(original_call, error_analysis)

        elif category == ErrorCategory.PERMISSION_ERROR:
            return self._handle_permission_error(original_call, error_analysis)

        else:
            # For unknown or complex errors, suggest human review
            return CorrectionResult(
                success=False,
                explanation=f"Unable to automatically correct {category.name}. "
                f"Root cause: {error_analysis.root_cause}",
                confidence=0.3,
                requires_human_review=True,
                alternative_approaches=error_analysis.suggestions,
            )

    def _handle_syntax_error(
        self,
        original_call: dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> CorrectionResult:
        """Handle syntax errors - typically need LLM regeneration."""
        return CorrectionResult(
            success=False,
            explanation="Syntax errors require code regeneration. "
            "Please review the code and fix syntax issues.",
            confidence=0.4,
            requires_human_review=False,
            alternative_approaches=[
                "Regenerate the code with proper syntax.",
                "Use an IDE or linter to identify syntax issues.",
                *error_analysis.suggestions,
            ],
        )

    def _handle_runtime_error(
        self,
        original_call: dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> CorrectionResult:
        """Handle runtime errors - may be fixable with input adjustments."""
        tool_input = original_call.get("input", {})

        # Suggest adding error handling
        corrected_input = tool_input.copy()

        # If it's a code execution, suggest wrapping in try-except
        if "code" in tool_input:
            corrected_input["code"] = self._wrap_with_error_handling(tool_input["code"])

        return CorrectionResult(
            success=True,
            corrected_call={
                "name": original_call.get("name"),
                "input": corrected_input,
            },
            explanation="Added error handling to catch runtime exceptions.",
            confidence=0.6,
            alternative_approaches=error_analysis.suggestions,
        )

    def _handle_not_found_error(
        self,
        original_call: dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> CorrectionResult:
        """Handle not found errors - check names and availability."""
        tool_name = original_call.get("name", "")

        # Common fixes for not found errors
        suggestions = [
            f"Verify that plugin '{tool_name}' exists and is loaded.",
            "Check for typos in the plugin name.",
            "Load the plugin before executing.",
        ]

        return CorrectionResult(
            success=False,
            explanation=f"Plugin or resource '{tool_name}' not found.",
            confidence=0.5,
            requires_human_review=False,
            alternative_approaches=suggestions + error_analysis.suggestions,
        )

    def _handle_validation_error(
        self,
        original_call: dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> CorrectionResult:
        """Handle validation errors - adjust input parameters."""
        tool_input = original_call.get("input", {})

        # Generic validation fix suggestions
        return CorrectionResult(
            success=False,
            explanation="Input validation failed. Review parameter types and constraints.",
            confidence=0.5,
            requires_human_review=False,
            alternative_approaches=[
                "Validate all input parameters match expected types.",
                "Check for required vs optional parameters.",
                "Review parameter value ranges and formats.",
                *error_analysis.suggestions,
            ],
        )

    def _handle_permission_error(
        self,
        original_call: dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> CorrectionResult:
        """Handle permission errors - suggest alternatives."""
        return CorrectionResult(
            success=False,
            explanation="Operation requires permissions that are not available.",
            confidence=0.4,
            requires_human_review=True,
            alternative_approaches=[
                "Request necessary permissions from administrator.",
                "Find alternative approach that doesn't require elevated permissions.",
                "Use sandboxed or restricted version of the operation.",
                *error_analysis.suggestions,
            ],
        )

    def _wrap_with_error_handling(self, code: str) -> str:
        """Wrap code in a try-except block for better error handling."""
        wrapped = f"""try:
{code}
except Exception as e:
    print(f"Error during execution: {{e}}")
    import traceback
    traceback.print_exc()
    raise
"""
        return wrapped

    def validate_correction(self, proposed_call: dict[str, Any]) -> bool:
        """Validate that a proposed correction is reasonable.

        Args:
            proposed_call: The corrected tool call to validate.

        Returns:
            True if the correction appears valid.
        """
        # Basic validation checks
        if not proposed_call:
            return False

        if "name" not in proposed_call:
            return False

        if "input" not in proposed_call:
            return False

        if not isinstance(proposed_call["input"], dict):
            return False

        return True


class ReflectionLoop:
    """Manages the reflection cycle for continuous improvement.

    Tracks errors, generates corrections, and learns from outcomes.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the reflection loop.

        Args:
            max_history: Maximum number of reflection events to keep in memory.
        """
        self._analyzer = ErrorAnalyzer()
        self._correction_generator = CorrectionGenerator()
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}

    def run_reflection_cycle(
        self,
        tool_call: dict[str, Any],
        result: dict[str, Any],
        traceback_str: str | None = None,
    ) -> dict[str, Any]:
        """Run a complete reflection cycle on a failed tool call.

        Args:
            tool_call: The tool call that failed.
            result: The error result.
            traceback_str: Optional full traceback.

        Returns:
            Dict with analysis, correction, and recommendations.
        """
        tool_name = tool_call.get("name", "unknown")

        # Step 1: Analyze the error
        error_analysis = self._analyzer.analyze_error(tool_call, result, traceback_str)

        logger.info(
            "Reflection: Analyzed error for '%s' - Category: %s, Confidence: %.2f",
            tool_name,
            error_analysis.error_category.name,
            error_analysis.confidence,
        )

        # Step 2: Generate correction
        correction = self._correction_generator.generate_correction(
            error_analysis, tool_call
        )

        # Step 3: Record the reflection event
        reflection_event = {
            "tool_call": tool_call,
            "error_analysis": error_analysis.to_dict(),
            "correction": correction.to_dict(),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        self._record_reflection(reflection_event)

        # Step 4: Update success/failure counts
        if correction.success:
            self._success_counts[tool_name] = self._success_counts.get(tool_name, 0) + 1
        else:
            self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1

        return {
            "analysis": error_analysis.to_dict(),
            "correction": correction.to_dict(),
            "recommendation": self._get_recommendation(tool_name, error_analysis, correction),
        }

    def _record_reflection(self, event: dict[str, Any]) -> None:
        """Record a reflection event in history."""
        self._history.append(event)

        # Trim history if needed
        while len(self._history) > self._max_history:
            self._history.pop(0)

    def log_reflection_event(self, reflection_data: dict[str, Any]) -> None:
        """Log a reflection event for later analysis.

        Args:
            reflection_data: Data about the reflection event.
        """
        # This could be extended to write to metrics or external storage
        logger.debug("Logged reflection event: %s", reflection_data.get("tool_name", "unknown"))

    def _get_recommendation(
        self,
        tool_name: str,
        error_analysis: ErrorAnalysis,
        correction: CorrectionResult,
    ) -> str:
        """Generate a recommendation based on the reflection."""
        if correction.success:
            return (
                f"Apply the suggested correction for '{tool_name}'. "
                f"Confidence: {correction.confidence:.0%}"
            )
        elif correction.requires_human_review:
            return (
                f"Human review recommended for '{tool_name}'. "
                f"Issue: {error_analysis.root_cause}"
            )
        else:
            return (
                f"Consider alternative approaches for '{tool_name}': "
                f"{', '.join(correction.alternative_approaches[:2])}"
            )

    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get the historical success rate for a tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Success rate between 0.0 and 1.0.
        """
        successes = self._success_counts.get(tool_name, 0)
        failures = self._failure_counts.get(tool_name, 0)
        total = successes + failures

        if total == 0:
            return 0.5  # Default prior

        return successes / total

    def get_reflection_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent reflection events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of reflection event dicts.
        """
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear the reflection history."""
        self._history.clear()
        self._success_counts.clear()
        self._failure_counts.clear()


# Singleton instance for easy access
_reflection_loop_instance: ReflectionLoop | None = None


def get_reflection_loop() -> ReflectionLoop:
    """Get the singleton reflection loop instance.

    Returns:
        The ReflectionLoop instance.
    """
    global _reflection_loop_instance
    if _reflection_loop_instance is None:
        _reflection_loop_instance = ReflectionLoop()
    return _reflection_loop_instance
