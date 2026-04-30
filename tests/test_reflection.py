"""Tests for the Tool Reflection Cycle (Sprint 2 - Task 1.2.1)."""

import pytest
from core.reflection import (
    CorrectionGenerator,
    CorrectionResult,
    ErrorAnalyzer,
    ErrorCategory,
    ErrorAnalysis,
    ReflectionLoop,
    get_reflection_loop,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_error_categories_exist(self):
        """Test that all expected error categories exist."""
        assert ErrorCategory.SYNTAX_ERROR is not None
        assert ErrorCategory.RUNTIME_ERROR is not None
        assert ErrorCategory.TIMEOUT_ERROR is not None
        assert ErrorCategory.RESOURCE_ERROR is not None
        assert ErrorCategory.PERMISSION_ERROR is not None
        assert ErrorCategory.NOT_FOUND_ERROR is not None
        assert ErrorCategory.VALIDATION_ERROR is not None
        assert ErrorCategory.NETWORK_ERROR is not None
        assert ErrorCategory.UNKNOWN_ERROR is not None


class TestErrorAnalysis:
    """Tests for ErrorAnalysis dataclass."""

    def test_error_analysis_creation(self):
        """Test creating an ErrorAnalysis instance."""
        analysis = ErrorAnalysis(
            error_category=ErrorCategory.RUNTIME_ERROR,
            error_message="Test error",
            error_type="TestError",
            root_cause="Test root cause",
            confidence=0.8,
            suggestions=["Fix this", "Fix that"],
        )

        assert analysis.error_category == ErrorCategory.RUNTIME_ERROR
        assert analysis.error_message == "Test error"
        assert len(analysis.suggestions) == 2

    def test_error_analysis_to_dict(self):
        """Test converting ErrorAnalysis to dictionary."""
        analysis = ErrorAnalysis(
            error_category=ErrorCategory.SYNTAX_ERROR,
            error_message="Syntax issue",
            error_type="SyntaxError",
            stack_trace="Traceback...",
            confidence=0.9,
        )

        data = analysis.to_dict()
        assert data["error_category"] == "SYNTAX_ERROR"
        assert data["error_message"] == "Syntax issue"
        assert data["error_type"] == "SyntaxError"
        assert data["stack_trace"] == "Traceback..."
        assert data["confidence"] == 0.9


class TestCorrectionResult:
    """Tests for CorrectionResult dataclass."""

    def test_correction_result_success(self):
        """Test successful correction result."""
        result = CorrectionResult(
            success=True,
            corrected_call={"name": "test", "input": {"x": 1}},
            explanation="Fixed the issue",
            confidence=0.85,
        )

        assert result.success is True
        assert result.corrected_call is not None
        assert result.requires_human_review is False

    def test_correction_result_failure(self):
        """Test failed correction result."""
        result = CorrectionResult(
            success=False,
            explanation="Cannot fix automatically",
            requires_human_review=True,
            alternative_approaches=["Try manual fix", "Ask for help"],
        )

        assert result.success is False
        assert result.corrected_call is None
        assert result.requires_human_review is True
        assert len(result.alternative_approaches) == 2

    def test_correction_result_to_dict(self):
        """Test converting CorrectionResult to dictionary."""
        result = CorrectionResult(
            success=True,
            corrected_call={"name": "test"},
            explanation="Success",
            confidence=0.7,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["corrected_call"] == {"name": "test"}
        assert data["explanation"] == "Success"


class TestErrorAnalyzer:
    """Tests for ErrorAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test ErrorAnalyzer initialization."""
        analyzer = ErrorAnalyzer()
        assert analyzer._error_patterns is not None
        assert len(analyzer._error_patterns) > 0

    def test_analyze_runtime_error(self):
        """Test analyzing a runtime error (TypeError)."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "run_code", "input": {"code": "x = None + 1"}}
        result = {
            "error": "unsupported operand type(s)",
            "error_type": "TypeError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.RUNTIME_ERROR
        assert analysis.confidence > 0
        assert len(analysis.suggestions) > 0

    def test_analyze_syntax_error(self):
        """Test analyzing a syntax error."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "run_code", "input": {"code": "if True print('hi')"}}
        result = {
            "error": "invalid syntax",
            "error_type": "SyntaxError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.SYNTAX_ERROR
        assert "syntax" in analysis.error_message.lower()

    def test_analyze_not_found_error(self):
        """Test analyzing a file not found error."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "read_file", "input": {"path": "/missing.txt"}}
        result = {
            "error": "[Errno 2] No such file or directory",
            "error_type": "FileNotFoundError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.NOT_FOUND_ERROR

    def test_analyze_permission_error(self):
        """Test analyzing a permission error."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "write_file", "input": {"path": "/root/test.txt"}}
        result = {
            "error": "[Errno 13] Permission denied",
            "error_type": "PermissionError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.PERMISSION_ERROR

    def test_analyze_timeout_error(self):
        """Test analyzing a timeout error."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "long_running_task", "input": {}}
        result = {
            "error": "Execution timed out after 30 seconds",
            "error_type": "TimeoutError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.TIMEOUT_ERROR

    def test_analyze_with_traceback(self):
        """Test analyzing error with full traceback."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "test", "input": {}}
        result = {"error": "KeyError: 'missing'", "error_type": "KeyError"}
        traceback_str = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    data['missing']
KeyError: 'missing'"""

        analysis = analyzer.analyze_error(tool_call, result, traceback_str)

        assert analysis.stack_trace is not None
        assert "KeyError" in analysis.stack_trace

    def test_categorize_error_quick(self):
        """Test quick error categorization without full analysis."""
        analyzer = ErrorAnalyzer()

        category = analyzer.categorize_error("ValueError", "Invalid value provided")
        assert category == ErrorCategory.RUNTIME_ERROR

        category = analyzer.categorize_error("ModuleNotFoundError", "No module named 'xyz'")
        assert category == ErrorCategory.NOT_FOUND_ERROR

    def test_analyze_unknown_error(self):
        """Test analyzing an unknown error type."""
        analyzer = ErrorAnalyzer()

        tool_call = {"name": "test", "input": {}}
        result = {
            "error": "Some weird error",
            "error_type": "WeirdCustomError",
        }

        analysis = analyzer.analyze_error(tool_call, result)

        assert analysis.error_category == ErrorCategory.UNKNOWN_ERROR
        assert analysis.confidence == 0.0


class TestCorrectionGenerator:
    """Tests for CorrectionGenerator class."""

    def test_generator_initialization(self):
        """Test CorrectionGenerator initialization."""
        generator = CorrectionGenerator()
        assert generator._analyzer is not None

    def test_generate_runtime_error_correction(self):
        """Test generating correction for runtime error."""
        analyzer = ErrorAnalyzer()
        generator = CorrectionGenerator()

        tool_call = {"name": "run_code", "input": {"code": "result = data['key']"}}
        result = {"error": "'NoneType' object is not subscriptable", "error_type": "TypeError"}

        analysis = analyzer.analyze_error(tool_call, result)
        correction = generator.generate_correction(analysis, tool_call)

        assert correction.success is True
        assert correction.corrected_call is not None
        assert "try:" in correction.corrected_call["input"]["code"]
        assert "except" in correction.corrected_call["input"]["code"]

    def test_generate_syntax_error_correction(self):
        """Test generating correction for syntax error."""
        analyzer = ErrorAnalyzer()
        generator = CorrectionGenerator()

        tool_call = {"name": "run_code", "input": {"code": "if True x = 1"}}
        result = {"error": "invalid syntax", "error_type": "SyntaxError"}

        analysis = analyzer.analyze_error(tool_call, result)
        correction = generator.generate_correction(analysis, tool_call)

        # Syntax errors can't be auto-fixed
        assert correction.success is False
        assert "syntax" in correction.explanation.lower()
        assert len(correction.alternative_approaches) > 0

    def test_generate_not_found_correction(self):
        """Test generating correction for not found error."""
        analyzer = ErrorAnalyzer()
        generator = CorrectionGenerator()

        tool_call = {"name": "my_plugin", "input": {}}
        result = {"error": "Plugin not found", "error_type": "NotFoundError"}

        analysis = analyzer.analyze_error(tool_call, result)
        correction = generator.generate_correction(analysis, tool_call)

        assert correction.success is False
        assert "not found" in correction.explanation.lower()

    def test_validate_correction(self):
        """Test validating a proposed correction."""
        generator = CorrectionGenerator()

        # Valid correction
        valid_call = {"name": "test", "input": {"x": 1}}
        assert generator.validate_correction(valid_call) is True

        # Invalid corrections
        assert generator.validate_correction(None) is False
        assert generator.validate_correction({}) is False
        assert generator.validate_correction({"name": "test"}) is False
        assert generator.validate_correction({"name": "test", "input": "not_a_dict"}) is False

    def test_wrap_with_error_handling(self):
        """Test wrapping code with error handling."""
        generator = CorrectionGenerator()

        code = "result = 1 / 0"
        wrapped = generator._wrap_with_error_handling(code)

        assert "try:" in wrapped
        assert "except Exception" in wrapped
        assert "traceback" in wrapped
        assert code in wrapped


class TestReflectionLoop:
    """Tests for ReflectionLoop class."""

    def test_loop_initialization(self):
        """Test ReflectionLoop initialization."""
        loop = ReflectionLoop()
        assert loop._analyzer is not None
        assert loop._correction_generator is not None
        assert loop._max_history == 100

    def test_run_reflection_cycle(self):
        """Test running a complete reflection cycle."""
        loop = ReflectionLoop()

        tool_call = {"name": "test_tool", "input": {"code": "x = None + 1"}}
        result = {"error": "unsupported operand type", "error_type": "TypeError"}

        reflection_result = loop.run_reflection_cycle(tool_call, result)

        assert "analysis" in reflection_result
        assert "correction" in reflection_result
        assert "recommendation" in reflection_result

        assert reflection_result["analysis"]["error_type"] == "TypeError"

    def test_record_reflection_history(self):
        """Test that reflection events are recorded in history."""
        loop = ReflectionLoop(max_history=10)

        # Run multiple reflection cycles
        for i in range(15):
            tool_call = {"name": f"tool_{i}", "input": {}}
            result = {"error": f"Error {i}", "error_type": "RuntimeError"}
            loop.run_reflection_cycle(tool_call, result)

        # History should be limited to max_history
        history = loop.get_reflection_history(limit=100)
        assert len(history) == 10  # Limited by max_history

    def test_get_tool_success_rate(self):
        """Test tracking tool success rates."""
        loop = ReflectionLoop()

        # Initial rate should be 0.5 (default prior)
        assert loop.get_tool_success_rate("new_tool") == 0.5

        # Simulate successes and failures through reflection cycles
        for _ in range(3):
            tool_call = {"name": "tested_tool", "input": {"code": "pass"}}
            # Force successful correction
            result = {"error": "test", "error_type": "TypeError"}
            loop.run_reflection_cycle(tool_call, result)

        # Rate should reflect history
        rate = loop.get_tool_success_rate("tested_tool")
        assert 0.0 <= rate <= 1.0

    def test_clear_history(self):
        """Test clearing reflection history."""
        loop = ReflectionLoop()

        # Add some history
        tool_call = {"name": "test", "input": {}}
        result = {"error": "test", "error_type": "Error"}
        loop.run_reflection_cycle(tool_call, result)

        assert len(loop.get_reflection_history()) > 0

        # Clear history
        loop.clear_history()

        assert len(loop.get_reflection_history()) == 0

    def test_log_reflection_event(self):
        """Test logging reflection events."""
        loop = ReflectionLoop()

        event_data = {"tool_name": "test", "event": "reflection_test"}
        loop.log_reflection_event(event_data)

        # Should not raise any exceptions


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_reflection_loop_singleton(self):
        """Test that get_reflection_loop returns same instance."""
        loop1 = get_reflection_loop()
        loop2 = get_reflection_loop()

        assert loop1 is loop2


class TestIntegration:
    """Integration tests for reflection cycle."""

    def test_full_reflection_workflow(self):
        """Test complete workflow: error -> analysis -> correction."""
        loop = get_reflection_loop()

        # Simulate a realistic error scenario
        tool_call = {
            "name": "run_plugin",
            "input": {
                "code": "data = None\nresult = data['key']"
            },
        }
        result = {
            "error": "'NoneType' object is not subscriptable",
            "error_type": "TypeError",
        }
        traceback_str = "Traceback...\nTypeError: 'NoneType' object is not subscriptable"

        # Run reflection cycle
        reflection = loop.run_reflection_cycle(tool_call, result, traceback_str)

        # Verify all components worked together
        assert reflection["analysis"]["error_category"] == "RUNTIME_ERROR"
        assert reflection["correction"]["success"] is True
        assert "try:" in reflection["correction"]["corrected_call"]["input"]["code"]

        # Check recommendation
        assert "correction" in reflection["recommendation"].lower() or \
               "apply" in reflection["recommendation"].lower()

    def test_multiple_error_types(self):
        """Test handling various error types in sequence."""
        loop = ReflectionLoop()

        error_scenarios = [
            ({"name": "t1", "input": {}}, {"error": "invalid syntax", "error_type": "SyntaxError"}),
            ({"name": "t2", "input": {}}, {"error": "file not found", "error_type": "FileNotFoundError"}),
            ({"name": "t3", "input": {}}, {"error": "permission denied", "error_type": "PermissionError"}),
            ({"name": "t4", "input": {}}, {"error": "connection refused", "error_type": "ConnectionError"}),
        ]

        for tool_call, result in error_scenarios:
            reflection = loop.run_reflection_cycle(tool_call, result)
            assert "analysis" in reflection
            assert "correction" in reflection

        # Check that all were recorded
        history = loop.get_reflection_history(limit=10)
        assert len(history) == len(error_scenarios)
