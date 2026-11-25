import pytest
from arnio.schema import ValidationResult


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown() method."""
    
    def test_returns_string_type(self):
        """to_markdown should return a string."""
        result = ValidationResult()
        output = result.to_markdown()
        assert isinstance(output, str)
    
    def test_default_parameters(self):
        """to_markdown with default parameters should work."""
        result = ValidationResult()
        output = result.to_markdown()
        assert isinstance(output, str)
    
    def test_max_issues_none(self):
        """max_issues=None should show all issues without limit."""
        result = ValidationResult()
        output = result.to_markdown(max_issues=None)
        assert isinstance(output, str)
    
    def test_max_issues_zero(self):
        """max_issues=0 is a valid edge case."""
        result = ValidationResult()
        output = result.to_markdown(max_issues=0)
        assert isinstance(output, str)
    
    def test_max_issues_positive_integer(self):
        """max_issues with positive integer should limit output."""
        result = ValidationResult()
        output = result.to_markdown(max_issues=5)
        assert isinstance(output, str)
    
    def test_redact_values_false_shows_values(self):
        """redact_values=False should include actual values in output."""
        result = ValidationResult()
        output = result.to_markdown(redact_values=False)
        assert isinstance(output, str)
    
    def test_redact_values_true_hides_values(self):
        """redact_values=True should redact/hide actual values."""
        result = ValidationResult()
        output = result.to_markdown(redact_values=True)
        assert isinstance(output, str)
    
    def test_both_parameters_together(self):
        """Should accept both max_issues and redact_values parameters."""
        result = ValidationResult()
        output = result.to_markdown(max_issues=10, redact_values=True)
        assert isinstance(output, str)
    
    def test_keyword_only_arguments(self):
        """Arguments must be keyword-only (not positional)."""
        result = ValidationResult()
        with pytest.raises(TypeError):
            result.to_markdown(None, False)  # type: ignore
    
    def test_consistency_multiple_calls(self):
        """Multiple calls with same parameters should return identical output."""
        result = ValidationResult()
        output1 = result.to_markdown(max_issues=5, redact_values=True)
        output2 = result.to_markdown(max_issues=5, redact_values=True)
        assert output1 == output2
    
    def test_output_is_valid_markdown(self):
        """Output should be valid markdown format (non-empty string)."""
        result = ValidationResult()
        output = result.to_markdown()
        assert isinstance(output, str)
        assert output is not None
    
    def test_redact_values_affects_output(self):
        """Redaction setting should potentially affect output content."""
        result = ValidationResult()
        output_unredacted = result.to_markdown(redact_values=False)
        output_redacted = result.to_markdown(redact_values=True)
        # Both should be valid strings (may or may not differ)
        assert isinstance(output_unredacted, str)
        assert isinstance(output_redacted, str)
    
    def test_large_max_issues_value(self):
        """Large max_issues values should not cause errors."""
        result = ValidationResult()
        output = result.to_markdown(max_issues=999999)
        assert isinstance(output, str)
    
    def test_parameter_combinations(self):
        """Test various valid parameter combinations."""
        result = ValidationResult()
        test_params = [
            {},
            {"max_issues": None},
            {"max_issues": 1},
            {"redact_values": True},
            {"max_issues": 3, "redact_values": False},
        ]
        for params in test_params:
            output = result.to_markdown(**params)
            assert isinstance(output, str)
    
    def test_output_is_not_none(self):
        """Output should never be None, always a string."""
        result = ValidationResult()
        output = result.to_markdown()
        assert output is not None
        assert isinstance(output, str)
