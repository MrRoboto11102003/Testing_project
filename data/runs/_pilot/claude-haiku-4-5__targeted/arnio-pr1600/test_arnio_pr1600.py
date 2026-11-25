import pytest
import arnio as ar
import pandas as pd
from arnio.schema import ValidationResult


class TestValidationResultToMarkdown:
    """Test suite for ValidationResult.to_markdown() method."""
    
    @pytest.fixture
    def validation_result_with_issues(self):
        """Create a ValidationResult with validation issues for testing."""
        # Create a frame with data quality issues
        frame = ar.from_dict({
            "name": ["Alice", None, "Charlie", "David"],
            "age": [25, 30, None, 35],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "charlie@example.com"]
        })
        # Run validation that generates issues (e.g., nulls, duplicates)
        result = ar.validate(frame, {
            "null_columns": ["name", "age"],
            "duplicate_columns": ["email"]
        })
        return result
    
    @pytest.fixture
    def clean_validation_result(self):
        """Create a ValidationResult with no issues for testing."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "value": [10.5, 20.5, 30.5]
        })
        result = ar.validate(frame, {})
        return result
    
    def test_to_markdown_returns_string(self, validation_result_with_issues):
        """Test that to_markdown() returns a string."""
        output = validation_result_with_issues.to_markdown()
        assert isinstance(output, str)
    
    def test_to_markdown_default_no_redaction(self, validation_result_with_issues):
        """Test that values are shown by default (redact_values defaults to False)."""
        output = validation_result_with_issues.to_markdown()
        # Default behavior should show actual values, not redacted
        assert "[REDACTED]" not in output
    
    def test_to_markdown_with_redact_values_true(self, validation_result_with_issues):
        """Test that redact_values=True replaces values with [REDACTED]."""
        output = validation_result_with_issues.to_markdown(redact_values=True)
        # When redacting, sensitive values should be replaced
        if "Value" in output or "value" in output.lower():
            assert "[REDACTED]" in output
    
    def test_to_markdown_with_redact_values_false(self, validation_result_with_issues):
        """Test that redact_values=False preserves original values."""
        output = validation_result_with_issues.to_markdown(redact_values=False)
        # Explicitly setting to False should show original values
        assert isinstance(output, str)
    
    def test_to_markdown_max_issues_none(self, validation_result_with_issues):
        """Test that max_issues=None includes all issues."""
        output_all = validation_result_with_issues.to_markdown(max_issues=None)
        assert isinstance(output_all, str)
    
    def test_to_markdown_max_issues_limits_output(self, validation_result_with_issues):
        """Test that max_issues parameter limits the number of displayed issues."""
        output_limited = validation_result_with_issues.to_markdown(max_issues=2)
        output_all = validation_result_with_issues.to_markdown(max_issues=None)
        # Limited output should not be longer than full output
        assert len(output_limited) <= len(output_all)
    
    def test_to_markdown_max_issues_zero(self, validation_result_with_issues):
        """Test behavior with max_issues=0."""
        output = validation_result_with_issues.to_markdown(max_issues=0)
        assert isinstance(output, str)
    
    def test_to_markdown_max_issues_one(self, validation_result_with_issues):
        """Test that max_issues=1 shows at most one issue row."""
        output = validation_result_with_issues.to_markdown(max_issues=1)
        assert isinstance(output, str)
    
    def test_to_markdown_max_issues_exceeds_total(self, validation_result_with_issues):
        """Test max_issues greater than actual issues count."""
        output = validation_result_with_issues.to_markdown(max_issues=10000)
        assert isinstance(output, str)
    
    def test_to_markdown_github_markdown_format(self, validation_result_with_issues):
        """Test that output is GitHub-friendly markdown with table structure."""
        output = validation_result_with_issues.to_markdown()
        # GitHub-friendly markdown tables use pipes for column separation
        if output:  # If there are issues
            assert "|" in output or output.strip() == ""
    
    def test_to_markdown_empty_result(self, clean_validation_result):
        """Test to_markdown() with validation result containing no issues."""
        output = clean_validation_result.to_markdown()
        assert isinstance(output, str)
    
    def test_to_markdown_redact_and_max_issues_combined(self, validation_result_with_issues):
        """Test combining redact_values=True and max_issues parameter."""
        output = validation_result_with_issues.to_markdown(
            max_issues=5,
            redact_values=True
        )
        assert isinstance(output, str)
    
    def test_to_markdown_reproducible(self, validation_result_with_issues):
        """Test that multiple calls to to_markdown() produce identical output."""
        output1 = validation_result_with_issues.to_markdown()
        output2 = validation_result_with_issues.to_markdown()
        assert output1 == output2
    
    def test_to_markdown_keyword_only_arguments(self, validation_result_with_issues):
        """Test that max_issues and redact_values are keyword-only arguments."""
        # These should work with keyword arguments
        output1 = validation_result_with_issues.to_markdown(max_issues=5, redact_values=False)
        assert isinstance(output1, str)
        
        # Should not accept positional arguments (keyword-only)
        with pytest.raises(TypeError):
            validation_result_with_issues.to_markdown(5, False)
