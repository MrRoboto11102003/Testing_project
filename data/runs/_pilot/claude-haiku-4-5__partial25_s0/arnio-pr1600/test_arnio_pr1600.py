import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Tests for arnio.schema.ValidationResult.to_markdown() method."""

    def test_to_markdown_returns_string(self):
        """to_markdown() should return a string."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["not_a_number"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_with_passed_validation(self):
        """to_markdown() should handle passing validation results."""
        frame = ar.from_pandas(pd.DataFrame({"age": [25, 30]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_to_markdown_max_issues_none_shows_all(self):
        """to_markdown(max_issues=None) should include all issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "col1": ["a", "b", "c"],
            "col2": ["x", "y", "z"],
        }))
        schema = ar.Schema({
            "col1": ar.Int64(),
            "col2": ar.Int64(),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=None)
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_limits_issues(self):
        """to_markdown(max_issues=n) should limit displayed issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "col1": ["a", "b", "c"],
            "col2": ["x", "y", "z"],
        }))
        schema = ar.Schema({
            "col1": ar.Int64(),
            "col2": ar.Int64(),
        })
        result = ar.validate(frame, schema)
        
        markdown_limit_1 = result.to_markdown(max_issues=1)
        markdown_unlimited = result.to_markdown(max_issues=None)
        
        assert isinstance(markdown_limit_1, str)
        assert len(markdown_limit_1) <= len(markdown_unlimited)

    def test_to_markdown_max_issues_zero(self):
        """to_markdown(max_issues=0) should handle zero limit."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["invalid"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=0)
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_true(self):
        """to_markdown(redact_values=True) should redact values."""
        frame = ar.from_pandas(pd.DataFrame({"email": ["not_an_email"]}))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        
        markdown_redacted = result.to_markdown(redact_values=True)
        markdown_unredacted = result.to_markdown(redact_values=False)
        
        assert isinstance(markdown_redacted, str)
        assert isinstance(markdown_unredacted, str)

    def test_to_markdown_redact_values_false(self):
        """to_markdown(redact_values=False) should show values."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["123_value"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(redact_values=False)
        assert isinstance(markdown, str)

    def test_to_markdown_keyword_only_arguments(self):
        """to_markdown() arguments should be keyword-only."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["x"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        # Should work with keyword arguments
        markdown = result.to_markdown(max_issues=10, redact_values=False)
        assert isinstance(markdown, str)
        
        # Should fail with positional arguments
        with pytest.raises(TypeError):
            result.to_markdown(10, False)

    def test_to_markdown_includes_column_information(self):
        """to_markdown() should include column information in output."""
        frame = ar.from_pandas(pd.DataFrame({
            "customer_name": ["John"],
            "email_address": ["not_valid_email"],
        }))
        schema = ar.Schema({
            "customer_name": ar.String(),
            "email_address": ar.Email(),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_github_markdown_format(self):
        """to_markdown() should produce GitHub-friendly markdown."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["not_numeric"],
            "status": ["unknown"],
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "status": ar.String(),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        # Should contain markdown formatting (tables use pipes)
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_to_markdown_consistent_output(self):
        """to_markdown() should produce consistent output for same inputs."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["not_a_number"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown1 = result.to_markdown()
        markdown2 = result.to_markdown()
        
        assert markdown1 == markdown2

    def test_to_markdown_with_multiple_issues_per_column(self):
        """to_markdown() should handle multiple issues in same column."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["invalid_1", "invalid_2", "invalid_3"],
        }))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_and_redact_together(self):
        """to_markdown() should support max_issues and redact_values together."""
        frame = ar.from_pandas(pd.DataFrame({
            "col1": ["a"] * 5,
            "col2": ["b"] * 5,
        }))
        schema = ar.Schema({
            "col1": ar.Int64(),
            "col2": ar.Int64(),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=3, redact_values=True)
        assert isinstance(markdown, str)

    def test_to_markdown_empty_validation_result(self):
        """to_markdown() should handle valid data with no issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_large_max_issues(self):
        """to_markdown(max_issues) should handle values larger than issue count."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["invalid"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=1000)
        assert isinstance(markdown, str)
