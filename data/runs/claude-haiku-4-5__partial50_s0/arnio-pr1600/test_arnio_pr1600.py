import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown() method."""

    @pytest.fixture
    def passing_validation(self):
        """Create a validation result that passes all checks."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "age": [25, 30, 35],
        }))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, min=0),
        })
        return ar.validate(frame, schema)

    @pytest.fixture
    def failing_validation(self):
        """Create a validation result with multiple issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, "invalid", 4],
            "email": ["valid@example.com", "not_email", None, "another@example.com"],
            "age": [25, 30, -10, 150],
        }))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=120),
        })
        return ar.validate(frame, schema)

    def test_to_markdown_returns_string(self, passing_validation):
        """to_markdown() returns a string."""
        output = passing_validation.to_markdown()
        assert isinstance(output, str)

    def test_to_markdown_with_no_issues(self, passing_validation):
        """to_markdown() works when validation passes."""
        assert passing_validation.passed
        output = passing_validation.to_markdown()
        assert isinstance(output, str)

    def test_to_markdown_with_issues(self, failing_validation):
        """to_markdown() produces output for failing validation."""
        assert not failing_validation.passed
        output = failing_validation.to_markdown()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_to_markdown_max_issues_limits_output(self, failing_validation):
        """max_issues parameter limits number of issues in output."""
        output_all = failing_validation.to_markdown()
        output_limit = failing_validation.to_markdown(max_issues=1)
        assert isinstance(output_all, str)
        assert isinstance(output_limit, str)
        # Limited output should not exceed all issues output in length
        assert len(output_limit) <= len(output_all)

    def test_to_markdown_max_issues_zero(self, failing_validation):
        """max_issues=0 produces valid output."""
        output = failing_validation.to_markdown(max_issues=0)
        assert isinstance(output, str)

    def test_to_markdown_max_issues_none(self, failing_validation):
        """max_issues=None shows all issues (default behavior)."""
        output = failing_validation.to_markdown(max_issues=None)
        assert isinstance(output, str)

    def test_to_markdown_max_issues_exceeds_count(self, failing_validation):
        """max_issues larger than actual issue count."""
        output = failing_validation.to_markdown(max_issues=10000)
        assert isinstance(output, str)

    def test_to_markdown_redact_values_false(self, failing_validation):
        """redact_values=False shows original values."""
        output = failing_validation.to_markdown(redact_values=False)
        assert isinstance(output, str)

    def test_to_markdown_redact_values_true(self, failing_validation):
        """redact_values=True hides sensitive values."""
        output = failing_validation.to_markdown(redact_values=True)
        assert isinstance(output, str)

    def test_to_markdown_both_parameters(self, failing_validation):
        """to_markdown() with both max_issues and redact_values."""
        output = failing_validation.to_markdown(max_issues=5, redact_values=True)
        assert isinstance(output, str)

    def test_to_markdown_keyword_only_parameters(self, passing_validation):
        """to_markdown() enforces keyword-only parameters."""
        with pytest.raises(TypeError):
            passing_validation.to_markdown(10)  # max_issues as positional

    def test_to_markdown_idempotent(self, failing_validation):
        """Multiple calls return identical output."""
        output1 = failing_validation.to_markdown()
        output2 = failing_validation.to_markdown()
        assert output1 == output2

    def test_to_markdown_consistent_with_parameters(self, failing_validation):
        """Same parameters produce consistent output."""
        output1 = failing_validation.to_markdown(max_issues=3, redact_values=True)
        output2 = failing_validation.to_markdown(max_issues=3, redact_values=True)
        assert output1 == output2
