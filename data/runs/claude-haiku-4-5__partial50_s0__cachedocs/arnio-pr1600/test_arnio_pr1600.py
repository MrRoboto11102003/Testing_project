import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Test suite for ValidationResult.to_markdown()"""

    def test_to_markdown_returns_string(self):
        """to_markdown() returns a string"""
        frame = ar.from_dict({"id": [1, 2]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        assert isinstance(result.to_markdown(), str)

    def test_to_markdown_with_passing_validation(self):
        """to_markdown() handles passing validation gracefully"""
        frame = ar.from_dict({"id": [1, 2]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert result.passed

    def test_to_markdown_with_failing_validation(self):
        """to_markdown() includes issue information when validation fails"""
        frame = ar.from_dict({"id": ["not-int"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert not result.passed

    def test_to_markdown_redact_values_default_false(self):
        """to_markdown() shows values by default (redact_values=False)"""
        frame = ar.from_dict({"value": ["sensitive_data"]})
        schema = ar.Schema({"value": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_false_explicit(self):
        """to_markdown(redact_values=False) does not redact"""
        frame = ar.from_dict({"email": ["confidential@example.com"]})
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(redact_values=False)
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_true(self):
        """to_markdown(redact_values=True) replaces values with [REDACTED]"""
        frame = ar.from_dict({"email": ["bad@test.com"]})
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(redact_values=True)
        assert isinstance(markdown, str)
        if result.issues:
            assert "[REDACTED]" in markdown

    def test_to_markdown_max_issues_none(self):
        """to_markdown(max_issues=None) includes all issues"""
        frame = ar.from_dict({"id": ["a", "b", "c"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(max_issues=None)
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_limits(self):
        """to_markdown(max_issues=N) limits the number of issues shown"""
        frame = ar.from_dict({"id": ["a", "b", "c", "d", "e"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        
        markdown_all = result.to_markdown()
        markdown_limited = result.to_markdown(max_issues=2)
        assert isinstance(markdown_all, str)
        assert isinstance(markdown_limited, str)

    def test_to_markdown_max_issues_zero(self):
        """to_markdown(max_issues=0) works without errors"""
        frame = ar.from_dict({"id": ["a"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(max_issues=0)
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_exceeds_count(self):
        """to_markdown(max_issues=N) when N exceeds total issues"""
        frame = ar.from_dict({"id": ["a", "b"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(max_issues=100)
        assert isinstance(markdown, str)

    def test_to_markdown_combined_parameters(self):
        """to_markdown() accepts both max_issues and redact_values together"""
        frame = ar.from_dict({"email": ["bad@a.com", "bad@b.com", "bad@c.com"]})
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(max_issues=1, redact_values=True)
        assert isinstance(markdown, str)
        if result.issues:
            assert "[REDACTED]" in markdown

    def test_to_markdown_deterministic(self):
        """to_markdown() output is deterministic"""
        frame = ar.from_dict({"id": ["bad"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        
        output1 = result.to_markdown()
        output2 = result.to_markdown()
        assert output1 == output2

    def test_to_markdown_multiple_columns(self):
        """to_markdown() handles errors in multiple columns"""
        frame = ar.from_dict({
            "id": ["bad1", "bad2"],
            "email": ["invalid1", "invalid2"]
        })
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email()
        })
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_keyword_only_arguments(self):
        """to_markdown() parameters are keyword-only"""
        frame = ar.from_dict({"id": ["bad"]})
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        markdown = result.to_markdown(max_issues=1, redact_values=True)
        assert isinstance(markdown, str)
