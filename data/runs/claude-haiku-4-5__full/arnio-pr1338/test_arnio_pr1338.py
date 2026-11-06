import pytest
import pandas as pd
import numpy as np
import arnio as ar


# Helper functions
def make_frame(data_dict):
    """Create an ArFrame from a dictionary."""
    df = pd.DataFrame(data_dict)
    return ar.from_pandas(df)


class TestAutoCleanBasic:
    """Basic functionality tests for ar.auto_clean()."""

    def test_auto_clean_safe_mode_default(self):
        """safe mode is the default and applies whitespace trimming."""
        frame = make_frame({"name": [" alice ", " bob "]})
        clean = ar.auto_clean(frame)
        result = ar.to_pandas(clean)
        assert result["name"].tolist() == ["alice", "bob"]

    def test_auto_clean_safe_mode_explicit(self):
        """Explicit safe mode trims whitespace only."""
        frame = make_frame({"city": [" paris ", " london "]})
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert result["city"].tolist() == ["paris", "london"]

    def test_auto_clean_safe_mode_preserves_duplicates(self):
        """safe mode does not remove duplicate rows."""
        frame = make_frame({"id": [1, 1, 2], "name": ["alice", "alice", "bob"]})
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 3

    def test_auto_clean_strict_mode_drops_exact_duplicates(self):
        """strict mode removes exact duplicate rows."""
        frame = make_frame({"id": [1, 1, 2], "name": ["alice", "alice", "bob"]})
        clean = ar.auto_clean(frame, mode="strict")
        result = ar.to_pandas(clean)
        assert len(result) == 2
        assert result["id"].tolist() == [1, 2]


class TestAutoCleanReturnTypes:
    """Test different return type combinations."""

    def test_auto_clean_returns_frame_by_default(self):
        """Default return is an ArFrame."""
        frame = make_frame({"name": ["alice"]})
        clean = ar.auto_clean(frame, mode="safe")
        assert isinstance(clean, ar.ArFrame)

    def test_auto_clean_dry_run_returns_report(self):
        """dry_run=True returns DataQualityReport without applying changes."""
        frame = make_frame({"name": [" alice ", " bob "], "id": [1, 1]})
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        assert hasattr(report, 'row_count')
        assert hasattr(report, 'suggestions')

    def test_auto_clean_return_report_true(self):
        """return_report=True returns (frame, report) tuple."""
        frame = make_frame({"name": [" alice "]})
        result = ar.auto_clean(frame, mode="safe", return_report=True)
        assert isinstance(result, tuple) and len(result) == 2
        clean, report = result
        assert isinstance(clean, ar.ArFrame)
        assert hasattr(report, 'row_count')

    def test_auto_clean_explain_true(self):
        """explain=True returns (frame, explanation) tuple."""
        frame = make_frame({"name": [" alice "]})
        result = ar.auto_clean(frame, mode="safe", explain=True)
        assert isinstance(result, tuple) and len(result) == 2
        clean, explanation = result
        assert isinstance(clean, ar.ArFrame)

    def test_auto_clean_explain_and_report_both_true(self):
        """Both explain and return_report=True returns 3-tuple."""
        frame = make_frame({"name": [" alice "]})
        result = ar.auto_clean(
            frame, mode="safe", explain=True, return_report=True
        )
        assert isinstance(result, tuple) and len(result) == 3
        clean, report, explanation = result
        assert isinstance(clean, ar.ArFrame)
        assert hasattr(report, 'row_count')


class TestAutoCleanModeValidation:
    """Test mode parameter validation."""

    def test_auto_clean_invalid_mode_raises_error(self):
        """Invalid mode raises ValueError."""
        frame = make_frame({"name": ["alice"]})
        with pytest.raises(ValueError):
            ar.auto_clean(frame, mode="invalid_mode")

    def test_auto_clean_mode_case_sensitive(self):
        """Mode parameter is case-sensitive."""
        frame = make_frame({"name": ["alice"]})
        with pytest.raises(ValueError):
            ar.auto_clean(frame, mode="Safe")  # Should be lowercase


class TestAutoCleanStrictMode:
    """Test strict mode behavior and safety gates."""

    def test_auto_clean_strict_mode_with_whitespace(self):
        """strict mode trims whitespace (like safe)."""
        frame = make_frame({"name": [" alice "]})
        clean = ar.auto_clean(frame, mode="strict")
        result = ar.to_pandas(clean)
        assert result["name"].iloc[0] == "alice"

    def test_auto_clean_strict_dry_run_shows_suggestions(self):
        """strict mode dry_run shows suggested transformations."""
        frame = make_frame({
            "code": [" 001 ", " 002 "],
            "id": [1, 1]  # duplicate
        })
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        assert hasattr(report, 'suggestions')


class TestAutoCleanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_auto_clean_empty_frame(self):
        """auto_clean handles empty frames gracefully."""
        frame = make_frame({"name": [], "age": []})
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 0
        assert list(result.columns) == ["name", "age"]

    def test_auto_clean_single_row(self):
        """auto_clean processes single-row frames correctly."""
        frame = make_frame({"name": [" alice "], "age": [30]})
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 1
        assert result["name"].iloc[0] == "alice"

    def test_auto_clean_all_nulls_column(self):
        """auto_clean preserves columns with all nulls."""
        frame = make_frame({
            "name": [" alice ", " bob "],
            "missing": [None, None]
        })
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 2
        assert result["missing"].isna().all()

    def test_auto_clean_mixed_nulls_and_whitespace(self):
        """auto_clean handles mix of nulls and whitespace."""
        frame = make_frame({
            "name": [" alice ", None, " bob "],
            "age": [25, None, 35]
        })
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert result["name"].iloc[0] == "alice"
        assert pd.isna(result["name"].iloc[1])
        assert result["name"].iloc[2] == "bob"


class TestAutoCleanMultipleColumns:
    """Test behavior with multiple columns."""

    def test_auto_clean_safe_mode_all_string_columns(self):
        """safe mode strips whitespace from all string columns."""
        frame = make_frame({
            "col1": [" a ", " b "],
            "col2": [" x ", " y "],
            "col3": [" p ", " q "]
        })
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert result["col1"].tolist() == ["a", "b"]
        assert result["col2"].tolist() == ["x", "y"]
        assert result["col3"].tolist() == ["p", "q"]

    def test_auto_clean_numeric_columns_unchanged(self):
        """safe mode does not modify numeric columns."""
        frame = make_frame({
            "age": [25, 30, 35],
            "score": [95.5, 88.2, 92.1]
        })
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert result["age"].tolist() == [25, 30, 35]
        assert result["score"].tolist() == [95.5, 88.2, 92.1]

    def test_auto_clean_mixed_column_types(self):
        """auto_clean preserves numeric columns while cleaning strings."""
        frame = make_frame({
            "id": [1, 2, 3],
            "name": [" alice ", " bob ", " charlie "],
            "score": [95.0, 88.0, 92.0]
        })
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["name"].tolist() == ["alice", "bob", "charlie"]
        assert result["score"].tolist() == [95.0, 88.0, 92.0]


class TestAutoCleanDryRunBehavior:
    """Test dry_run parameter behavior."""

    def test_auto_clean_dry_run_does_not_modify_frame(self):
        """dry_run=True does not modify input (when passed a report)."""
        frame = make_frame({"name": [" alice ", " bob "], "id": [1, 1]})
        original_count = ar.to_pandas(frame).shape[0]
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        # Frame should remain unchanged after dry_run (report shows what would happen)
        assert ar.to_pandas(frame).shape[0] == original_count

    def test_auto_clean_strict_dry_run_returns_report_not_frame(self):
        """strict mode dry_run returns report, not cleaned frame."""
        frame = make_frame({"id": [1, 1, 2]})
        result = ar.auto_clean(frame, mode="strict", dry_run=True)
        # Result should be a report, not an ArFrame
        assert hasattr(result, 'row_count')
        assert not isinstance(result, ar.ArFrame)
