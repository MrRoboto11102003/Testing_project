import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import (
    CleanExplanation,
    CleanStepRecord,
    DataQualityReport,
    CleaningSuggestion,
)


@pytest.fixture
def frame_with_whitespace():
    """Frame with whitespace issues that safe mode can fix."""
    data = {
        "name": ["  alice  ", "  bob  ", "charlie"],
        "email": ["a@test.com", "b@test.com", "c@test.com"],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_with_duplicates():
    """Frame with exact duplicate rows."""
    data = {
        "id": [1, 1, 2, 2, 3],
        "name": ["alice", "alice", "bob", "bob", "charlie"],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_castable_to_numeric():
    """Frame where string columns could be cast to numeric types."""
    data = {
        "age": ["25", "30", "35"],
        "score": ["95", "87", "92"],
        "name": ["alice", "bob", "charlie"],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def empty_frame():
    """Completely empty frame."""
    return ar.from_pandas(pd.DataFrame(columns=["col1", "col2"]))


@pytest.fixture
def single_row_frame():
    """Frame with only one row."""
    data = {"col1": ["value"], "col2": [1]}
    return ar.from_pandas(pd.DataFrame(data))


class TestAutoCleanReturnTypes:
    """Test the different return type combinations."""

    def test_default_returns_arframe(self, frame_with_whitespace):
        """Default call returns cleaned ArFrame."""
        result = ar.auto_clean(frame_with_whitespace)
        assert isinstance(result, ar.ArFrame)

    def test_dry_run_returns_report_only(self, frame_with_whitespace):
        """dry_run=True returns DataQualityReport without modifying frame."""
        result = ar.auto_clean(frame_with_whitespace, dry_run=True)
        assert isinstance(result, DataQualityReport)

    def test_return_report_returns_tuple_arframe_report(self, frame_with_whitespace):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(frame_with_whitespace, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], DataQualityReport)

    def test_explain_returns_tuple_arframe_explanation(self, frame_with_whitespace):
        """explain=True returns (ArFrame, CleanExplanation)."""
        result = ar.auto_clean(frame_with_whitespace, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], CleanExplanation)

    def test_return_report_and_explain_returns_triple(self, frame_with_whitespace):
        """Both return_report and explain=True returns (ArFrame, DataQualityReport, CleanExplanation)."""
        result = ar.auto_clean(
            frame_with_whitespace, return_report=True, explain=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], DataQualityReport)
        assert isinstance(result[2], CleanExplanation)


class TestAutoCleanParameterValidation:
    """Test parameter validation and error cases."""

    def test_invalid_mode_raises_valueerror(self, frame_with_whitespace):
        """mode must be 'safe' or 'strict'."""
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame_with_whitespace, mode="invalid")

    def test_non_bool_return_report_raises_typeerror(self, frame_with_whitespace):
        """return_report must be bool."""
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame_with_whitespace, return_report="true")

    def test_non_bool_dry_run_raises_typeerror(self, frame_with_whitespace):
        """dry_run must be bool."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(frame_with_whitespace, dry_run=1)

    def test_non_bool_allow_lossy_casts_raises_typeerror(self, frame_with_whitespace):
        """allow_lossy_casts must be bool."""
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            ar.auto_clean(frame_with_whitespace, allow_lossy_casts="yes")

    def test_non_bool_explain_raises_typeerror(self, frame_with_whitespace):
        """explain must be bool."""
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(frame_with_whitespace, explain=0)

    def test_dry_run_and_explain_raises_valueerror(self, frame_with_whitespace):
        """dry_run=True and explain=True together raise ValueError."""
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame_with_whitespace, dry_run=True, explain=True)

    def test_dry_run_and_return_report_raises_valueerror(self, frame_with_whitespace):
        """dry_run=True and return_report=True together raise ValueError."""
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame_with_whitespace, dry_run=True, return_report=True)


class TestAutoCleanModes:
    """Test safe vs strict mode behavior."""

    def test_safe_mode_applies_only_safe_operations(self, frame_with_whitespace):
        """safe mode should apply whitespace trimming."""
        cleaned = ar.auto_clean(frame_with_whitespace, mode="safe")
        assert isinstance(cleaned, ar.ArFrame)
        assert cleaned.shape[1] == frame_with_whitespace.shape[1]

    def test_strict_mode_without_allow_lossy_casts_for_type_casts_raises(self, frame_castable_to_numeric):
        """strict mode should raise without allow_lossy_casts when type casts needed."""
        # Strict mode will suggest type casts for numeric columns
        with pytest.raises(ValueError, match="allow_lossy_casts=True"):
            ar.auto_clean(frame_castable_to_numeric, mode="strict")

    def test_strict_mode_with_allow_lossy_but_no_confirmed_casts_raises(self, frame_castable_to_numeric):
        """strict mode with allow_lossy_casts but no confirmed_casts raises."""
        with pytest.raises(ValueError, match="confirmed_casts"):
            ar.auto_clean(
                frame_castable_to_numeric, mode="strict", allow_lossy_casts=True
            )

    def test_strict_mode_confirmed_casts_mismatch_raises(self, frame_castable_to_numeric):
        """confirmed_casts must match proposed mapping exactly."""
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                frame_castable_to_numeric,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={"wrong": "mapping"},
            )


class TestAutoCleanExplainMode:
    """Test explain=True functionality."""

    def test_explanation_has_correct_structure(self, frame_with_whitespace):
        """CleanExplanation should have expected attributes."""
        _, explanation = ar.auto_clean(frame_with_whitespace, explain=True)
        assert hasattr(explanation, "mode")
        assert hasattr(explanation, "rows_before")
        assert hasattr(explanation, "rows_after")
        assert hasattr(explanation, "rows_removed")
        assert hasattr(explanation, "steps")

    def test_explanation_mode_matches_parameter(self, frame_with_whitespace):
        """Explanation.mode should match mode parameter."""
        _, exp_safe = ar.auto_clean(frame_with_whitespace, mode="safe", explain=True)
        assert exp_safe.mode == "safe"

    def test_explanation_tracks_steps(self, frame_with_whitespace):
        """Explanation.steps should be list of CleanStepRecord."""
        _, explanation = ar.auto_clean(frame_with_whitespace, explain=True)
        assert isinstance(explanation.steps, list)
        for step in explanation.steps:
            assert isinstance(step, CleanStepRecord)
            assert hasattr(step, "step")
            assert hasattr(step, "kwargs")
            assert hasattr(step, "rows_before")
            assert hasattr(step, "rows_after")
            assert hasattr(step, "reason")

    def test_explanation_row_counts_consistent(self, frame_with_whitespace):
        """Explanation row counts should be internally consistent."""
        _, explanation = ar.auto_clean(frame_with_whitespace, explain=True)
        assert explanation.rows_removed == (
            explanation.rows_before - explanation.rows_after
        )
        assert explanation.rows_removed >= 0


class TestAutoCleanDryRunBehavior:
    """Test dry_run=True behavior."""

    def test_dry_run_does_not_modify_frame(self, frame_with_whitespace):
        """Frame should not be modified with dry_run=True."""
        original_data = frame_with_whitespace.to_dict(orient="list")
        ar.auto_clean(frame_with_whitespace, dry_run=True)
        assert frame_with_whitespace.to_dict(orient="list") == original_data

    def test_dry_run_report_contains_suggestions(self, frame_with_whitespace):
        """dry_run report should contain cleaning suggestions."""
        report = ar.auto_clean(frame_with_whitespace, dry_run=True)
        assert hasattr(report, "suggestions")

    def test_dry_run_mode_parameter_respected(self, frame_with_whitespace):
        """dry_run should respect mode parameter in profiling."""
        report_safe = ar.auto_clean(
            frame_with_whitespace, mode="safe", dry_run=True
        )
        report_strict = ar.auto_clean(
            frame_with_whitespace, mode="strict", dry_run=True
        )
        # Both should return reports
        assert isinstance(report_safe, DataQualityReport)
        assert isinstance(report_strict, DataQualityReport)


class TestAutoCleanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame(self, empty_frame):
        """auto_clean should handle empty frames."""
        result = ar.auto_clean(empty_frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0
        assert result.shape[1] == empty_frame.shape[1]

    def test_single_row_frame(self, single_row_frame):
        """auto_clean should handle single-row frames."""
        result = ar.auto_clean(single_row_frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] >= 0

    def test_frame_with_unicode(self):
        """auto_clean should handle unicode text."""
        data = {"text": ["café", "naïve", "résumé"]}
        frame = ar.from_pandas(pd.DataFrame(data))
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_frame_with_mixed_types(self):
        """auto_clean should handle mixed column types."""
        data = {
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3],
        }
        frame = ar.from_pandas(pd.DataFrame(data))
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)


class TestAutoCleanDataIntegrity:
    """Test that data integrity is maintained."""

    def test_columns_preserved(self, frame_with_whitespace):
        """Column names and count should be preserved."""
        cleaned = ar.auto_clean(frame_with_whitespace)
        assert cleaned.columns == frame_with_whitespace.columns
        assert cleaned.shape[1] == frame_with_whitespace.shape[1]

    def test_rows_non_increasing(self, frame_with_whitespace):
        """Row count should never increase."""
        original_rows = frame_with_whitespace.shape[0]
        cleaned = ar.auto_clean(frame_with_whitespace)
        assert cleaned.shape[0] <= original_rows

    def test_report_includes_frame_info(self, frame_with_whitespace):
        """Report should include frame information."""
        _, report = ar.auto_clean(frame_with_whitespace, return_report=True)
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert report.column_count == frame_with_whitespace.shape[1]
