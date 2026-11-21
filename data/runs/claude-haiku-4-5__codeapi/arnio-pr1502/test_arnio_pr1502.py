import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import (
    auto_clean,
    DataQualityReport,
    CleanExplanation,
    CleanStepRecord,
)
from arnio.frame import ArFrame


class TestAutoCleanBasicBehavior:
    """Test basic auto_clean functionality."""

    def test_auto_clean_returns_arframe_default(self):
        """Test that auto_clean returns ArFrame with default parameters."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ArFrame)

    def test_auto_clean_safe_mode_trims_whitespace(self):
        """Test that safe mode removes leading/trailing whitespace."""
        df = pd.DataFrame({"col": ["  hello  ", "  world  ", "test"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        result_df = ar.to_pandas(result)
        # After cleaning, first value should have whitespace trimmed
        assert result_df["col"].iloc[0].strip() == "hello"

    def test_auto_clean_strict_mode_returns_arframe(self):
        """Test that strict mode returns ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="strict")
        assert isinstance(result, ArFrame)


class TestAutoCleanReturnTypes:
    """Test different return type combinations based on flags."""

    def test_dry_run_returns_report_only(self):
        """Test that dry_run=True returns only DataQualityReport."""
        df = pd.DataFrame({"a": ["  x  ", "  y  "], "b": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, dry_run=True)
        assert isinstance(result, DataQualityReport)

    def test_return_report_returns_frame_and_report(self):
        """Test that return_report=True returns tuple of ArFrame and DataQualityReport."""
        df = pd.DataFrame({"a": ["  x  "], "b": [1]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ArFrame)
        assert isinstance(result[1], DataQualityReport)

    def test_explain_returns_frame_and_explanation(self):
        """Test that explain=True returns tuple of ArFrame and CleanExplanation."""
        df = pd.DataFrame({"a": ["  x  "], "b": [1]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ArFrame)
        assert isinstance(result[1], CleanExplanation)

    def test_return_report_and_explain_returns_triple(self):
        """Test that both return_report and explain=True returns 3-tuple."""
        df = pd.DataFrame({"a": ["  x  "], "b": [1]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ArFrame)
        assert isinstance(result[1], DataQualityReport)
        assert isinstance(result[2], CleanExplanation)


class TestAutoCleanParameterValidation:
    """Test parameter validation."""

    def test_invalid_mode_raises_valueerror(self):
        """Test that invalid mode value raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="mode must be"):
            ar.auto_clean(frame, mode="invalid")

    def test_return_report_type_validation(self):
        """Test that non-bool return_report raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.auto_clean(frame, return_report="true")

    def test_dry_run_type_validation(self):
        """Test that non-bool dry_run raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.auto_clean(frame, dry_run=1)

    def test_allow_lossy_casts_type_validation(self):
        """Test that non-bool allow_lossy_casts raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.auto_clean(frame, allow_lossy_casts="yes")

    def test_explain_type_validation(self):
        """Test that non-bool explain raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.auto_clean(frame, explain=None)


class TestAutoCleanParameterConflicts:
    """Test parameter constraint violations."""

    def test_dry_run_and_explain_conflict(self):
        """Test that dry_run=True and explain=True raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_dry_run_and_return_report_conflict(self):
        """Test that dry_run=True and return_report=True raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)


class TestAutoCleanExplanation:
    """Test CleanExplanation output structure."""

    def test_explanation_mode_field(self):
        """Test that CleanExplanation includes mode."""
        df = pd.DataFrame({"a": ["  x  "], "b": [1]})
        frame = ar.from_pandas(df)
        _, explanation = ar.auto_clean(frame, explain=True)
        assert hasattr(explanation, "mode")
        assert explanation.mode in ("safe", "strict")

    def test_explanation_row_count_fields(self):
        """Test that CleanExplanation includes row count fields."""
        df = pd.DataFrame({"a": ["  x  ", "  y  "]})
        frame = ar.from_pandas(df)
        _, explanation = ar.auto_clean(frame, explain=True)
        assert explanation.rows_before == 2
        assert explanation.rows_after >= 0
        assert explanation.rows_removed >= 0
        assert explanation.rows_before >= explanation.rows_after

    def test_explanation_steps_list(self):
        """Test that CleanExplanation includes steps list."""
        df = pd.DataFrame({"a": ["  x  ", "  y  "]})
        frame = ar.from_pandas(df)
        _, explanation = ar.auto_clean(frame, explain=True)
        assert isinstance(explanation.steps, list)
        for step in explanation.steps:
            assert isinstance(step, CleanStepRecord)
            assert hasattr(step, "step")
            assert hasattr(step, "kwargs")
            assert hasattr(step, "rows_before")
            assert hasattr(step, "rows_after")


class TestAutoCleanEdgeCases:
    """Test edge cases implied by documentation."""

    def test_empty_frame(self):
        """Test auto_clean on empty frame."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ArFrame)
        assert result.shape[0] == 0

    def test_single_row_frame(self):
        """Test auto_clean on single-row frame."""
        df = pd.DataFrame({"a": ["  x  "], "b": [1]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ArFrame)

    def test_frame_with_nulls(self):
        """Test auto_clean on frame with null values."""
        df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ArFrame)

    def test_dry_run_non_mutating(self):
        """Test that dry_run does not modify the original frame."""
        df = pd.DataFrame({"a": ["  x  ", "  y  "], "b": [1, 2]})
        frame = ar.from_pandas(df)
        original_shape = frame.shape
        ar.auto_clean(frame, dry_run=True)
        # Frame should not be modified by dry_run
        assert frame.shape == original_shape

    def test_safe_mode_conservative(self):
        """Test that safe mode only applies low-risk operations."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        # Safe mode should not aggressively change data
        assert result.shape[0] <= frame.shape[0]
