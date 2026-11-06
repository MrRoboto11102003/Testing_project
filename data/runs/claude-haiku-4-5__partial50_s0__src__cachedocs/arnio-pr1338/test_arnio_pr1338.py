import pytest
import pandas as pd
import arnio as ar


class TestAutoCleanBasic:
    """Test basic auto_clean functionality."""

    def test_auto_clean_default_returns_frame(self):
        """Test that default auto_clean returns ArFrame."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_strips_whitespace(self):
        """Test safe mode strips leading/trailing whitespace."""
        frame = ar.from_dict({"name": [" alice ", " bob "], "city": ["  ny  ", "la"]})
        result = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "alice"
        assert df["name"].iloc[1] == "bob"
        assert df["city"].iloc[0] == "ny"


class TestAutoCleanDryRun:
    """Test dry_run parameter behavior."""

    def test_dry_run_returns_report(self):
        """Test that dry_run=True returns DataQualityReport."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame, dry_run=True)
        assert isinstance(result, ar.DataQualityReport)

    def test_dry_run_preserves_original_frame(self):
        """Test that dry_run does not modify frame."""
        frame = ar.from_dict({"name": [" alice ", " bob "]})
        original_shape = frame.shape
        ar.auto_clean(frame, dry_run=True)
        assert frame.shape == original_shape


class TestAutoCleanReturnReport:
    """Test return_report parameter."""

    def test_return_report_returns_tuple(self):
        """Test return_report=True returns tuple of (frame, report)."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, report = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)


class TestAutoCleanExplain:
    """Test explain parameter."""

    def test_explain_returns_tuple_with_frame_and_explanation(self):
        """Test explain=True returns tuple with frame and explanation."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)

    def test_explain_and_return_report_returns_triple(self):
        """Test both explain and return_report flags return 3-tuple."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame, explain=True, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        clean_frame, report, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)


class TestAutoCleanValidation:
    """Test parameter validation and error handling."""

    def test_invalid_mode_raises_valueerror(self):
        """Test invalid mode raises ValueError."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame, mode="invalid")

    def test_dry_run_with_return_report_raises_valueerror(self):
        """Test incompatible dry_run and return_report flags."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)

    def test_dry_run_with_explain_raises_valueerror(self):
        """Test incompatible dry_run and explain flags."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_return_report_type_validation(self):
        """Test return_report must be bool."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame, return_report="yes")

    def test_dry_run_type_validation(self):
        """Test dry_run must be bool."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(frame, dry_run=1)

    def test_allow_lossy_casts_type_validation(self):
        """Test allow_lossy_casts must be bool."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            ar.auto_clean(frame, allow_lossy_casts="yes")

    def test_explain_type_validation(self):
        """Test explain must be bool."""
        frame = ar.from_dict({"col": ["value"]})
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(frame, explain=0)


class TestAutoCleanModes:
    """Test safe vs strict mode behavior."""

    def test_safe_mode_only_strips_whitespace(self):
        """Test safe mode applies minimal cleaning."""
        frame = ar.from_dict({"name": [" alice ", " bob "]})
        original_rows = frame.shape[0]
        result = ar.auto_clean(frame, mode="safe")
        # Safe mode should not remove rows
        assert result.shape[0] == original_rows

    def test_strict_mode_applies_more_cleaning(self):
        """Test strict mode can remove more rows than safe mode."""
        # Create frame with duplicates
        frame = ar.from_dict({
            "name": [" alice ", " alice ", " bob "],
            "age": ["30", "30", "25"]
        })
        safe_result = ar.auto_clean(frame, mode="safe")
        strict_result = ar.auto_clean(frame, mode="strict")
        # Strict mode may remove duplicates, potentially fewer rows
        assert strict_result.shape[0] <= safe_result.shape[0]


class TestAutoCleanEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_frame(self):
        """Test auto_clean with empty frame."""
        frame = ar.from_dict({"col": []})
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_single_row_frame(self):
        """Test auto_clean with single row."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame)
        assert result.shape[0] == 1
        assert result.shape[1] == 1

    def test_multiple_columns_with_mixed_whitespace(self):
        """Test cleaning multiple columns with different whitespace."""
        frame = ar.from_dict({
            "col1": [" a ", "b"],
            "col2": ["  c  ", "d  "],
            "col3": ["e", "  f"]
        })
        result = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(result)
        assert df["col1"].iloc[0] == "a"
        assert df["col2"].iloc[0] == "c"
        assert df["col3"].iloc[0] == "e"
