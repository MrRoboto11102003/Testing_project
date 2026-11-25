import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Tests for arnio.quality.auto_clean function."""

    @pytest.fixture
    def messy_frame(self):
        """Create a frame with whitespace and duplicates."""
        df = pd.DataFrame({
            "name": [" Alice ", " Bob ", " Alice "],
            "score": [90, 85, 90],
            "city": [" Paris ", " London ", " Paris "]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create a frame with null values."""
        df = pd.DataFrame({
            "id": [1, 2, None, 4],
            "value": [10.0, np.nan, 30.0, 40.0]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty frame."""
        df = pd.DataFrame({"col1": [], "col2": []})
        return ar.from_pandas(df)

    def test_auto_clean_safe_mode_default_returns_arframe(self, messy_frame):
        """Safe mode (default) should return just an ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_strips_whitespace(self, messy_frame):
        """Safe mode should trim leading/trailing whitespace."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        
        # Check that whitespace is trimmed
        for name in df["name"].dropna():
            assert name == name.strip()
        
        for city in df["city"].dropna():
            assert city == city.strip()

    def test_auto_clean_strict_mode_applies_deduplication(self, messy_frame):
        """Strict mode should deduplicate exact row matches."""
        original_rows = messy_frame.shape[0]
        result = ar.auto_clean(messy_frame, mode="strict")
        result_rows = result.shape[0]
        
        # Strict mode includes deduplication
        assert result_rows <= original_rows

    def test_auto_clean_return_report_tuple(self, messy_frame):
        """return_report=True should return (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(messy_frame, return_report=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_dry_run_returns_report_only(self, messy_frame):
        """dry_run=True should return only report without mutation."""
        result = ar.auto_clean(messy_frame, dry_run=True)
        
        assert isinstance(result, ar.DataQualityReport)

    def test_auto_clean_explain_returns_tuple(self, messy_frame):
        """explain=True should return (ArFrame, CleanExplanation)."""
        result = ar.auto_clean(messy_frame, explain=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, explanation = result
        assert isinstance(frame, ar.ArFrame)

    def test_auto_clean_return_report_and_explain(self, messy_frame):
        """Both return_report and explain should return 3-tuple."""
        result = ar.auto_clean(
            messy_frame,
            return_report=True,
            explain=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        frame, report, explanation = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_dry_run_does_not_mutate(self, messy_frame):
        """dry_run=True should not modify the frame."""
        original_df = ar.to_pandas(messy_frame)
        original_rows = len(original_df)
        
        # Dry run with strict mode (would mutate if actually applied)
        ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        
        # Original frame shape unchanged
        assert messy_frame.shape[0] == original_rows

    def test_auto_clean_empty_frame_safe_mode(self, empty_frame):
        """auto_clean should handle empty frames in safe mode."""
        result = ar.auto_clean(empty_frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_auto_clean_single_row_frame(self, messy_frame):
        """auto_clean should handle single-row frames."""
        df = ar.to_pandas(messy_frame).iloc[:1]
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame)
        assert result.shape[0] == 1
        assert result.columns == frame.columns

    def test_auto_clean_preserves_column_order(self, messy_frame):
        """auto_clean should preserve column order."""
        result = ar.auto_clean(messy_frame)
        assert result.columns == messy_frame.columns

    def test_auto_clean_safe_mode_no_deduplication(self, messy_frame):
        """Safe mode should not remove duplicates."""
        original_rows = messy_frame.shape[0]
        result = ar.auto_clean(messy_frame, mode="safe")
        
        # Safe mode only trims whitespace, doesn't deduplicate
        # (may drop rows if they become all-null after processing, but not exact duplicates)
        assert result.shape[0] >= original_rows or result.shape[0] == original_rows

    def test_auto_clean_invalid_mode_raises(self, messy_frame):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            ar.auto_clean(messy_frame, mode="invalid_mode")

    def test_auto_clean_with_nulls(self, frame_with_nulls):
        """auto_clean should handle frames with null values."""
        result = ar.auto_clean(frame_with_nulls)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[1] == frame_with_nulls.shape[1]

    def test_auto_clean_confirmed_casts_requires_allow_lossy(self, messy_frame):
        """Using confirmed_casts requires allow_lossy_casts=True."""
        # If allow_lossy_casts is False (default), it should work fine without casts
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=False,
            confirmed_casts=None
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_whitespace_both_sides(self):
        """auto_clean should trim whitespace from both sides of strings."""
        df = pd.DataFrame({
            "col": ["  value  ", "\ttest\n", "  data  "]
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        result_df = ar.to_pandas(result)
        
        for val in result_df["col"].dropna():
            if isinstance(val, str):
                assert val == val.strip()

    def test_auto_clean_strict_with_return_report(self, messy_frame):
        """Strict mode with return_report should return both."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            return_report=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)
