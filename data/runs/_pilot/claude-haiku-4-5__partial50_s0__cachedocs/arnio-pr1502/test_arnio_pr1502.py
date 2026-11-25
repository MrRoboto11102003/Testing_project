import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Test suite for arnio.quality.auto_clean function."""

    @pytest.fixture
    def messy_frame(self):
        """Sample messy frame with whitespace and duplicates."""
        df = pd.DataFrame({
            "name": [" Alice ", " Bob ", " Alice "],
            "age": ["30", "25", "30"],
            "city": [" NYC ", "LA", " NYC "],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def clean_frame(self):
        """Sample clean frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Frame with null values."""
        df = pd.DataFrame({
            "col1": ["a", None, "c"],
            "col2": [1.0, 2.0, None],
        })
        return ar.from_pandas(df)

    def test_auto_clean_safe_mode_returns_frame(self, messy_frame):
        """Safe mode should return an ArFrame."""
        result = ar.auto_clean(messy_frame, mode="safe")
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_strict_mode_returns_frame(self, messy_frame):
        """Strict mode should return an ArFrame."""
        result = ar.auto_clean(messy_frame, mode="strict")
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_with_return_report(self, messy_frame):
        """return_report=True should return tuple of (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(messy_frame, mode="safe", return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert isinstance(frame, ar.ArFrame)
        # DataQualityReport has summary method
        assert callable(getattr(report, 'summary', None))

    def test_auto_clean_dry_run_returns_report(self, messy_frame):
        """dry_run=True should return report/metadata for inspection."""
        result = ar.auto_clean(messy_frame, mode="safe", dry_run=True)
        # dry_run should return a report or DataQualityReport
        assert result is not None
        # Should not be an ArFrame (no transformation applied)
        assert not isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_strips_whitespace(self, messy_frame):
        """Safe mode should strip leading/trailing whitespace."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        # Whitespace should be removed from string columns
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"
        assert df["city"].iloc[0] == "NYC"

    def test_auto_clean_safe_mode_preserves_duplicates(self, messy_frame):
        """Safe mode should NOT remove duplicates (only strip whitespace)."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        # All 3 rows should remain (includes duplicate Alice rows)
        assert len(df) == 3

    def test_auto_clean_strict_mode_removes_duplicates(self, messy_frame):
        """Strict mode should remove duplicate rows."""
        result = ar.auto_clean(messy_frame, mode="strict")
        df = ar.to_pandas(result)
        # Should have fewer rows due to duplicate removal + whitespace strip
        assert len(df) < 3

    def test_auto_clean_invalid_mode_raises_error(self, messy_frame):
        """Invalid mode should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ar.auto_clean(messy_frame, mode="invalid_mode")

    def test_auto_clean_empty_frame(self):
        """Auto clean should handle empty frames gracefully."""
        df = pd.DataFrame({"col": pd.Series([], dtype=str)})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_auto_clean_with_null_values(self, frame_with_nulls):
        """Auto clean should handle null values gracefully."""
        result = ar.auto_clean(frame_with_nulls, mode="safe")
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        # Nulls should be preserved in safe mode
        assert df.isna().sum().sum() > 0

    def test_auto_clean_allow_lossy_casts(self, messy_frame):
        """allow_lossy_casts=True should permit lossy type conversions."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=True
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_explain_flag(self, messy_frame):
        """explain=True should return tuple with explanation."""
        result = ar.auto_clean(messy_frame, mode="safe", explain=True)
        assert isinstance(result, tuple)
        frame, explanation = result
        assert isinstance(frame, ar.ArFrame)
        # explanation should be an object/dict with change information
        assert explanation is not None

    def test_auto_clean_return_report_and_explain(self, messy_frame):
        """Combining return_report and explain should return 3-tuple."""
        result = ar.auto_clean(
            messy_frame,
            mode="safe",
            return_report=True,
            explain=True
        )
        assert isinstance(result, tuple)
        # Should be (ArFrame, DataQualityReport, CleanExplanation)
        assert len(result) == 3
        frame, report, explanation = result
        assert isinstance(frame, ar.ArFrame)
        assert callable(getattr(report, 'summary', None))
        assert explanation is not None

    def test_auto_clean_safe_preserves_data_integrity(self, clean_frame):
        """Safe mode on clean data should not lose rows."""
        original_rows = clean_frame.shape[0]
        result = ar.auto_clean(clean_frame, mode="safe")
        assert result.shape[0] == original_rows

    def test_auto_clean_confirmed_casts_parameter(self, messy_frame):
        """confirmed_casts parameter should allow explicit type mappings."""
        # Only test that the parameter is accepted without error
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={"age": "int64"}
        )
        assert isinstance(result, ar.ArFrame)
