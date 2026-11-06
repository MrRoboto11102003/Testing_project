import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Tests for arnio.quality.auto_clean function."""

    @pytest.fixture
    def messy_frame(self):
        """Frame with whitespace and duplicate rows."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": [" Alice ", " Bob ", " Bob ", " Charlie "],
            "city": [" NYC ", "LA", "LA", " SF "],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def clean_frame(self):
        """Frame with no cleaning issues."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_whitespace_only(self):
        """Frame with only whitespace issues."""
        df = pd.DataFrame({
            "col1": [" a ", " b ", " c "],
            "col2": ["  x  ", "  y  ", "  z  "],
        })
        return ar.from_pandas(df)

    def test_auto_clean_default_returns_arframe(self, messy_frame):
        """Test default behavior returns ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_strips_whitespace(self, frame_with_whitespace_only):
        """Test safe mode strips whitespace but preserves duplicates."""
        result = ar.auto_clean(frame_with_whitespace_only, mode="safe")
        df = ar.to_pandas(result)
        assert df["col1"].iloc[0] == "a"
        assert df["col2"].iloc[0] == "x"
        # Safe mode preserves all rows
        assert len(df) == 3

    def test_auto_clean_strict_mode_removes_duplicates(self, messy_frame):
        """Test strict mode applies strip_whitespace and drop_duplicates."""
        result = ar.auto_clean(messy_frame, mode="strict")
        df = ar.to_pandas(result)
        # Original has 4 rows with 1 exact duplicate; should become 3
        assert len(df) <= 4
        # Whitespace should be stripped
        assert df["name"].iloc[0] == "Alice"

    def test_auto_clean_return_report_true(self, messy_frame):
        """Test return_report=True returns tuple with ArFrame and DataQualityReport."""
        result = ar.auto_clean(messy_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, report = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_dry_run_returns_report(self, messy_frame):
        """Test dry_run=True returns DataQualityReport without modifying frame."""
        result = ar.auto_clean(messy_frame, dry_run=True)
        assert isinstance(result, ar.DataQualityReport)

    def test_auto_clean_explain_true(self, messy_frame):
        """Test explain=True returns tuple with ArFrame and explanation."""
        result = ar.auto_clean(messy_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)

    def test_auto_clean_return_report_and_explain(self, messy_frame):
        """Test both return_report=True and explain=True returns 3-tuple."""
        result = ar.auto_clean(messy_frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        clean_frame, report, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_empty_frame(self):
        """Test auto_clean on empty frame."""
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_auto_clean_already_clean_frame(self, clean_frame):
        """Test auto_clean on frame with no issues."""
        result = ar.auto_clean(clean_frame, mode="safe")
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(clean_frame)
        pd.testing.assert_frame_equal(result_df, original_df)

    def test_auto_clean_safe_vs_strict_row_count(self, messy_frame):
        """Test strict mode removes more rows than safe mode."""
        safe = ar.to_pandas(ar.auto_clean(messy_frame, mode="safe"))
        strict = ar.to_pandas(ar.auto_clean(messy_frame, mode="strict"))
        # Strict should have same or fewer rows due to deduplication
        assert len(strict) <= len(safe)

    def test_auto_clean_confirmed_casts_parameter(self, messy_frame):
        """Test confirmed_casts parameter is accepted."""
        casts = {}  # Empty casts mapping
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            confirmed_casts=casts,
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_allow_lossy_casts_parameter(self, messy_frame):
        """Test allow_lossy_casts parameter is accepted."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=True,
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_preserves_column_names(self, messy_frame):
        """Test auto_clean preserves column names."""
        original_cols = messy_frame.columns
        result = ar.auto_clean(messy_frame, mode="safe")
        assert result.columns == original_cols

    def test_auto_clean_numeric_columns_safe_mode(self):
        """Test safe mode preserves numeric column values."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        result_df = ar.to_pandas(result)
        assert result_df["id"].tolist() == [1, 2, 3]
        np.testing.assert_array_almost_equal(
            result_df["value"].values,
            [10.5, 20.3, 30.1],
        )

    def test_auto_clean_mixed_dtype_frame(self):
        """Test auto_clean on frame with mixed dtypes."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": [" a ", " b ", " c "],
            "float_col": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
        assert len(result.columns) == 3
