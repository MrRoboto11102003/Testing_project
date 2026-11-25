import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Tests for arnio.quality.auto_clean function."""

    @pytest.fixture
    def messy_frame(self):
        """Create a messy ArFrame with whitespace and duplicates."""
        df = pd.DataFrame({
            "order_id": [1001, 1002, 1002, 1003, 1004],
            "customer": [" Ishan ", " Prasoon ", " Prasoon ", " Pranay ", " Dhruv "],
            "city": [" Paris ", "London", "London", " New York ", " Tokyo "],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def clean_frame(self):
        """Create a clean ArFrame with no quality issues."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.0],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame."""
        df = pd.DataFrame({"col1": [], "col2": []})
        return ar.from_pandas(df)

    def test_auto_clean_default_returns_arframe(self, messy_frame):
        """Test that auto_clean with defaults returns an ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)

    def test_auto_clean_safe_mode_strips_whitespace(self, messy_frame):
        """Test that safe mode strips whitespace from strings."""
        result = ar.auto_clean(messy_frame, mode="safe")
        result_df = ar.to_pandas(result)
        assert result_df["customer"].iloc[0] == "Ishan"
        assert result_df["city"].iloc[0] == "Paris"

    def test_auto_clean_strict_mode_removes_duplicates(self, messy_frame):
        """Test that strict mode removes exact duplicate rows."""
        result = ar.auto_clean(messy_frame, mode="strict")
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(messy_frame)
        assert len(result_df) < len(original_df)
        assert len(result_df) == 4  # 5 rows - 1 duplicate

    def test_auto_clean_with_return_report(self, messy_frame):
        """Test that return_report=True returns (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(messy_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_dry_run_returns_report(self, messy_frame):
        """Test that dry_run=True returns a DataQualityReport."""
        result = ar.auto_clean(messy_frame, dry_run=True)
        assert isinstance(result, ar.DataQualityReport)

    def test_auto_clean_on_clean_data_unchanged(self, clean_frame):
        """Test that auto_clean on clean data returns equivalent frame."""
        result = ar.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)
        original_df = ar.to_pandas(clean_frame)
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(original_df, result_df)

    def test_auto_clean_on_empty_frame(self, empty_frame):
        """Test auto_clean handles empty frames gracefully."""
        result = ar.auto_clean(empty_frame)
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["col1", "col2"]

    def test_auto_clean_preserves_column_names_and_order(self, messy_frame):
        """Test that column names and order are preserved."""
        result = ar.auto_clean(messy_frame, mode="strict")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["order_id", "customer", "city"]

    def test_auto_clean_strict_mode_with_return_report(self, messy_frame):
        """Test strict mode combined with return_report."""
        frame, report = ar.auto_clean(
            messy_frame,
            mode="strict",
            return_report=True
        )
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_with_numeric_data(self):
        """Test auto_clean on frame with numeric columns."""
        df = pd.DataFrame({
            "value": [1.5, 2.5, 2.5, 3.5],
            "count": [10, 20, 20, 30],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="strict")
        result_df = ar.to_pandas(result)
        assert result_df["value"].dtype in [np.float64, float]
        assert len(result_df) == 3  # One duplicate removed

    def test_auto_clean_with_null_values(self):
        """Test auto_clean preserves null handling appropriately."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", None, "Bob", "David"],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert result_df["name"].isna().sum() == 1

    def test_auto_clean_safe_vs_strict_row_count(self, messy_frame):
        """Test that strict mode produces fewer or equal rows than safe mode."""
        safe_result = ar.auto_clean(messy_frame, mode="safe")
        strict_result = ar.auto_clean(messy_frame, mode="strict")
        safe_df = ar.to_pandas(safe_result)
        strict_df = ar.to_pandas(strict_result)
        assert len(strict_df) <= len(safe_df)

    def test_auto_clean_mixed_column_types(self):
        """Test auto_clean with mixed column types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": [" hello ", " world ", " test "],
            "float_col": [1.1, 2.2, 3.3],
            "bool_col": [True, False, True],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert result_df["str_col"].iloc[0] == "hello"
        assert result_df["int_col"].dtype in [np.int64, int]

    def test_auto_clean_duplicate_detection_exact_match(self):
        """Test that duplicate detection finds exact row matches."""
        df = pd.DataFrame({
            "x": [1, 2, 2],
            "y": ["a", "b", "b"],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="strict")
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert list(result_df["x"]) == [1, 2]
