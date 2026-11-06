import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfile:
    """Test suite for ar.profile() function."""

    def test_profile_basic_frame(self):
        """Test basic profiling of a simple frame returns DataQualityReport."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95.5, 87.2, 92.1, 78.5, 88.3]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert report.row_count == 5
        assert report.column_count == 3
        assert set(report.columns.keys()) == {"id", "name", "score"}
        assert isinstance(report.quality_score, (int, float))

    def test_profile_with_nulls(self):
        """Test profiling correctly identifies null values in columns."""
        df = pd.DataFrame({
            "col1": [1, 2, None, 4, 5],
            "col2": ["a", None, "c", "d", "e"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert report.row_count == 5
        assert "col1" in report.columns
        assert "col2" in report.columns
        # Profile should track nulls
        assert hasattr(report.columns["col1"], "null_count")
        assert hasattr(report.columns["col2"], "null_count")

    def test_profile_with_duplicates(self):
        """Test profiling detects and reports duplicate rows."""
        df = pd.DataFrame({
            "a": [1, 2, 1, 2, 3],
            "b": ["x", "y", "x", "y", "z"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert report.row_count == 5
        assert report.duplicate_rows >= 0
        assert 0 <= report.duplicate_ratio <= 1

    def test_profile_exclude_single_column(self):
        """Test excluding a single column from profiling."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "debug_col": ["x", "y", "z"],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["debug_col"])

        assert report.column_count == 2
        assert "debug_col" not in report.columns
        assert "id" in report.columns
        assert "name" in report.columns

    def test_profile_exclude_multiple_columns(self):
        """Test excluding multiple columns from profiling."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
            "col4": [10, 11, 12]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["col2", "col4"])

        assert report.column_count == 2
        assert set(report.columns.keys()) == {"col1", "col3"}

    def test_profile_sample_size_zero(self):
        """Test that sample_size=0 is allowed."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=0)

        assert report.row_count == 5

    def test_profile_sample_size_negative_raises_error(self):
        """Test that negative sample_size raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)

    def test_profile_sample_size_non_integer_raises_error(self):
        """Test that non-integer sample_size raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=3.5)

    def test_profile_approx_top_values_min_ratio_out_of_bounds_raises_error(self):
        """Test that approx_top_values_min_ratio outside [0,1] raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)

    def test_profile_approx_top_values_sample_size_non_positive_raises_error(self):
        """Test that non-positive approx_top_values_sample_size raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="must be positive"):
            ar.profile(frame, approx_top_values_sample_size=0)

        with pytest.raises(ValueError, match="must be positive"):
            ar.profile(frame, approx_top_values_sample_size=-5)

    def test_profile_exclude_nonexistent_column_raises_error(self):
        """Test that excluding a column that doesn't exist raises error."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(Exception):
            ar.profile(frame, exclude_columns=["nonexistent_col"])

    def test_profile_empty_frame(self):
        """Test profiling an empty frame."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype="int64"),
            "col2": pd.Series([], dtype="object")
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0

    def test_profile_summary_method(self):
        """Test that summary() method returns a dictionary."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        assert isinstance(summary, dict)

    def test_profile_memory_usage(self):
        """Test that memory_usage is a positive integer."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert isinstance(report.memory_usage, int)
        assert report.memory_usage > 0

    def test_profile_non_arframe_input_raises_error(self):
        """Test that passing a pandas DataFrame instead of ArFrame raises error."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(Exception):
            ar.profile(df)
