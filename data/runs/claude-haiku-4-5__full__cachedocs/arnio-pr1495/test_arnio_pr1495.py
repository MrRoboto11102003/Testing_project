import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfileBasic:
    """Test basic profiling functionality."""

    def test_profile_returns_data_quality_report(self):
        """Profile should return a DataQualityReport object."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert hasattr(report, "columns")
        assert hasattr(report, "quality_score")

    def test_profile_captures_correct_row_column_counts(self):
        """Profile should capture correct row and column counts."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.1, 2.2, 3.3]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report.row_count == 3
        assert report.column_count == 3

    def test_profile_creates_column_profiles_for_each_column(self):
        """Profile should create a ColumnProfile for each column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert len(report.columns) == 2
        assert "a" in report.columns
        assert "b" in report.columns
        assert hasattr(report.columns["a"], "dtype")
        assert hasattr(report.columns["a"], "null_count")

    def test_profile_detects_null_values_and_ratios(self):
        """Profile should detect null values and calculate ratios."""
        df = pd.DataFrame({
            "col1": [1, 2, None],
            "col2": [None, None, None, "c"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report.columns["col1"].null_count == 1
        assert report.columns["col2"].null_count == 3
        assert report.columns["col2"].null_ratio == 0.75

    def test_profile_counts_unique_values(self):
        """Profile should count unique non-null values."""
        df = pd.DataFrame({"col": [1, 1, 2, 3, 3, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report.columns["col"].unique_count == 3

    def test_profile_calculates_unique_ratio(self):
        """Profile should calculate unique ratio correctly."""
        df = pd.DataFrame({"col": ["a", "a", "b", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        # 3 unique / 5 non-null = 0.6
        assert report.columns["col"].unique_ratio == pytest.approx(0.6, abs=0.01)

    def test_profile_infers_dtypes_correctly(self):
        """Profile should infer column dtypes."""
        df = pd.DataFrame({
            "ints": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["a", "b", "c"],
            "bools": [True, False, True]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report.columns["ints"].dtype == "int64"
        assert report.columns["floats"].dtype == "float64"
        assert report.columns["strings"].dtype == "string"
        assert report.columns["bools"].dtype == "bool"

    def test_profile_respects_sample_size_parameter(self):
        """Profile should respect the sample_size parameter."""
        df = pd.DataFrame({"col": list(range(100))})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=3)
        col_profile = report.columns["col"]
        assert hasattr(col_profile, "sample_values")
        if col_profile.sample_values:
            assert len(col_profile.sample_values) <= 3

    def test_profile_excludes_specified_columns(self):
        """Profile should exclude columns in exclude_columns."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["col2"])
        assert "col1" in report.columns
        assert "col2" not in report.columns
        assert "col3" in report.columns
        assert report.column_count == 2

    def test_profile_summary_method_returns_dict(self):
        """Profile.summary() should return a dictionary."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_profile_includes_numeric_statistics(self):
        """Profile should include numeric statistics for numeric columns."""
        df = pd.DataFrame({"age": [20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col_profile = report.columns["age"]
        assert hasattr(col_profile, "min")
        assert hasattr(col_profile, "max")
        assert hasattr(col_profile, "mean")
        assert col_profile.min == 20.0
        assert col_profile.max == 50.0
        assert col_profile.mean == pytest.approx(35.0, abs=0.1)

    def test_profile_quality_score_is_valid(self):
        """Profile quality_score should be a number between 0 and 100."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert hasattr(report, "quality_score")
        assert isinstance(report.quality_score, (int, float))
        assert 0 <= report.quality_score <= 100

    def test_profile_handles_empty_frame(self):
        """Profile should handle frames with 0 rows."""
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report.row_count == 0
        assert report.column_count == 1

    def test_profile_handles_all_nulls_column(self):
        """Profile should handle columns where all values are null."""
        df = pd.DataFrame({"col": [None, None, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col_profile = report.columns["col"]
        assert col_profile.null_count == 3
        assert col_profile.null_ratio == 1.0
        assert col_profile.unique_count == 0

    def test_profile_detects_duplicate_rows(self):
        """Profile should track duplicate row information."""
        df = pd.DataFrame({
            "col1": [1, 1, 2],
            "col2": [4, 4, 6]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert hasattr(report, "duplicate_rows")
        assert hasattr(report, "duplicate_ratio")
        assert report.duplicate_rows >= 0

    def test_profile_supports_approx_top_values_parameter(self):
        """Profile should support approximating top values for high-cardinality columns."""
        df = pd.DataFrame({
            "col": [f"val_{i % 500}" for i in range(5000)]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=100,
            approx_top_values_sample_size=1000
        )
        assert report.row_count == 5000
        col_profile = report.columns["col"]
        assert col_profile is not None
