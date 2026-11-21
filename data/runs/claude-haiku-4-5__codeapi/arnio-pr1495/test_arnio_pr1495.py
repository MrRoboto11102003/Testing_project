import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import profile, DataQualityReport


class TestProfileBasicFunctionality:
    """Test basic profiling functionality."""

    def test_profile_returns_data_quality_report(self):
        """Test that profile returns a DataQualityReport instance."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)

        result = profile(frame)

        assert isinstance(result, DataQualityReport)
        assert result.row_count == 3
        assert result.column_count == 2

    def test_profile_with_null_values(self):
        """Test profiling frame with null values."""
        df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, "z"]})
        frame = ar.from_pandas(df)

        report = profile(frame)

        assert report.columns["a"].null_count == 1
        assert report.columns["a"].null_ratio == pytest.approx(1 / 3)
        assert report.columns["b"].null_count == 1

    def test_profile_with_duplicates(self):
        """Test profiling frame with duplicate rows."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        frame = ar.from_pandas(df)

        report = profile(frame)

        assert report.duplicate_rows == 1
        assert report.duplicate_ratio == pytest.approx(1 / 3)

    def test_profile_empty_frame(self):
        """Test profiling an empty frame."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)

        report = profile(frame)

        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0

    def test_profile_all_nulls(self):
        """Test profiling column with all null values."""
        df = pd.DataFrame({"a": [None, None, None]})
        frame = ar.from_pandas(df)

        report = profile(frame)

        assert report.columns["a"].null_count == 3
        assert report.columns["a"].null_ratio == 1.0


class TestProfileExcludeColumns:
    """Test exclude_columns parameter."""

    def test_exclude_single_column(self):
        """Test excluding a single column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        frame = ar.from_pandas(df)

        report = profile(frame, exclude_columns=["b"])

        assert report.column_count == 2
        assert "a" in report.columns
        assert "c" in report.columns
        assert "b" not in report.columns

    def test_exclude_columns_nonexistent_raises_error(self):
        """Test that excluding non-existent column raises error."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        frame = ar.from_pandas(df)

        with pytest.raises(Exception):
            profile(frame, exclude_columns=["nonexistent"])


class TestProfileParameterValidation:
    """Test parameter validation."""

    def test_sample_size_bool_raises_type_error(self):
        """Test sample_size as bool raises TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError):
            profile(frame, sample_size=True)

    def test_sample_size_negative_raises_value_error(self):
        """Test negative sample_size raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            profile(frame, sample_size=-1)

    def test_approx_top_values_non_bool_raises_type_error(self):
        """Test approx_top_values with non-bool raises TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError):
            profile(frame, approx_top_values="true")

    def test_approx_top_values_min_unique_bool_raises_type_error(self):
        """Test approx_top_values_min_unique as bool raises TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError):
            profile(frame, approx_top_values_min_unique=True)

    def test_approx_top_values_min_ratio_below_zero_raises_value_error(self):
        """Test approx_top_values_min_ratio < 0 raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            profile(frame, approx_top_values_min_ratio=-0.1)

    def test_approx_top_values_min_ratio_above_one_raises_value_error(self):
        """Test approx_top_values_min_ratio > 1 raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            profile(frame, approx_top_values_min_ratio=1.1)

    def test_approx_top_values_min_ratio_nan_raises_value_error(self):
        """Test approx_top_values_min_ratio=NaN raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            profile(frame, approx_top_values_min_ratio=float("nan"))

    def test_approx_top_values_sample_size_zero_raises_value_error(self):
        """Test approx_top_values_sample_size=0 raises ValueError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            profile(frame, approx_top_values_sample_size=0)


class TestProfileReportStructure:
    """Test the structure of the returned DataQualityReport."""

    def test_report_has_quality_score_and_suggestions(self):
        """Test that report includes quality_score and suggestions."""
        df = pd.DataFrame({"a": [1, None, None]})
        frame = ar.from_pandas(df)

        report = profile(frame)

        assert hasattr(report, "quality_score")
        assert 0 <= report.quality_score <= 100
        assert hasattr(report, "suggestions")
        assert isinstance(report.suggestions, list)

    def test_column_profile_attributes(self):
        """Test that column profiles include expected attributes."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)

        report = profile(frame)
        col_profile = report.columns["a"]

        assert col_profile.name == "a"
        assert col_profile.row_count == 5
        assert col_profile.null_count == 0
        assert col_profile.unique_count == 5
