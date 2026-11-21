import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfileBasic:
    """Test basic profiling functionality."""

    def test_profile_returns_valid_report(self):
        """Test that profile returns a valid DataQualityReport."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 2
        assert "a" in report.columns
        assert "b" in report.columns
        assert hasattr(report, "quality_score")

    def test_profile_with_exclude_columns(self):
        """Test excluding columns from profiling."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["b"])
        
        assert report.column_count == 2
        assert "a" in report.columns
        assert "c" in report.columns
        assert "b" not in report.columns

    def test_profile_detects_duplicates(self):
        """Test that profile detects duplicate rows."""
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 1
        assert report.duplicate_ratio > 0


class TestSampleSizeValidation:
    """Test sample_size parameter validation."""

    def test_sample_size_negative_raises_value_error(self):
        """Test that negative sample_size raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="non-negative"):
            ar.profile(frame, sample_size=-1)

    def test_sample_size_float_raises_type_error(self):
        """Test that float sample_size raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, sample_size=3.5)

    def test_sample_size_bool_raises_type_error(self):
        """Test that bool sample_size raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, sample_size=True)


class TestApproxTopValuesValidation:
    """Test approximate top values parameter validation."""

    def test_approx_top_values_non_bool_raises_type_error(self):
        """Test that non-bool approx_top_values raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values="yes")

    def test_approx_min_unique_float_raises_type_error(self):
        """Test that float approx_top_values_min_unique raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_min_unique=1000.5)

    def test_approx_min_unique_bool_raises_type_error(self):
        """Test that bool approx_top_values_min_unique raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_min_unique=True)

    def test_approx_min_unique_negative_raises_value_error(self):
        """Test that negative approx_top_values_min_unique raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="non-negative"):
            ar.profile(frame, approx_top_values_min_unique=-1)

    def test_approx_min_ratio_string_raises_type_error(self):
        """Test that string approx_top_values_min_ratio raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_min_ratio="0.5")

    def test_approx_min_ratio_bool_raises_type_error(self):
        """Test that bool approx_top_values_min_ratio raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_min_ratio=True)

    def test_approx_min_ratio_infinity_raises_value_error(self):
        """Test that infinite approx_top_values_min_ratio raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="finite"):
            ar.profile(frame, approx_top_values_min_ratio=np.inf)

    def test_approx_min_ratio_below_zero_raises_value_error(self):
        """Test that approx_top_values_min_ratio < 0 raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)

    def test_approx_min_ratio_above_one_raises_value_error(self):
        """Test that approx_top_values_min_ratio > 1 raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.1)

    def test_approx_sample_size_float_raises_type_error(self):
        """Test that float approx_top_values_sample_size raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_sample_size=2000.5)

    def test_approx_sample_size_bool_raises_type_error(self):
        """Test that bool approx_top_values_sample_size raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.profile(frame, approx_top_values_sample_size=True)

    def test_approx_sample_size_zero_raises_value_error(self):
        """Test that zero approx_top_values_sample_size raises ValueError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="positive"):
            ar.profile(frame, approx_top_values_sample_size=0)


class TestExcludeColumnsValidation:
    """Test exclude_columns parameter validation."""

    def test_exclude_nonexistent_column_raises_error(self):
        """Test that excluding non-existent column raises error."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(Exception):  # Raises ValueError for missing column
            ar.profile(frame, exclude_columns=["nonexistent"])

    def test_exclude_all_columns_valid(self):
        """Test profiling with all columns excluded."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["a", "b"])
        
        assert report.column_count == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_profile_empty_frame(self):
        """Test profiling an empty frame."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int)})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 1

    def test_profile_frame_with_all_nulls(self):
        """Test profiling a frame with all null values."""
        df = pd.DataFrame({"a": [None, None, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 1
