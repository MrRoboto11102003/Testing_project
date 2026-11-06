import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfile:
    """Test suite for ar.profile() function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a basic sample ArFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95.5, 87.0, 92.5, 88.0, 95.5],
            "active": [True, False, True, True, False],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values."""
        df = pd.DataFrame({
            "id": [1.0, 2.0, np.nan, 4.0, 5.0],
            "name": ["Alice", None, "Charlie", "David", None],
            "score": [95.5, 87.0, np.nan, 88.0, 95.5],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_duplicates(self):
        """Create an ArFrame with duplicate rows."""
        df = pd.DataFrame({
            "id": [1, 2, 1, 2, 3],
            "name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
        })
        return ar.from_pandas(df)

    def test_profile_basic_returns_report(self, sample_frame):
        """Test basic profile with default parameters returns DataQualityReport."""
        report = ar.profile(sample_frame)
        
        assert report.row_count == 5
        assert report.column_count == 4
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0
        assert isinstance(report.columns, dict)
        assert len(report.columns) == 4
        assert all(col in report.columns for col in ["id", "name", "score", "active"])

    def test_profile_structure_has_all_fields(self, sample_frame):
        """Test that returned report contains all documented fields."""
        report = ar.profile(sample_frame)
        
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert hasattr(report, "memory_usage")
        assert hasattr(report, "duplicate_rows")
        assert hasattr(report, "duplicate_ratio")
        assert hasattr(report, "columns")
        assert hasattr(report, "quality_score")
        assert hasattr(report, "score_components")
        assert hasattr(report, "suggestions")

    def test_profile_with_nulls(self, frame_with_nulls):
        """Test profile correctly counts null values."""
        report = ar.profile(frame_with_nulls)
        
        assert report.row_count == 5
        assert report.column_count == 3
        assert report.columns["id"].null_count == 1
        assert report.columns["name"].null_count == 2
        assert report.columns["score"].null_count == 1

    def test_profile_with_duplicates(self, frame_with_duplicates):
        """Test profile detects duplicate rows."""
        report = ar.profile(frame_with_duplicates)
        
        assert report.row_count == 5
        assert report.duplicate_rows == 2
        assert report.duplicate_ratio > 0.0

    def test_profile_exclude_columns(self, sample_frame):
        """Test excluding specific columns from profile."""
        report = ar.profile(sample_frame, exclude_columns=["score", "active"])
        
        assert report.column_count == 2
        assert "id" in report.columns
        assert "name" in report.columns
        assert "score" not in report.columns
        assert "active" not in report.columns

    def test_profile_exclude_nonexistent_column_raises(self, sample_frame):
        """Test that excluding non-existent columns raises ValueError."""
        with pytest.raises(ValueError):
            ar.profile(sample_frame, exclude_columns=["nonexistent"])

    def test_profile_sample_size_zero(self, sample_frame):
        """Test profile works with sample_size=0."""
        report = ar.profile(sample_frame, sample_size=0)
        
        assert report.row_count == 5
        assert report.column_count == 4

    def test_profile_sample_size_negative_raises(self, sample_frame):
        """Test that negative sample_size raises ValueError."""
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(sample_frame, sample_size=-1)

    def test_profile_sample_size_non_int_raises(self, sample_frame):
        """Test that non-integer sample_size raises TypeError."""
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(sample_frame, sample_size=3.5)

    def test_profile_approx_top_values_not_bool_raises(self, sample_frame):
        """Test that non-bool approx_top_values raises TypeError."""
        with pytest.raises(TypeError, match="approx_top_values must be a bool"):
            ar.profile(sample_frame, approx_top_values=1)

    def test_profile_approx_top_values_min_ratio_out_of_range(self, sample_frame):
        """Test that approx_top_values_min_ratio outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ar.profile(sample_frame, approx_top_values_min_ratio=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ar.profile(sample_frame, approx_top_values_min_ratio=-0.1)

    def test_profile_approx_top_values_sample_size_zero_raises(self, sample_frame):
        """Test that non-positive approx_top_values_sample_size raises ValueError."""
        with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
            ar.profile(sample_frame, approx_top_values_sample_size=0)

    def test_profile_with_approximation_enabled(self, sample_frame):
        """Test profile with approximation features enabled."""
        report = ar.profile(
            sample_frame,
            approx_top_values=True,
            approx_top_values_min_unique=1,
            approx_top_values_sample_size=100,
        )
        
        assert report.row_count == 5
        assert report.column_count == 4

    def test_profile_empty_frame(self):
        """Test profile on empty ArFrame."""
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 1
        assert report.duplicate_ratio == 0.0

    def test_profile_includes_suggestions(self, frame_with_nulls):
        """Test that profile includes cleaning suggestions."""
        report = ar.profile(frame_with_nulls)
        
        assert isinstance(report.suggestions, list)
