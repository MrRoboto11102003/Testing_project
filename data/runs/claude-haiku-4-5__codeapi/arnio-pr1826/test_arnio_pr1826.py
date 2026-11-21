import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import DataQualityReport, ColumnProfile


class TestProfile:
    """Test suite for arnio.quality.profile function."""

    @pytest.fixture
    def simple_frame(self):
        """Create a simple test ArFrame."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [95.5, 87.3, 92.1, 88.9, 91.2]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values."""
        df = pd.DataFrame({
            'id': [1, 2, None, 4, 5],
            'name': ['Alice', None, 'Charlie', 'David', 'Eve'],
            'score': [95.5, 87.3, None, 88.9, 91.2]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_duplicates(self):
        """Create an ArFrame with duplicate rows."""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3],
            'name': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie'],
            'value': [10, 10, 20, 20, 30]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame."""
        df = pd.DataFrame({'col1': pd.Series([], dtype='int64'), 'col2': pd.Series([], dtype='object')})
        return ar.from_pandas(df)

    def test_profile_basic_returns_report(self, simple_frame):
        """Test that profile returns a DataQualityReport."""
        report = ar.profile(simple_frame)
        assert isinstance(report, DataQualityReport)
        assert report.row_count == 5
        assert report.column_count == 3
        assert report.duplicate_rows == 0

    def test_profile_with_nulls_counts_correctly(self, frame_with_nulls):
        """Test that profile correctly counts null values."""
        report = ar.profile(frame_with_nulls)
        assert report.row_count == 5
        assert report.columns['id'].null_count == 1
        assert report.columns['name'].null_count == 1
        assert report.columns['score'].null_count == 1

    def test_profile_detects_duplicates(self, frame_with_duplicates):
        """Test that profile detects duplicate rows."""
        report = ar.profile(frame_with_duplicates)
        assert report.duplicate_rows == 2
        assert report.duplicate_ratio > 0.0

    def test_profile_empty_frame(self, empty_frame):
        """Test profiling an empty frame."""
        report = ar.profile(empty_frame)
        assert report.row_count == 0
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0

    def test_profile_exclude_columns_functionality(self, simple_frame):
        """Test that exclude_columns parameter works correctly."""
        report = ar.profile(simple_frame, exclude_columns=['score'])
        assert 'id' in report.columns
        assert 'name' in report.columns
        assert 'score' not in report.columns
        assert report.column_count == 2

    def test_profile_exclude_nonexistent_column_raises(self, simple_frame):
        """Test that excluding non-existent column raises an error."""
        with pytest.raises(Exception):
            ar.profile(simple_frame, exclude_columns=['nonexistent'])

    def test_profile_sample_size_negative_raises_error(self, simple_frame):
        """Test that negative sample_size raises ValueError."""
        with pytest.raises(ValueError):
            ar.profile(simple_frame, sample_size=-1)

    def test_profile_sample_size_non_integer_raises_error(self, simple_frame):
        """Test that non-integer sample_size raises TypeError."""
        with pytest.raises(TypeError):
            ar.profile(simple_frame, sample_size=5.5)

    def test_profile_sample_size_bool_raises_error(self, simple_frame):
        """Test that bool sample_size raises TypeError."""
        with pytest.raises(TypeError):
            ar.profile(simple_frame, sample_size=True)

    def test_profile_approx_top_values_ratio_bounds(self, simple_frame):
        """Test that approx_top_values_min_ratio must be between 0 and 1."""
        with pytest.raises(ValueError):
            ar.profile(simple_frame, approx_top_values_min_ratio=1.5)
        with pytest.raises(ValueError):
            ar.profile(simple_frame, approx_top_values_min_ratio=-0.1)

    def test_profile_approx_top_values_sample_size_positive(self, simple_frame):
        """Test that approx_top_values_sample_size must be positive."""
        with pytest.raises(ValueError):
            ar.profile(simple_frame, approx_top_values_sample_size=0)
        with pytest.raises(ValueError):
            ar.profile(simple_frame, approx_top_values_sample_size=-1)

    def test_profile_quality_score_in_valid_range(self, simple_frame):
        """Test that quality_score is between 0 and 100."""
        report = ar.profile(simple_frame)
        assert isinstance(report.quality_score, (int, float))
        assert 0 <= report.quality_score <= 100

    def test_profile_report_structure_and_columns(self, simple_frame):
        """Test that report has expected structure with column profiles."""
        report = ar.profile(simple_frame)
        
        assert hasattr(report, 'row_count')
        assert hasattr(report, 'column_count')
        assert hasattr(report, 'memory_usage')
        assert hasattr(report, 'duplicate_rows')
        assert hasattr(report, 'duplicate_ratio')
        assert hasattr(report, 'quality_score')
        assert hasattr(report, 'columns')
        assert hasattr(report, 'suggestions')
        
        assert isinstance(report.columns, dict)
        for col_name, col_profile in report.columns.items():
            assert isinstance(col_profile, ColumnProfile)
            assert col_profile.name == col_name
            assert col_profile.dtype is not None
            assert col_profile.null_ratio >= 0.0
            assert col_profile.unique_ratio >= 0.0
