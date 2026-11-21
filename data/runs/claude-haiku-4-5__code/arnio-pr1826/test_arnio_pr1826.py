import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfile:
    """Tests for arnio.quality.profile function."""

    @pytest.fixture
    def simple_frame(self):
        """Create a simple test frame."""
        df = pd.DataFrame({
            'col_a': [1, 2, 3, 4, 5],
            'col_b': ['a', 'b', 'c', 'd', 'e'],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create a frame with null values."""
        df = pd.DataFrame({
            'col_a': [1, 2, None, 4, 5],
            'col_b': ['a', None, 'c', 'd', 'e'],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_duplicates(self):
        """Create a frame with duplicate rows."""
        df = pd.DataFrame({
            'col_a': [1, 1, 2, 2, 3],
            'col_b': ['a', 'a', 'b', 'b', 'c'],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty frame."""
        df = pd.DataFrame({'col_a': [], 'col_b': []})
        return ar.from_pandas(df)

    @pytest.fixture
    def high_cardinality_frame(self):
        """Create a frame with high-cardinality string column."""
        df = pd.DataFrame({
            'id': range(2500),
            'value': [f'val_{i}' for i in range(2500)],
        })
        return ar.from_pandas(df)

    def test_profile_basic(self, simple_frame):
        """Test basic profiling of a simple frame."""
        report = ar.quality.profile(simple_frame)
        assert report.row_count == 5
        assert report.column_count == 2
        assert 'col_a' in report.columns
        assert 'col_b' in report.columns

    def test_profile_return_type(self, simple_frame):
        """Test that profile returns correct structure."""
        report = ar.quality.profile(simple_frame)
        assert hasattr(report, 'row_count')
        assert hasattr(report, 'column_count')
        assert hasattr(report, 'columns')
        assert hasattr(report, 'quality_score')

    def test_profile_with_nulls(self, frame_with_nulls):
        """Test profiling a frame with null values."""
        report = ar.quality.profile(frame_with_nulls)
        assert report.row_count == 5
        assert len(report.columns) == 2

    def test_profile_with_duplicates(self, frame_with_duplicates):
        """Test profiling a frame with duplicate rows."""
        report = ar.quality.profile(frame_with_duplicates)
        assert report.row_count == 5
        assert report.duplicate_rows >= 0

    def test_profile_exclude_columns(self, simple_frame):
        """Test excluding columns from profiling."""
        report = ar.quality.profile(simple_frame, exclude_columns=['col_b'])
        assert report.column_count == 1
        assert 'col_a' in report.columns
        assert 'col_b' not in report.columns

    def test_profile_exclude_nonexistent_column(self, simple_frame):
        """Test that excluding nonexistent column raises error."""
        with pytest.raises(Exception):
            ar.quality.profile(simple_frame, exclude_columns=['nonexistent'])

    def test_profile_empty_frame(self, empty_frame):
        """Test profiling an empty frame."""
        report = ar.quality.profile(empty_frame)
        assert report.row_count == 0
        assert report.column_count == 2

    def test_profile_sample_size_type_validation(self, simple_frame):
        """Test that sample_size must be int, not float or bool."""
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, sample_size=3.5)
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, sample_size=True)

    def test_profile_sample_size_value_validation(self, simple_frame):
        """Test that sample_size must be non-negative."""
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, sample_size=-1)

    def test_profile_approx_top_values_type_validation(self, simple_frame):
        """Test that approx_top_values must be bool."""
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, approx_top_values="true")

    def test_profile_approx_top_values_min_unique_validation(self, simple_frame):
        """Test approx_top_values_min_unique must be int and non-negative."""
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, approx_top_values_min_unique=True)
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_min_unique=-1)

    def test_profile_approx_top_values_min_ratio_validation(self, simple_frame):
        """Test approx_top_values_min_ratio must be finite number between 0-1."""
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, approx_top_values_min_ratio=True)
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_min_ratio=float('inf'))
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_min_ratio=1.5)
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_min_ratio=-0.1)

    def test_profile_approx_top_values_sample_size_validation(self, simple_frame):
        """Test approx_top_values_sample_size must be positive int."""
        with pytest.raises(TypeError):
            ar.quality.profile(simple_frame, approx_top_values_sample_size=True)
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_sample_size=0)
        with pytest.raises(ValueError):
            ar.quality.profile(simple_frame, approx_top_values_sample_size=-1)

    def test_profile_approx_top_values_enabled(self, high_cardinality_frame):
        """Test profiling with approx_top_values enabled."""
        report = ar.quality.profile(
            high_cardinality_frame,
            approx_top_values=True,
            approx_top_values_min_unique=100,
            approx_top_values_min_ratio=0.1,
            approx_top_values_sample_size=500,
        )
        assert report.row_count == 2500
        assert report.column_count == 2

    def test_profile_sample_size_zero(self, simple_frame):
        """Test that sample_size=0 is valid."""
        report = ar.quality.profile(simple_frame, sample_size=0)
        assert report is not None
