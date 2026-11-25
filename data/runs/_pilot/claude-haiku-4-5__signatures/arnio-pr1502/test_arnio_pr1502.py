import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Tests for arnio.quality.auto_clean function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with data quality issues."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
            'age': [25, 30, 35, None, 28],
            'salary': [50000.5, 60000.0, None, 75000.5, 55000.0],
            'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
            'value': ['1', '2', '3', '4', '5']
        })
        return ar.frame(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame."""
        return ar.frame(pd.DataFrame())

    @pytest.fixture
    def frame_with_nulls(self):
        """Create a frame with various null values."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', None, 'c', 'd', None],
            'col3': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
        return ar.frame(df)

    def test_auto_clean_default_returns_frame(self, sample_frame):
        """Test that auto_clean with defaults returns an ArFrame, not a tuple."""
        result = ar.quality.auto_clean(sample_frame)
        assert not isinstance(result, tuple)

    def test_auto_clean_with_return_report(self, sample_frame):
        """Test that return_report=True returns a tuple of (frame, report)."""
        result = ar.quality.auto_clean(sample_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_auto_clean_with_explain(self, sample_frame):
        """Test that explain=True returns a tuple of (frame, explanation)."""
        result = ar.quality.auto_clean(sample_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_auto_clean_with_report_and_explain(self, sample_frame):
        """Test that both report and explain returns (frame, report, explanation)."""
        result = ar.quality.auto_clean(sample_frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_auto_clean_dry_run_no_modification(self, sample_frame):
        """Test that dry_run=True returns the frame without modifications."""
        original_df = sample_frame.to_pandas().copy()
        result = ar.quality.auto_clean(sample_frame, dry_run=True)
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        # Dry run should not modify data
        assert result_df.shape == original_df.shape

    def test_auto_clean_mode_safe(self, sample_frame):
        """Test auto_clean with mode='safe'."""
        result = ar.quality.auto_clean(sample_frame, mode='safe')
        assert result is not None

    def test_auto_clean_empty_frame(self, empty_frame):
        """Test auto_clean handles empty frame gracefully."""
        result = ar.quality.auto_clean(empty_frame)
        assert result is not None

    def test_auto_clean_with_confirmed_casts(self, sample_frame):
        """Test auto_clean with explicit type casts via confirmed_casts."""
        confirmed_casts = {'id': 'int64'}
        result = ar.quality.auto_clean(sample_frame, confirmed_casts=confirmed_casts)
        assert result is not None

    def test_auto_clean_lossy_casts_disabled(self, sample_frame):
        """Test auto_clean with allow_lossy_casts=False."""
        result = ar.quality.auto_clean(sample_frame, allow_lossy_casts=False)
        assert result is not None

    def test_auto_clean_lossy_casts_enabled(self, sample_frame):
        """Test auto_clean with allow_lossy_casts=True."""
        result = ar.quality.auto_clean(sample_frame, allow_lossy_casts=True)
        assert result is not None

    def test_auto_clean_handles_nulls(self, frame_with_nulls):
        """Test that auto_clean handles null and NaN values."""
        result = ar.quality.auto_clean(frame_with_nulls)
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert isinstance(result_df, pd.DataFrame)

    def test_auto_clean_preserves_frame_structure(self, sample_frame):
        """Test that cleaned frame has same or fewer rows and columns."""
        original_shape = sample_frame.to_pandas().shape
        result = ar.quality.auto_clean(sample_frame)
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        # Should not add rows/columns, only remove or consolidate
        assert result_df.shape[0] <= original_shape[0]
        assert result_df.shape[1] <= original_shape[1]

    def test_auto_clean_idempotent(self, sample_frame):
        """Test that applying auto_clean twice produces consistent results."""
        result1 = ar.quality.auto_clean(sample_frame)
        result2 = ar.quality.auto_clean(result1)
        df1 = result1.to_pandas() if hasattr(result1, 'to_pandas') else result1
        df2 = result2.to_pandas() if hasattr(result2, 'to_pandas') else result2
        assert df1.shape == df2.shape

    def test_auto_clean_report_not_none(self, sample_frame):
        """Test that report is not None when return_report=True."""
        frame, report = ar.quality.auto_clean(sample_frame, return_report=True)
        assert report is not None

    def test_auto_clean_explanation_not_none(self, sample_frame):
        """Test that explanation is not None when explain=True."""
        frame, explanation = ar.quality.auto_clean(sample_frame, explain=True)
        assert explanation is not None
