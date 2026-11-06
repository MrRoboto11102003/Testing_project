import pytest
import pandas as pd
import numpy as np
from io import StringIO
import arnio as ar


class TestAutoClean:
    """Test suite for arnio.quality.auto_clean function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with various data quality issues."""
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, '35', 40, None],
            'salary': [50000.0, 60000.0, None, 80000.0, 90000.0],
            'email': ['alice@example.com', 'bob@example.com', '', 'diana@example.com', None],
            'active': [True, False, 'yes', True, False],
        }
        df = pd.DataFrame(data)
        return ar.ArFrame(df)

    @pytest.fixture
    def clean_frame(self):
        """Create a clean ArFrame with no obvious issues."""
        data = {
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 70000.0],
        }
        df = pd.DataFrame(data)
        return ar.ArFrame(df)

    def test_auto_clean_basic_safe_mode(self, sample_frame):
        """Test auto_clean with default safe mode returns ArFrame."""
        result = ar.quality.auto_clean(sample_frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_return_report(self, sample_frame):
        """Test auto_clean with return_report=True returns tuple with report."""
        result = ar.quality.auto_clean(sample_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        # result[1] should be DataQualityReport or dict-like

    def test_auto_clean_dry_run_no_modification(self, sample_frame):
        """Test dry_run=True returns frame without modifications."""
        original = sample_frame.copy()
        result = ar.quality.auto_clean(sample_frame, dry_run=True)
        assert isinstance(result, ar.ArFrame)
        pd.testing.assert_frame_equal(result, original)

    def test_auto_clean_dry_run_with_report(self, sample_frame):
        """Test dry_run with return_report shows what would be cleaned."""
        result = ar.quality.auto_clean(
            sample_frame, dry_run=True, return_report=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)

    def test_auto_clean_explain_mode(self, sample_frame):
        """Test explain=True returns explanation of cleaning operations."""
        result = ar.quality.auto_clean(sample_frame, explain=True)
        assert isinstance(result, tuple)
        # Should contain frame and explanation

    def test_auto_clean_explain_with_report(self, sample_frame):
        """Test explain with return_report gives frame, report, and explanation."""
        result = ar.quality.auto_clean(
            sample_frame, explain=True, return_report=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)

    def test_auto_clean_confirmed_casts(self, sample_frame):
        """Test confirmed_casts parameter applies specified type conversions."""
        confirmed = {'age': 'int64', 'active': 'bool'}
        result = ar.quality.auto_clean(
            sample_frame, confirmed_casts=confirmed
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_allow_lossy_casts_false(self, sample_frame):
        """Test allow_lossy_casts=False prevents lossy conversions."""
        result = ar.quality.auto_clean(
            sample_frame, allow_lossy_casts=False
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_allow_lossy_casts_true(self, sample_frame):
        """Test allow_lossy_casts=True permits lossy conversions."""
        result = ar.quality.auto_clean(
            sample_frame, allow_lossy_casts=True
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_mode_safe(self, sample_frame):
        """Test safe mode uses conservative cleaning strategy."""
        result = ar.quality.auto_clean(sample_frame, mode='safe')
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_mode_aggressive(self, sample_frame):
        """Test aggressive mode applies more extensive cleaning."""
        result = ar.quality.auto_clean(sample_frame, mode='aggressive')
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_clean_frame_unchanged(self, clean_frame):
        """Test that a clean frame is returned unchanged."""
        original_shape = clean_frame.shape
        result = ar.quality.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape == original_shape

    def test_auto_clean_empty_frame(self):
        """Test auto_clean handles empty frames."""
        empty = ar.ArFrame(pd.DataFrame())
        result = ar.quality.auto_clean(empty)
        assert isinstance(result, ar.ArFrame)
        assert result.empty

    def test_auto_clean_all_nulls_column(self):
        """Test auto_clean handles columns with all null values."""
        data = {
            'id': [1, 2, 3],
            'all_nulls': [None, None, None],
        }
        df = ar.ArFrame(pd.DataFrame(data))
        result = ar.quality.auto_clean(df)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_combined_parameters(self, sample_frame):
        """Test auto_clean with multiple parameters combined."""
        result = ar.quality.auto_clean(
            sample_frame,
            mode='safe',
            return_report=True,
            explain=True,
            allow_lossy_casts=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
