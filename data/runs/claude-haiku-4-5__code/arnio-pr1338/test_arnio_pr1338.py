import pytest
import pandas as pd
import arnio as ar


class TestAutoClean:
    """Test suite for arnio.quality.auto_clean function."""
    
    @pytest.fixture
    def simple_frame(self):
        """Create a simple test frame."""
        return ar.ArFrame(pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        }))
    
    @pytest.fixture
    def dirty_frame(self):
        """Frame with whitespace and duplicates."""
        return ar.ArFrame(pd.DataFrame({
            'name': ['  Alice  ', 'Bob', '  Alice  '],
            'value': [10, 20, 10]
        }))
    
    # Return type tests
    def test_default_returns_frame(self, simple_frame):
        """Default call returns just the frame."""
        result = ar.auto_clean(simple_frame)
        assert isinstance(result, (ar.ArFrame, pd.DataFrame))
    
    def test_return_report_returns_frame_and_report(self, simple_frame):
        """return_report=True returns (frame, report) tuple."""
        result = ar.auto_clean(simple_frame, return_report=True)
        assert isinstance(result, tuple) and len(result) == 2
    
    def test_dry_run_returns_report_only(self, simple_frame):
        """dry_run=True returns just the report, not a frame."""
        result = ar.auto_clean(simple_frame, dry_run=True)
        assert not isinstance(result, (ar.ArFrame, pd.DataFrame))
    
    def test_explain_returns_frame_and_explanation(self, simple_frame):
        """explain=True returns (frame, explanation) tuple."""
        result = ar.auto_clean(simple_frame, explain=True)
        assert isinstance(result, tuple) and len(result) == 2
    
    def test_return_report_and_explain_returns_triple(self, simple_frame):
        """Both return_report and explain flags return 3-tuple."""
        result = ar.auto_clean(simple_frame, return_report=True, explain=True)
        assert isinstance(result, tuple) and len(result) == 3
    
    # Type validation tests
    def test_return_report_type_validation(self, simple_frame):
        """return_report must be bool."""
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(simple_frame, return_report="yes")
    
    def test_dry_run_type_validation(self, simple_frame):
        """dry_run must be bool."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(simple_frame, dry_run=1)
    
    def test_allow_lossy_casts_type_validation(self, simple_frame):
        """allow_lossy_casts must be bool."""
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            ar.auto_clean(simple_frame, allow_lossy_casts="yes")
    
    def test_explain_type_validation(self, simple_frame):
        """explain must be bool."""
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(simple_frame, explain=1)
    
    # Value validation tests
    def test_invalid_mode(self, simple_frame):
        """mode must be 'safe' or 'strict'."""
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(simple_frame, mode="aggressive")
    
    def test_explain_with_dry_run_conflict(self, simple_frame):
        """explain and dry_run cannot both be True."""
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(simple_frame, explain=True, dry_run=True)
    
    def test_return_report_with_dry_run_conflict(self, simple_frame):
        """return_report and dry_run cannot both be True."""
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(simple_frame, return_report=True, dry_run=True)
    
    # Functionality tests
    def test_dry_run_preserves_frame(self, simple_frame):
        """dry_run=True does not modify the input frame."""
        original_shape = simple_frame.shape
        original_values = simple_frame.values.copy()
        ar.auto_clean(simple_frame, dry_run=True)
        assert simple_frame.shape == original_shape
        assert (simple_frame.values == original_values).all()
    
    def test_safe_mode_cleans_frame(self, dirty_frame):
        """safe mode applies safe cleaning operations."""
        result = ar.auto_clean(dirty_frame, mode="safe")
        assert isinstance(result, (ar.ArFrame, pd.DataFrame))
        assert result.shape[0] > 0
    
    def test_strict_mode_cleans_frame(self, dirty_frame):
        """strict mode applies more aggressive cleaning operations."""
        result = ar.auto_clean(dirty_frame, mode="strict")
        assert isinstance(result, (ar.ArFrame, pd.DataFrame))
        assert result.shape[0] > 0
