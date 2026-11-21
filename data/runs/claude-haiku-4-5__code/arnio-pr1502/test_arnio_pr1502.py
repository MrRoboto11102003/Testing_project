import pytest
import pandas as pd
import arnio as ar


class TestAutoClean:
    """Tests for ar.auto_clean() function."""
    
    @pytest.fixture
    def simple_frame(self):
        """Simple frame for testing."""
        return ar.wrap(pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['alice', 'bob', 'charlie']
        }))
    
    def test_default_returns_frame(self, simple_frame):
        """Default call returns an ArFrame."""
        result = ar.auto_clean(simple_frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_dry_run_returns_report(self, simple_frame):
        """dry_run=True returns a report object, not a frame."""
        result = ar.auto_clean(simple_frame, dry_run=True)
        assert not isinstance(result, ar.ArFrame)
    
    def test_return_report_returns_tuple(self, simple_frame):
        """return_report=True returns (frame, report) tuple."""
        result = ar.auto_clean(simple_frame, return_report=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
    
    def test_explain_returns_tuple(self, simple_frame):
        """explain=True returns (frame, explanation) tuple."""
        result = ar.auto_clean(simple_frame, explain=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
    
    def test_return_report_and_explain_returns_triple(self, simple_frame):
        """Both return_report and explain return 3-tuple."""
        result = ar.auto_clean(simple_frame, return_report=True, explain=True)
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
    
    def test_invalid_mode_raises_error(self, simple_frame):
        """Invalid mode value raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(simple_frame, mode='invalid')
    
    def test_non_string_mode_raises_error(self, simple_frame):
        """Non-string mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(simple_frame, mode=123)
    
    def test_dry_run_and_explain_conflict(self, simple_frame):
        """dry_run=True and explain=True together raise ValueError."""
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(simple_frame, dry_run=True, explain=True)
    
    def test_dry_run_and_return_report_conflict(self, simple_frame):
        """dry_run=True and return_report=True together raise ValueError."""
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(simple_frame, dry_run=True, return_report=True)
    
    def test_invalid_return_report_type(self, simple_frame):
        """Non-bool return_report raises TypeError."""
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(simple_frame, return_report='true')
    
    def test_invalid_dry_run_type(self, simple_frame):
        """Non-bool dry_run raises TypeError."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(simple_frame, dry_run=1)
    
    def test_invalid_allow_lossy_casts_type(self, simple_frame):
        """Non-bool allow_lossy_casts raises TypeError."""
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            ar.auto_clean(simple_frame, allow_lossy_casts='true')
    
    def test_invalid_explain_type(self, simple_frame):
        """Non-bool explain raises TypeError."""
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(simple_frame, explain=None)
    
    def test_mode_safe_accepted(self, simple_frame):
        """mode='safe' is accepted and returns frame."""
        result = ar.auto_clean(simple_frame, mode='safe')
        assert isinstance(result, ar.ArFrame)
    
    def test_mode_strict_accepted(self, simple_frame):
        """mode='strict' is accepted and returns frame."""
        result = ar.auto_clean(simple_frame, mode='strict')
        assert isinstance(result, ar.ArFrame)
