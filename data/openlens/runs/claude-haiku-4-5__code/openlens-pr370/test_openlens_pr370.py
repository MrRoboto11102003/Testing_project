import pytest
from unittest.mock import Mock
import numpy as np
from openlens.performance_metrics import PerformanceMetrics


class TestCalculateBackFocalLength:
    """Test suite for PerformanceMetrics.calculate_back_focal_length()"""
    
    def test_returns_none_when_no_lens_or_system(self):
        """Should return None when neither lens nor system is set"""
        metrics = PerformanceMetrics()
        metrics.lens = None
        metrics.system = None
        result = metrics.calculate_back_focal_length()
        assert result is None
    
    def test_delegates_to_lens_when_lens_exists(self):
        """Should delegate to lens.calculate_back_focal_length() when lens is set"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 25.5
        metrics.lens = mock_lens
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        mock_lens.calculate_back_focal_length.assert_called_once()
        assert result == 25.5
    
    def test_delegates_to_system_when_system_exists_no_lens(self):
        """Should delegate to system.calculate_back_focal_length() when system is set"""
        metrics = PerformanceMetrics()
        mock_system = Mock()
        mock_system.calculate_back_focal_length.return_value = 30.0
        metrics.lens = None
        metrics.system = mock_system
        
        result = metrics.calculate_back_focal_length()
        
        mock_system.calculate_back_focal_length.assert_called_once()
        assert result == 30.0
    
    def test_lens_takes_precedence_over_system(self):
        """When both lens and system exist, lens should take precedence"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 20.0
        mock_system = Mock()
        mock_system.calculate_back_focal_length.return_value = 40.0
        
        metrics.lens = mock_lens
        metrics.system = mock_system
        
        result = metrics.calculate_back_focal_length()
        
        mock_lens.calculate_back_focal_length.assert_called_once()
        mock_system.calculate_back_focal_length.assert_not_called()
        assert result == 20.0
    
    def test_return_value_is_float_from_lens(self):
        """Return value should be numeric when delegating to lens"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 15.75
        metrics.lens = mock_lens
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        assert isinstance(result, (int, float, np.floating))
        assert result == 15.75
    
    def test_return_value_is_float_from_system(self):
        """Return value should be numeric when delegating to system"""
        metrics = PerformanceMetrics()
        mock_system = Mock()
        mock_system.calculate_back_focal_length.return_value = 42.5
        metrics.lens = None
        metrics.system = mock_system
        
        result = metrics.calculate_back_focal_length()
        
        assert isinstance(result, (int, float, np.floating))
        assert result == 42.5
    
    def test_bfl_zero_value(self):
        """BFL can be zero (focus at last surface)"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 0.0
        metrics.lens = mock_lens
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        assert result == 0.0
    
    def test_bfl_small_positive_value(self):
        """BFL should handle small positive distances"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 0.001
        metrics.lens = mock_lens
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        assert result == pytest.approx(0.001, rel=1e-9)
    
    def test_bfl_large_positive_value(self):
        """BFL should handle large positive distances"""
        metrics = PerformanceMetrics()
        mock_system = Mock()
        mock_system.calculate_back_focal_length.return_value = 1000.0
        metrics.lens = None
        metrics.system = mock_system
        
        result = metrics.calculate_back_focal_length()
        
        assert result == 1000.0
    
    def test_bfl_negative_value(self):
        """BFL can be negative (virtual focus behind last surface)"""
        metrics = PerformanceMetrics()
        mock_system = Mock()
        mock_system.calculate_back_focal_length.return_value = -15.0
        metrics.lens = None
        metrics.system = mock_system
        
        result = metrics.calculate_back_focal_length()
        
        assert result == -15.0
    
    def test_multiple_calls_consistent(self):
        """Multiple calls should return consistent results"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 25.0
        metrics.lens = mock_lens
        metrics.system = None
        
        result1 = metrics.calculate_back_focal_length()
        result2 = metrics.calculate_back_focal_length()
        
        assert result1 == result2
        assert mock_lens.calculate_back_focal_length.call_count == 2
    
    def test_return_type_is_optional_float(self):
        """Return type should be Optional[float]"""
        metrics = PerformanceMetrics()
        metrics.lens = None
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        assert result is None or isinstance(result, (int, float, np.floating))
    
    def test_does_not_modify_state(self):
        """Calling the method should not modify object state"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = 20.0
        metrics.lens = mock_lens
        original_lens = metrics.lens
        original_system = metrics.system
        
        metrics.calculate_back_focal_length()
        
        assert metrics.lens is original_lens
        assert metrics.system is original_system
    
    def test_very_small_negative_bfl(self):
        """BFL can be a very small negative value"""
        metrics = PerformanceMetrics()
        mock_lens = Mock()
        mock_lens.calculate_back_focal_length.return_value = -0.0001
        metrics.lens = mock_lens
        metrics.system = None
        
        result = metrics.calculate_back_focal_length()
        
        assert result == pytest.approx(-0.0001, abs=1e-7)
