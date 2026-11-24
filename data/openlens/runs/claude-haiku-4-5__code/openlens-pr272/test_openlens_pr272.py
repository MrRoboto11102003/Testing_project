import pytest
from unittest.mock import Mock, patch
from openlens.optical_system import OpticalSystem


class TestGetSystemFocalLength:
    """Test cases for OpticalSystem.get_system_focal_length()"""
    
    def test_empty_system_returns_none(self):
        """Empty system (no elements) should return None"""
        system = OpticalSystem()
        assert system.get_system_focal_length() is None
    
    def test_single_lens_positive_focal_length(self):
        """Single lens with positive focal length returns its focal length"""
        system = OpticalSystem()
        mock_lens = Mock()
        mock_lens.calculate_focal_length.return_value = 50.0
        mock_element = Mock()
        mock_element.lens = mock_lens
        
        system.elements = [mock_element]
        result = system.get_system_focal_length()
        
        assert result == 50.0
        mock_lens.calculate_focal_length.assert_called_once()
    
    def test_single_lens_negative_focal_length(self):
        """Single lens with negative focal length (diverging)"""
        system = OpticalSystem()
        mock_lens = Mock()
        mock_lens.calculate_focal_length.return_value = -30.0
        mock_element = Mock()
        mock_element.lens = mock_lens
        
        system.elements = [mock_element]
        assert system.get_system_focal_length() == -30.0
    
    def test_two_element_system_positive_c(self):
        """Two-element system with positive C uses matrix method: f = -1/C"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 0.02, 1)):
            assert system.get_system_focal_length() == pytest.approx(-50.0)
    
    def test_two_element_system_negative_c(self):
        """Two-element system with negative C: f = -1/C gives positive f"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, -0.02, 1)):
            assert system.get_system_focal_length() == pytest.approx(50.0)
    
    def test_multi_element_system_uses_matrix(self):
        """Multi-element system (3+) uses exact ABCD matrix method"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 5, -0.05, 1)):
            assert system.get_system_focal_length() == pytest.approx(20.0)
    
    def test_afocal_system_c_zero(self):
        """Afocal system with C=0 (infinite focal length) returns None"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 10, 0.0, 1)):
            assert system.get_system_focal_length() is None
    
    def test_afocal_system_c_below_threshold(self):
        """System with |C| < 1e-10 (below threshold) returns None"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 5e-11, 1)):
            assert system.get_system_focal_length() is None
    
    def test_system_c_at_threshold_boundary(self):
        """System with |C| exactly at 1e-10 threshold computes focal length"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 1e-10, 1)):
            result = system.get_system_focal_length()
            assert result == pytest.approx(-1e10)
    
    def test_system_c_just_below_threshold(self):
        """System with |C| just below 1e-10 threshold returns None"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 9.99e-11, 1)):
            assert system.get_system_focal_length() is None
    
    def test_matrix_calculation_returns_none(self):
        """When _calculate_system_matrix returns None, returns None"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=None):
            assert system.get_system_focal_length() is None
    
    def test_matrix_calculation_returns_falsy_sequence(self):
        """When _calculate_system_matrix returns falsy value, returns None"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=()):
            assert system.get_system_focal_length() is None
    
    def test_very_large_focal_length(self):
        """Very large focal length from small C value"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 1e-6, 1)):
            result = system.get_system_focal_length()
            assert result == pytest.approx(-1e6)
    
    def test_very_small_focal_length(self):
        """Very small focal length from large C value"""
        system = OpticalSystem()
        system.elements = [Mock(), Mock()]
        
        with patch.object(system, '_calculate_system_matrix', return_value=(1, 0, 1000, 1)):
            result = system.get_system_focal_length()
            assert result == pytest.approx(-0.001)
    
    def test_single_lens_does_not_call_matrix_method(self):
        """Single lens should NOT call _calculate_system_matrix method"""
        system = OpticalSystem()
        mock_lens = Mock()
        mock_lens.calculate_focal_length.return_value = 75.0
        mock_element = Mock()
        mock_element.lens = mock_lens
        system.elements = [mock_element]
        
        with patch.object(system, '_calculate_system_matrix') as mock_matrix:
            result = system.get_system_focal_length()
            assert result == 75.0
            mock_matrix.assert_not_called()
