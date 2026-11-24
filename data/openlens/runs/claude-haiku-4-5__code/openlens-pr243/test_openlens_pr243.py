import pytest
from unittest.mock import patch
from openlens.optical_system import OpticalSystem


class TestCalculateBackFocalLength:
    """Test suite for OpticalSystem.calculate_back_focal_length method."""
    
    @pytest.fixture
    def system(self):
        """Create an optical system instance for testing."""
        return OpticalSystem()
    
    def test_valid_matrix_calculation(self, system):
        """Test BFL calculation with a valid ABCD matrix."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(2.0, 1.0, 0.5, 2.0)):
            result = system.calculate_back_focal_length()
            assert result == -4.0
    
    def test_zero_a_coefficient(self, system):
        """Test BFL calculation when A coefficient is zero."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(0.0, 1.0, 0.5, 2.0)):
            result = system.calculate_back_focal_length()
            assert result == 0.0
    
    def test_negative_a_coefficient(self, system):
        """Test BFL calculation with negative A coefficient."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(-2.0, 1.0, 0.5, 2.0)):
            result = system.calculate_back_focal_length()
            assert result == 4.0
    
    def test_negative_c_coefficient(self, system):
        """Test BFL calculation with negative C coefficient."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, -0.2, 2.0)):
            result = system.calculate_back_focal_length()
            assert result == 5.0
    
    def test_none_matrix_returns_none(self, system):
        """Test that None matrix returns None."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=None):
            result = system.calculate_back_focal_length()
            assert result is None
    
    def test_empty_matrix_returns_none(self, system):
        """Test that empty list matrix returns None."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=[]):
            result = system.calculate_back_focal_length()
            assert result is None
    
    def test_c_at_threshold_boundary(self, system):
        """Test C value at the 1e-10 threshold boundary (not below)."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, 1e-10, 2.0)):
            result = system.calculate_back_focal_length()
            assert result is not None
            assert result == pytest.approx(-1e10, rel=1e-6)
    
    def test_c_below_threshold_positive(self, system):
        """Test positive C value below the 1e-10 threshold."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, 5e-11, 2.0)):
            result = system.calculate_back_focal_length()
            assert result is None
    
    def test_c_below_threshold_negative(self, system):
        """Test negative C value below the 1e-10 threshold."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, -5e-11, 2.0)):
            result = system.calculate_back_focal_length()
            assert result is None
    
    def test_with_n_overrides_parameter(self, system):
        """Test BFL calculation passes n_overrides to system matrix calculation."""
        n_overrides = {'glass': 1.5}
        
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.5, 2.0, 0.3, 1.0)) as mock_method:
            result = system.calculate_back_focal_length(n_overrides=n_overrides)
            mock_method.assert_called_once_with(n_overrides=n_overrides)
            assert result == -5.0
    
    def test_without_n_overrides_parameter(self, system):
        """Test BFL calculation without n_overrides parameter."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, 0.1, 1.0)) as mock_method:
            result = system.calculate_back_focal_length()
            mock_method.assert_called_once_with(n_overrides=None)
            assert result == -10.0
    
    def test_large_magnitude_values(self, system):
        """Test BFL calculation with large magnitude values."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1000.0, 100.0, 0.001, 50.0)):
            result = system.calculate_back_focal_length()
            assert result == -1e6
    
    def test_very_small_non_zero_c(self, system):
        """Test BFL calculation with very small but non-zero C."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, 1e-9, 2.0)):
            result = system.calculate_back_focal_length()
            assert result is not None
            assert result == pytest.approx(-1e9, rel=1e-6)
    
    def test_identity_matrix_with_zero_c(self, system):
        """Test BFL with identity-like matrix where C=0."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 0.0, 0.0, 1.0)):
            result = system.calculate_back_focal_length()
            assert result is None
    
    def test_converging_lens_typical_values(self, system):
        """Test BFL with typical converging lens system values."""
        with patch.object(system, '_calculate_system_matrix', 
                         return_value=(1.0, 1.0, -0.1, 1.0)):
            result = system.calculate_back_focal_length()
            assert result == 10.0
