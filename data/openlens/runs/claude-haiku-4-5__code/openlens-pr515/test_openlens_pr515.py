import pytest
import math
from unittest.mock import Mock
from openlens.aberrations import AberrationsCalculator

# WAVELENGTH_GREEN constant value used in openlens
WAVELENGTH_GREEN = 555.0


class TestAberrationsCalculatorGetAberrationsummary:
    """Test suite for AberrationsCalculator.get_aberration_summary."""

    @pytest.fixture
    def mock_lens(self):
        """Create a mock lens object."""
        lens = Mock()
        lens.name = "Test Lens 50mm"
        lens.material = "BK7 Glass"
        return lens

    @pytest.fixture
    def mock_system(self):
        """Create a mock optical system."""
        system = Mock()
        system.name = "Test System"
        system.elements = [Mock(), Mock(), Mock()]
        return system

    @pytest.fixture
    def calculator_lens(self, mock_lens):
        """Create calculator for single lens."""
        calc = AberrationsCalculator(mock_lens)
        calc.lens = mock_lens
        calc.is_system = False
        return calc

    @pytest.fixture
    def calculator_system(self, mock_system):
        """Create calculator for optical system."""
        calc = AberrationsCalculator(mock_system)
        calc.target = mock_system
        calc.is_system = True
        return calc

    @pytest.fixture
    def typical_results(self):
        """Typical aberration results."""
        return {
            'focal_length': 50.0,
            'f_number': 2.8,
            'numerical_aperture': 0.1785,
            'airy_disk_diameter': 0.0032,
            'spherical': 0.0005,
            'coma': 0.0012,
            'astigmatism': 0.0008,
            'field_curvature': 0.1,
            'distortion': 0.5,
            'chromatic': 0.0015,
        }

    def test_default_parameters_calls_calculate_all_aberrations(self, calculator_lens, typical_results):
        """Test that default parameters are passed correctly to calculate_all_aberrations."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary()
        
        calculator_lens.calculate_all_aberrations.assert_called_once_with(
            None, 5.0, wavelength_nm=WAVELENGTH_GREEN
        )

    def test_object_distance_parameter(self, calculator_lens, typical_results):
        """Test that object_distance parameter is passed correctly."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary(object_distance=200.0)
        
        call_args = calculator_lens.calculate_all_aberrations.call_args
        assert call_args[0][0] == 200.0

    def test_field_angle_parameter(self, calculator_lens, typical_results):
        """Test that field_angle parameter is passed correctly."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary(field_angle=10.0)
        
        call_args = calculator_lens.calculate_all_aberrations.call_args
        assert call_args[0][1] == 10.0

    def test_wavelength_parameter(self, calculator_lens, typical_results):
        """Test that wavelength_nm parameter is passed correctly."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary(wavelength_nm=632.8)
        
        call_args = calculator_lens.calculate_all_aberrations.call_args
        assert call_args[1]['wavelength_nm'] == 632.8

    def test_backward_compatibility_field_angle_deg_kwarg(self, calculator_lens, typical_results):
        """Test backward compatibility with field_angle_deg in kwargs."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary(field_angle_deg=15.0)
        
        call_args = calculator_lens.calculate_all_aberrations.call_args
        assert call_args[0][1] == 15.0

    def test_field_angle_deg_takes_precedence(self, calculator_lens, typical_results):
        """Test that field_angle_deg kwarg takes precedence over field_angle."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        calculator_lens.get_aberration_summary(field_angle=5.0, field_angle_deg=20.0)
        
        call_args = calculator_lens.calculate_all_aberrations.call_args
        assert call_args[0][1] == 20.0

    def test_error_handling_from_calculate_all_aberrations(self, calculator_lens):
        """Test that errors from calculate_all_aberrations are properly formatted."""
        calculator_lens.calculate_all_aberrations = Mock(
            return_value={'error': 'Ray trace failed'}
        )
        
        summary = calculator_lens.get_aberration_summary()
        
        assert "Error: Ray trace failed" in summary

    def test_none_values_marked_unavailable(self, calculator_lens):
        """Test that None values in results are displayed as unavailable."""
        none_results = {
            'focal_length': None,
            'f_number': None,
            'numerical_aperture': None,
            'airy_disk_diameter': None,
            'spherical': None,
            'coma': None,
            'astigmatism': None,
            'field_curvature': None,
            'distortion': None,
            'chromatic': None,
        }
        calculator_lens.calculate_all_aberrations = Mock(return_value=none_results)
        
        summary = calculator_lens.get_aberration_summary()
        
        assert "unavailable" in summary.lower()
        assert isinstance(summary, str)

    def test_non_finite_values_marked_unavailable(self, calculator_lens):
        """Test that non-finite values (inf, nan) are marked unavailable."""
        inf_results = {
            'focal_length': float('inf'),
            'f_number': float('nan'),
            'numerical_aperture': 0.1785,
            'airy_disk_diameter': float('-inf'),
            'spherical': 0.0005,
            'coma': 0.0012,
            'astigmatism': 0.0008,
            'field_curvature': 0.1,
            'distortion': 0.5,
            'chromatic': 0.0015,
        }
        calculator_lens.calculate_all_aberrations = Mock(return_value=inf_results)
        
        summary = calculator_lens.get_aberration_summary()
        
        assert "unavailable" in summary.lower()

    def test_single_lens_format_includes_lens_info(self, calculator_lens, typical_results):
        """Test that single lens summary includes lens name and material."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        summary = calculator_lens.get_aberration_summary()
        
        assert "Test Lens 50mm" in summary
        assert "BK7 Glass" in summary
        assert "LENS ABERRATIONS ANALYSIS" in summary

    def test_system_format_includes_element_count(self, calculator_system, typical_results):
        """Test that system summary includes element count and shows dash for material."""
        calculator_system.calculate_all_aberrations = Mock(return_value=typical_results)
        
        summary = calculator_system.get_aberration_summary()
        
        assert "Test System (3 elements)" in summary
        assert "Material:" in summary and "-" in summary

    def test_output_contains_required_sections(self, calculator_lens, typical_results):
        """Test that output contains all required sections and metrics."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        summary = calculator_lens.get_aberration_summary()
        
        required_sections = [
            "LENS ABERRATIONS ANALYSIS",
            "BASIC PARAMETERS",
            "PRIMARY ABERRATIONS",
            "CHROMATIC ABERRATION",
            "INTERPRETATION:",
            "Focal Length:",
            "F-number",
            "Numerical Aperture:",
            "Spherical Aberration:",
            "Distortion:",
        ]
        for section in required_sections:
            assert section in summary

    def test_field_angle_displayed_in_output(self, calculator_lens, typical_results):
        """Test that custom field angle is displayed in coma and distortion sections."""
        calculator_lens.calculate_all_aberrations = Mock(return_value=typical_results)
        
        summary = calculator_lens.get_aberration_summary(field_angle=8.0)
        
        assert "@ 8.0°" in summary
