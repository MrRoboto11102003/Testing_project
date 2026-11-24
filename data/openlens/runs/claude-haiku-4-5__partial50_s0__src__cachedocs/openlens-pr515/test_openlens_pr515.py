import pytest
import math
from unittest.mock import Mock
from openlens.aberrations import AberrationsCalculator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.constants import WAVELENGTH_GREEN


class TestGetAberrationSummary:
    """Tests for AberrationsCalculator.get_aberration_summary method."""

    @pytest.fixture
    def mock_lens(self):
        """Create a mock lens."""
        lens = Mock(spec=Lens)
        lens.name = "Test Lens"
        lens.material = "BK7"
        return lens

    @pytest.fixture
    def mock_system(self):
        """Create a mock optical system."""
        system = Mock(spec=OpticalSystem)
        system.name = "Test System"
        system.elements = [Mock(), Mock()]
        return system

    @pytest.fixture
    def healthy_results(self):
        """Return a complete healthy results dict."""
        return {
            'focal_length': 50.0,
            'f_number': 2.0,
            'numerical_aperture': 0.25,
            'airy_disk_diameter': 1.5,
            'spherical': 0.0005,
            'coma': 0.001,
            'astigmatism': 0.002,
            'field_curvature': 0.1,
            'distortion': 0.5,
            'chromatic': 0.01,
        }

    def _create_calculator(self, mock_obj, is_system=False):
        """Helper to create a calculator with mocked calculate_all_aberrations."""
        calc = Mock(spec=AberrationsCalculator)
        if is_system:
            calc.is_system = True
            calc.target = mock_obj
        else:
            calc.is_system = False
            calc.lens = mock_obj
            calc.target = mock_obj
        return calc

    def test_returns_string(self, mock_lens, healthy_results):
        """Test that get_aberration_summary returns a string."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        result = AberrationsCalculator.get_aberration_summary(
            calc, object_distance=None, field_angle=5.0
        )

        assert isinstance(result, str)

    def test_contains_lens_name_and_material(self, mock_lens, healthy_results):
        """Test that single lens name and material appear in summary."""
        calc = self._create_calculator(mock_lens, is_system=False)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "Test Lens" in result
        assert "BK7" in result

    def test_contains_system_name_and_element_count(self, mock_system, healthy_results):
        """Test that system name and element count appear in summary."""
        calc = self._create_calculator(mock_system, is_system=True)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "Test System" in result
        assert "2 elements" in result

    def test_contains_all_major_sections(self, mock_lens, healthy_results):
        """Test that all major sections are present."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "LENS ABERRATIONS ANALYSIS" in result
        assert "BASIC PARAMETERS" in result
        assert "PRIMARY ABERRATIONS" in result
        assert "CHROMATIC ABERRATION" in result
        assert "INTERPRETATION" in result

    def test_handles_error_in_results(self, mock_lens):
        """Test graceful handling of errors from calculate_all_aberrations."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value={'error': 'Trace failed'})

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "Error:" in result
        assert "Trace failed" in result

    def test_handles_all_none_values(self, mock_lens):
        """Test that None values display as unavailable."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value={
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
        })

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "unavailable" in result
        assert "Error:" not in result

    def test_handles_inf_values(self, mock_lens):
        """Test that inf and -inf values display as unavailable."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value={
            'focal_length': float('inf'),
            'f_number': float('-inf'),
            'numerical_aperture': 0.25,
            'airy_disk_diameter': 1.5,
            'spherical': 0.0,
            'coma': 0.0,
            'astigmatism': 0.0,
            'field_curvature': 0.0,
            'distortion': 0.0,
            'chromatic': 0.0,
        })

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "unavailable" in result

    def test_handles_nan_values(self, mock_lens):
        """Test that NaN values display as unavailable."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value={
            'focal_length': float('nan'),
            'f_number': 2.0,
            'numerical_aperture': 0.25,
            'airy_disk_diameter': 1.5,
            'spherical': 0.0,
            'coma': 0.0,
            'astigmatism': 0.0,
            'field_curvature': 0.0,
            'distortion': 0.0,
            'chromatic': 0.0,
        })

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "unavailable" in result

    def test_default_parameters(self, mock_lens, healthy_results):
        """Test that default parameters are used correctly."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        AberrationsCalculator.get_aberration_summary(calc)

        call_args = calc.calculate_all_aberrations.call_args
        assert call_args[0][0] is None  # object_distance
        assert call_args[0][1] == 5.0  # field_angle
        assert call_args[1]['wavelength_nm'] == WAVELENGTH_GREEN

    def test_backward_compat_field_angle_deg(self, mock_lens, healthy_results):
        """Test backward compatibility with field_angle_deg kwarg."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        AberrationsCalculator.get_aberration_summary(
            calc, field_angle_deg=10.0
        )

        call_args = calc.calculate_all_aberrations.call_args
        assert call_args[0][1] == 10.0  # field_angle_deg overrides default

    def test_custom_parameters_passed_correctly(self, mock_lens, healthy_results):
        """Test that custom parameters are passed to calculate_all_aberrations."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        AberrationsCalculator.get_aberration_summary(
            calc,
            object_distance=200.0,
            field_angle=15.0,
            wavelength_nm=656.0
        )

        call_args = calc.calculate_all_aberrations.call_args
        assert call_args[0][0] == 200.0
        assert call_args[0][1] == 15.0
        assert call_args[1]['wavelength_nm'] == 656.0

    def test_includes_specific_metrics_in_output(self, mock_lens, healthy_results):
        """Test that specific metric labels appear in the output."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value=healthy_results)

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "Focal Length:" in result
        assert "F-number" in result
        assert "Numerical Aperture:" in result
        assert "Airy Disk Diameter:" in result
        assert "Spherical Aberration:" in result
        assert "Coma" in result
        assert "Astigmatism" in result
        assert "Field Curvature:" in result
        assert "Distortion" in result
        assert "Longitudinal CA:" in result

    def test_handles_mixed_valid_and_invalid_values(self, mock_lens):
        """Test handling of mixed valid and invalid values in results."""
        calc = self._create_calculator(mock_lens)
        calc.calculate_all_aberrations = Mock(return_value={
            'focal_length': 50.0,
            'f_number': None,
            'numerical_aperture': 0.25,
            'airy_disk_diameter': float('nan'),
            'spherical': 0.0005,
            'coma': float('inf'),
            'astigmatism': 0.002,
            'field_curvature': None,
            'distortion': 0.5,
            'chromatic': 0.01,
        })

        result = AberrationsCalculator.get_aberration_summary(calc)

        assert "50" in result
        assert "unavailable" in result
        assert "Error:" not in result
