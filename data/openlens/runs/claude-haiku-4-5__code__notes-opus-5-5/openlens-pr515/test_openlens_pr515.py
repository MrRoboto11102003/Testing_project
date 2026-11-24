import pytest
import math
from unittest.mock import Mock
import numpy as np
from openlens.aberrations import AberrationsCalculator
from openlens.lens import Lens


class TestAberrationsCalculatorGetAberrationSummary:
    """Tests for AberrationsCalculator.get_aberration_summary method."""
    
    @pytest.fixture
    def simple_lens(self):
        """Create a simple test lens."""
        return Lens(name="TestLens", radius1=50, radius2=-50, thickness=5, material="BK7")
    
    @pytest.fixture
    def calculator(self, simple_lens):
        """Create an AberrationsCalculator instance."""
        return AberrationsCalculator(simple_lens)
    
    @pytest.fixture
    def valid_results(self):
        """Valid mock results from calculate_all_aberrations."""
        return {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": -0.005,
            "coma": 0.0,
            "astigmatism": 0.02,
            "field_curvature": 0.1,
            "distortion": 0.5,
            "chromatic": 0.3,
        }
    
    # ========== Basic functionality ==========
    
    def test_returns_string_type(self, calculator, valid_results):
        """Test that get_aberration_summary returns a string."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        result = calculator.get_aberration_summary()
        assert isinstance(result, str)
    
    def test_summary_starts_with_newline(self, calculator, valid_results):
        """Test that normal summary starts with newline."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        result = calculator.get_aberration_summary()
        assert result.startswith("\n")
    
    def test_summary_contains_required_sections(self, calculator, valid_results):
        """Test that summary contains all expected section headers."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        result = calculator.get_aberration_summary()
        assert "LENS ABERRATIONS ANALYSIS" in result
        assert "BASIC PARAMETERS" in result
        assert "PRIMARY ABERRATIONS (Seidel)" in result
        assert "CHROMATIC ABERRATION" in result
        assert "INTERPRETATION:" in result
        assert "Lens: TestLens" in result
    
    # ========== Error handling ==========
    
    def test_object_distance_not_none_raises_error(self, calculator):
        """Test that non-None object_distance raises ValueError."""
        calculator.calculate_all_aberrations = Mock(
            side_effect=ValueError("object distance is not supported: got 100.0 mm.")
        )
        with pytest.raises(ValueError, match="object distance is not supported"):
            calculator.get_aberration_summary(object_distance=100.0)
    
    def test_error_result_returns_error_string(self, calculator):
        """Test that error in results returns 'Error: ...' string."""
        error_msg = "Cannot calculate focal length (zero optical power)"
        calculator.calculate_all_aberrations = Mock(return_value={"error": error_msg})
        result = calculator.get_aberration_summary()
        assert result == f"Error: {error_msg}"
    
    # ========== Parameter passing ==========
    
    def test_field_angle_appears_in_labels(self, calculator, valid_results):
        """Test that field_angle parameter appears in metric labels."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        result = calculator.get_aberration_summary(field_angle=7.5)
        assert "(@ 7.5°)" in result
    
    def test_field_angle_deg_kwarg_overrides_calculation(self, calculator, valid_results):
        """Test field_angle_deg kwarg overrides field_angle for calculation."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        calculator.get_aberration_summary(field_angle=5.0, field_angle_deg=10.0)
        
        call_args = calculator.calculate_all_aberrations.call_args
        assert call_args[0][1] == 10.0  # Second positional arg is field_angle_deg
    
    def test_wavelength_parameter_forwarded(self, calculator, valid_results):
        """Test wavelength_nm is forwarded to calculate_all_aberrations."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        calculator.get_aberration_summary(wavelength_nm=600.0)
        
        call_kwargs = calculator.calculate_all_aberrations.call_args[1]
        assert call_kwargs["wavelength_nm"] == 600.0
    
    def test_object_distance_none_forwarded(self, calculator, valid_results):
        """Test that object_distance=None is passed correctly."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        calculator.get_aberration_summary(object_distance=None)
        
        call_args = calculator.calculate_all_aberrations.call_args[0]
        assert call_args[0] is None
    
    # ========== Metric formatting ==========
    
    def test_metric_with_valid_float(self, calculator, valid_results):
        """Test that valid floats are formatted with correct precision."""
        calculator.calculate_all_aberrations = Mock(return_value=valid_results)
        result = calculator.get_aberration_summary()
        assert "100.00 mm" in result  # focal_length .2f
        assert "4.00" in result  # f_number .2f
        assert "0.1250" in result  # numerical_aperture .4f
    
    def test_metric_negative_value_retains_sign(self, calculator):
        """Test that negative values retain sign in formatting."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": -0.0123, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "-0.0123 mm (longitudinal)" in result
    
    def test_metric_none_renders_unavailable(self, calculator):
        """Test that None values render as 'unavailable'."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": None,
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": None,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "unavailable" in result
        assert "unavailable (relative)" in result  # coma None
    
    def test_metric_nan_renders_unavailable(self, calculator):
        """Test that NaN values render as 'unavailable'."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": float("nan"),
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": float("nan"), "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "unavailable" in result
    
    def test_metric_infinity_renders_unavailable(self, calculator):
        """Test that infinite values render as 'unavailable'."""
        results = {
            "focal_length": float("inf"), "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": float("-inf"), "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "unavailable" in result
    
    # ========== Trend interpretation ==========
    
    def test_trend_spherical_negligible(self, calculator):
        """Test spherical aberration < 0.001 mm marked as Negligible."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.0005, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Negligible" in result
    
    def test_trend_spherical_moderate(self, calculator):
        """Test spherical aberration in [0.001, 0.01) marked as Moderate."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": -0.005, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Moderate" in result
        assert "(0.0050 mm)" in result  # detail line, absolute value
    
    def test_trend_spherical_significant(self, calculator):
        """Test spherical aberration >= 0.01 mm marked as Significant."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.05, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Significant" in result
    
    def test_trend_distortion_none_when_below_threshold(self, calculator):
        """Test distortion < 0.1% marked as None."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.05, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "Distortion: None" in result
        assert "(0.05 %)" in result  # detail with absolute value
    
    def test_trend_distortion_present_above_threshold(self, calculator):
        """Test distortion >= 0.1% marked as Present."""
        results = {
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 1.5, "chromatic": 0.0,
        }
        calculator.calculate_all_aberrations = Mock(return_value=results)
        result = calculator.get_aberration_summary()
        assert "Distortion: Present" in result
    
    def test_system_target_element_count(self, calculator):
        """Test that system targets display element count."""
        mock_system = Mock()
        mock_system.name = "OpticalSystem"
        mock_system.elements = [Mock(), Mock(), Mock()]
        
        calc_sys = AberrationsCalculator(mock_system, is_system=True)
        calc_sys.calculate_all_aberrations = Mock(return_value={
            "focal_length": 100.0, "f_number": 4.0, "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013, "spherical": 0.0, "coma": 0.0,
            "astigmatism": 0.0, "field_curvature": 0.0, "distortion": 0.0, "chromatic": 0.0,
        })
        result = calc_sys.get_aberration_summary()
        assert "OpticalSystem (3 elements)" in result
        assert "Material: -" in result
