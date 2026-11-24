"""Tests for openlens.aberrations.AberrationsCalculator.get_aberration_summary"""

import pytest
import math
from unittest.mock import Mock
from openlens.lens import Lens
from openlens.aberrations import AberrationsCalculator


class TestGetAberrationSummary:
    """Test suite for AberrationsCalculator.get_aberration_summary"""

    @pytest.fixture
    def test_lens(self):
        """Create a simple test lens"""
        return Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def calculator(self, test_lens):
        """Create an AberrationsCalculator instance"""
        return AberrationsCalculator(test_lens)

    def test_returns_string(self, calculator):
        """Result should always be a string"""
        result = calculator.get_aberration_summary()
        assert isinstance(result, str)

    def test_starts_and_ends_with_newline(self, calculator):
        """Report should start and end with newline character"""
        result = calculator.get_aberration_summary()
        assert result.startswith("\n")
        assert result.endswith("\n")

    def test_contains_required_sections(self, calculator):
        """Report should contain all major sections"""
        result = calculator.get_aberration_summary()
        assert "LENS ABERRATIONS ANALYSIS" in result
        assert "BASIC PARAMETERS" in result
        assert "PRIMARY ABERRATIONS (Seidel)" in result
        assert "CHROMATIC ABERRATION" in result
        assert "INTERPRETATION:" in result

    def test_contains_lens_metadata(self, calculator, test_lens):
        """Report should display lens name and material"""
        result = calculator.get_aberration_summary()
        assert test_lens.name in result
        assert test_lens.material in result

    def test_field_angle_in_labels(self, calculator):
        """Field angle parameter should appear in coma/astigmatism/distortion labels"""
        result = calculator.get_aberration_summary(field_angle=7.5)
        # Count how many times the field angle appears in labels
        assert "(@ 7.5°)" in result

    def test_field_angle_deg_kwarg_computation(self, calculator):
        """field_angle_deg kwarg should control computation, not labels"""
        # Track which angle is actually passed to calculate_all_aberrations
        call_args = {}

        def mock_calc(od, fa, wavelength_nm=None):
            call_args["field_angle"] = fa
            call_args["wavelength_nm"] = wavelength_nm
            return {
                "focal_length": 100.0,
                "f_number": 4.0,
                "numerical_aperture": 0.125,
                "airy_disk_diameter": 0.0013,
                "spherical": 0.0,
                "coma": 0.0,
                "astigmatism": 0.0,
                "field_curvature": 0.0,
                "distortion": 0.0,
                "chromatic": 0.0,
            }

        calculator.calculate_all_aberrations = mock_calc
        result = calculator.get_aberration_summary(
            field_angle=3.0, field_angle_deg=7.0
        )

        # Computed field angle should be 7.0 (from field_angle_deg kwarg)
        assert call_args["field_angle"] == 7.0
        # But label should still show 3.0 (from field_angle parameter)
        assert "(@ 3.0°)" in result

    def test_object_distance_not_none_raises_valueerror(self, calculator):
        """get_aberration_summary should raise ValueError for non-None object_distance"""
        with pytest.raises(ValueError, match="object distance is not supported"):
            calculator.get_aberration_summary(object_distance=100.0)

    def test_object_distance_none_allowed(self, calculator):
        """object_distance=None should be accepted without error"""
        result = calculator.get_aberration_summary(object_distance=None)
        assert isinstance(result, str)

    def test_error_result_format(self, calculator):
        """If calculate_all_aberrations returns error key, should return error string"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "error": "Cannot calculate focal length (zero optical power)"
        }
        result = calculator.get_aberration_summary()
        assert result == "Error: Cannot calculate focal length (zero optical power)"

    def test_wavelength_parameter_forwarded(self, calculator):
        """wavelength_nm parameter should be forwarded to calculate_all_aberrations"""
        call_args = {}

        def mock_calc(od, fa, wavelength_nm=None):
            call_args["wavelength_nm"] = wavelength_nm
            return {
                "focal_length": 100.0,
                "f_number": 4.0,
                "numerical_aperture": 0.125,
                "airy_disk_diameter": 0.0013,
                "spherical": 0.0,
                "coma": 0.0,
                "astigmatism": 0.0,
                "field_curvature": 0.0,
                "distortion": 0.0,
                "chromatic": 0.0,
            }

        calculator.calculate_all_aberrations = mock_calc
        calculator.get_aberration_summary(wavelength_nm=600.0)
        assert call_args["wavelength_nm"] == 600.0

    def test_none_values_render_as_unavailable(self, calculator):
        """None values in results should render as 'unavailable' not crash"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": None,
            "numerical_aperture": None,
            "airy_disk_diameter": 0.0013,
            "spherical": None,
            "coma": None,
            "astigmatism": None,
            "field_curvature": None,
            "distortion": 0.0,
            "chromatic": None,
        }
        result = calculator.get_aberration_summary()
        assert "unavailable" in result
        # Should not raise or produce garbage output
        assert isinstance(result, str)
        assert len(result) > 100

    def test_nan_values_render_as_unavailable(self, calculator):
        """NaN values should render as 'unavailable' not crash"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": float("nan"),
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.005,
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": float("nan"),
            "distortion": 0.0,
            "chromatic": 0.0,
        }
        result = calculator.get_aberration_summary()
        assert "unavailable" in result

    def test_negative_values_retain_sign(self, calculator):
        """Negative metric values should retain sign in formatting"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": -50.0,
            "f_number": 2.0,
            "numerical_aperture": 0.5,
            "airy_disk_diameter": 0.001,
            "spherical": -0.0123,
            "coma": 0.0,
            "astigmatism": -0.01,
            "field_curvature": -2.5,
            "distortion": -0.3,
            "chromatic": -0.15,
        }
        result = calculator.get_aberration_summary()
        assert "-50.00" in result
        assert "-0.0123" in result

    def test_system_target_element_count(self):
        """For system targets, name should include element count"""
        # Create a mock optical system
        system = Mock()
        system.name = "My System"
        system.elements = [Mock(), Mock(), Mock()]  # 3 elements
        system.is_system = True

        calculator = AberrationsCalculator(system)
        # Need to set is_system on calculator itself
        calculator.is_system = True
        calculator.target = system

        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.0,
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": 0.0,
            "distortion": 0.0,
            "chromatic": 0.0,
        }

        result = calculator.get_aberration_summary()
        assert "My System (3 elements)" in result
        assert "Material: -" in result

    def test_metric_formatting_decimal_places(self, calculator):
        """Metrics should be formatted with correct decimal places"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.005,
            "coma": 0.0,
            "astigmatism": 0.02,
            "field_curvature": 1.5,
            "distortion": 0.5,
            "chromatic": 0.3,
        }
        result = calculator.get_aberration_summary()
        # Check format specs: focal_length uses .2f
        assert "100.00" in result
        # f_number uses .2f
        assert "4.00" in result
        # numerical_aperture uses .4f
        assert "0.1250" in result

    def test_trend_spherical_aberration_ranges(self, calculator):
        """Spherical aberration should categorize as Negligible/Moderate/Significant"""
        # Test Negligible (< 0.001)
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.0005,
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": 0.0,
            "distortion": 0.0,
            "chromatic": 0.0,
        }
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Negligible" in result

        # Test Moderate (0.001 to 0.01)
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.005,
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": 0.0,
            "distortion": 0.0,
            "chromatic": 0.0,
        }
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Moderate" in result

        # Test Significant (> 0.01)
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": 0.02,
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": 0.0,
            "distortion": 0.0,
            "chromatic": 0.0,
        }
        result = calculator.get_aberration_summary()
        assert "Spherical Aberration: Significant" in result

    def test_detail_uses_absolute_value(self, calculator):
        """Detail section should show absolute values with proper format"""
        calculator.calculate_all_aberrations = lambda od, fa, wavelength_nm=None: {
            "focal_length": 100.0,
            "f_number": 4.0,
            "numerical_aperture": 0.125,
            "airy_disk_diameter": 0.0013,
            "spherical": -0.005,  # negative input
            "coma": 0.0,
            "astigmatism": 0.0,
            "field_curvature": 0.0,
            "distortion": 0.0,
            "chromatic": 0.0,
        }
        result = calculator.get_aberration_summary()
        # Detail should show abs value as (0.0050 mm)
        assert "(0.0050 mm)" in result
