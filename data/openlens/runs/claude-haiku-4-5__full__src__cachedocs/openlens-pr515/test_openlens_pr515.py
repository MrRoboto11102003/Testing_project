import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.aberrations import AberrationsCalculator


@pytest.fixture
def biconvex_lens():
    """Simple biconvex lens for testing."""
    return Lens(
        name="Test Biconvex",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def optical_system():
    """Two-element optical system for testing."""
    system = OpticalSystem(name="Test Assembly")
    lens1 = Lens(
        name="L1",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )
    lens2 = Lens(
        name="L2",
        radius_of_curvature_1=-50.0,
        radius_of_curvature_2=50.0,
        thickness=4.0,
        diameter=20.0,
        material="BK7",
    )
    system.add_lens(lens1)
    system.add_lens(lens2, air_gap_before=5.0)
    return system


class TestGetAberrationSummary:
    """Tests for AberrationsCalculator.get_aberration_summary()"""

    def test_returns_string(self, biconvex_lens):
        """Test that method returns a non-empty string."""
        calculator = AberrationsCalculator(biconvex_lens)
        result = calculator.get_aberration_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_contains_title_and_main_sections(self, biconvex_lens):
        """Test that summary contains all major document sections."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        
        expected_sections = [
            "LENS ABERRATIONS ANALYSIS",
            "BASIC PARAMETERS",
            "PRIMARY ABERRATIONS (Seidel)",
            "CHROMATIC ABERRATION",
            "INTERPRETATION:",
        ]
        for section in expected_sections:
            assert section in summary, f"Missing section: {section}"

    def test_summary_includes_all_optical_metrics(self, biconvex_lens):
        """Test that summary includes all documented optical metrics."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        
        expected_metrics = [
            "Focal Length:",
            "F-number (f/#):",
            "Numerical Aperture:",
            "Airy Disk Diameter:",
            "Spherical Aberration:",
            "Coma",
            "Astigmatism",
            "Field Curvature:",
            "Distortion",
            "Longitudinal CA:",
        ]
        for metric in expected_metrics:
            assert metric in summary, f"Missing metric: {metric}"

    def test_object_distance_none_for_infinity(self, biconvex_lens):
        """Test that object_distance=None (at infinity) is accepted."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary(object_distance=None)
        assert isinstance(summary, str)
        assert "LENS ABERRATIONS ANALYSIS" in summary

    def test_object_distance_finite_value(self, biconvex_lens):
        """Test that finite object_distance values are accepted."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary(object_distance=100.0)
        assert isinstance(summary, str)
        assert "LENS ABERRATIONS ANALYSIS" in summary

    def test_default_field_angle_is_five_degrees(self, biconvex_lens):
        """Test that default field_angle=5.0 degrees appears in output."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        # Field angle should appear in the Coma and Distortion lines
        assert "@ 5" in summary

    def test_custom_field_angle_parameter(self, biconvex_lens):
        """Test that custom field_angle parameter is used."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary(field_angle=12.0)
        assert "@ 12" in summary

    def test_wavelength_nm_parameter_accepted(self, biconvex_lens):
        """Test that wavelength_nm parameter is accepted and used."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary_550 = calculator.get_aberration_summary(wavelength_nm=550.0)
        summary_632 = calculator.get_aberration_summary(wavelength_nm=632.8)
        
        assert isinstance(summary_550, str)
        assert isinstance(summary_632, str)
        # Both should produce valid summaries
        assert "LENS ABERRATIONS ANALYSIS" in summary_550
        assert "LENS ABERRATIONS ANALYSIS" in summary_632

    def test_backward_compatibility_field_angle_deg_kwarg(self, biconvex_lens):
        """Test backward compatibility with field_angle_deg keyword argument."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary(field_angle_deg=8.5)
        # Should use field_angle_deg from kwargs
        assert "@ 8.5" in summary

    def test_single_lens_displays_name_and_material(self, biconvex_lens):
        """Test that single lens summary shows lens name and material."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        
        assert "Test Biconvex" in summary
        assert "BK7" in summary

    def test_multi_element_system_displays_element_count(self, optical_system):
        """Test that multi-element system shows name and element count."""
        calculator = AberrationsCalculator(optical_system)
        summary = calculator.get_aberration_summary()
        
        assert "Test Assembly" in summary
        assert "2 elements" in summary

    def test_unavailable_metrics_handled_gracefully(self, biconvex_lens):
        """Test that None/unavailable metric values don't cause errors."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        
        # Should produce valid string even if some metrics are unavailable
        assert isinstance(summary, str)
        assert "LENS ABERRATIONS ANALYSIS" in summary

    def test_output_format_uses_box_drawing_characters(self, biconvex_lens):
        """Test that summary uses box-drawing characters for formatting."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary = calculator.get_aberration_summary()
        
        # Verify box-drawing characters are present
        assert "╔" in summary  # top-left corner
        assert "╚" in summary  # bottom-left corner
        assert "║" in summary  # vertical line

    def test_multiple_calls_with_same_params_are_consistent(self, biconvex_lens):
        """Test that identical calls produce identical output."""
        calculator = AberrationsCalculator(biconvex_lens)
        summary1 = calculator.get_aberration_summary(field_angle=5.0)
        summary2 = calculator.get_aberration_summary(field_angle=5.0)
        
        assert summary1 == summary2
