import pytest
import numpy as np
from unittest.mock import Mock, patch
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.constants import WAVELENGTH_GREEN


class TestImageQualityAnalyzerCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf method."""

    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def simple_system(self, simple_lens):
        """Create a simple optical system with one lens."""
        system = OpticalSystem(name="Test System")
        system.add_element(simple_lens)
        return system

    @pytest.fixture
    def analyzer(self, simple_system):
        """Create an ImageQualityAnalyzer with a simple system."""
        return ImageQualityAnalyzer(simple_system)

    def test_empty_system_returns_empty_arrays(self):
        """Empty system should return empty arrays for freq, mtf_tan, mtf_sag."""
        empty_system = OpticalSystem(name="Empty System")
        analyzer = ImageQualityAnalyzer(empty_system)
        
        result = analyzer.calculate_mtf()
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0
        assert isinstance(result["freq"], np.ndarray)

    def test_return_dict_structure_geometric_mode(self, analyzer):
        """Return dict should have freq, mtf_tan, mtf_sag keys with numpy arrays."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_return_dict_structure_diffraction_mode(self, analyzer):
        """Diffraction mode should also return valid dict structure."""
        result = analyzer.calculate_mtf(use_diffraction=True)
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_mtf_array_lengths_match(self, analyzer):
        """All returned arrays should have same length."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        
        assert len(result["freq"]) == len(result["mtf_tan"])
        assert len(result["freq"]) == len(result["mtf_sag"])

    def test_frequencies_non_negative(self, analyzer):
        """All returned frequencies should be non-negative."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] >= 0)

    def test_max_freq_parameter_filters_output(self, analyzer):
        """max_freq parameter should limit frequencies returned."""
        result_limited = analyzer.calculate_mtf(max_freq=10.0, use_diffraction=False)
        result_unlimited = analyzer.calculate_mtf(max_freq=1000.0, use_diffraction=False)
        
        if len(result_limited["freq"]) > 0:
            assert np.all(result_limited["freq"] <= 10.0)
        
        # Limited should have <= frequencies than unlimited
        assert len(result_limited["freq"]) <= len(result_unlimited["freq"])

    def test_mtf_values_normalized_range(self, analyzer):
        """MTF values should be in valid range [0, 1] after normalization."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        
        # Allow small numerical errors (±1%)
        tolerance = 0.01
        
        if len(result["mtf_tan"]) > 0:
            assert np.all(result["mtf_tan"] >= -tolerance)
            assert np.all(result["mtf_tan"] <= 1.0 + tolerance)
        
        if len(result["mtf_sag"]) > 0:
            assert np.all(result["mtf_sag"] >= -tolerance)
            assert np.all(result["mtf_sag"] <= 1.0 + tolerance)

    def test_field_angle_parameter_accepted(self, analyzer):
        """field_angle_deg parameter should be accepted without error."""
        result = analyzer.calculate_mtf(field_angle_deg=5.0, use_diffraction=False)
        assert isinstance(result, dict)
        assert "freq" in result

    def test_wavelength_parameter_accepted(self, analyzer):
        """wavelength_nm parameter should be accepted without error."""
        result = analyzer.calculate_mtf(wavelength_nm=650.0, use_diffraction=False)
        assert isinstance(result, dict)
        assert "freq" in result

    def test_focus_shift_parameter_accepted(self, analyzer):
        """focus_shift_mm parameter should be accepted without error."""
        result = analyzer.calculate_mtf(focus_shift_mm=0.5, use_diffraction=False)
        assert isinstance(result, dict)
        assert "freq" in result

    def test_legacy_field_angle_kwarg_compatibility(self, analyzer):
        """Legacy field_angle kwarg should work (not field_angle_deg)."""
        # Should handle field_angle as backward-compatible parameter name
        result = analyzer.calculate_mtf(field_angle=5.0, use_diffraction=False)
        assert isinstance(result, dict)
        assert "freq" in result

    def test_invalid_kwargs_raise_type_error(self, analyzer):
        """Unexpected kwargs should raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_parameter=True, use_diffraction=False)

    def test_multiple_parameters_combined(self, analyzer):
        """Multiple parameters should be accepted together."""
        result = analyzer.calculate_mtf(
            field_angle_deg=3.0,
            wavelength_nm=550.0,
            focus_shift_mm=0.1,
            max_freq=50.0,
            use_diffraction=False,
        )
        
        assert isinstance(result, dict)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 50.0)
