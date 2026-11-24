import pytest
import numpy as np
from unittest.mock import Mock, patch

try:
    from openlens.lens import Lens
    from openlens.optical_system import OpticalSystem
    from openlens.analysis.psf_mtf import ImageQualityAnalyzer
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False


pytestmark = pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")


class TestCalculateMTF:
    """Test suite for ImageQualityAnalyzer.calculate_mtf"""

    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex test lens"""
        return Lens(
            name="Test Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def optical_system(self, simple_lens):
        """Create a test optical system with one element"""
        system = OpticalSystem(name="Test System")
        system.add_lens(simple_lens)
        return system

    @pytest.fixture
    def analyzer(self, optical_system):
        """Create an ImageQualityAnalyzer instance"""
        return ImageQualityAnalyzer(optical_system)

    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system"""
        return OpticalSystem(name="Empty System")

    def test_empty_system_returns_empty_arrays(self):
        """Empty system returns empty freq/mtf arrays"""
        empty_sys = OpticalSystem(name="Empty")
        analyzer = ImageQualityAnalyzer(empty_sys)
        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0

    def test_returns_dict_with_required_keys(self, analyzer):
        """calculate_mtf returns dict with freq, mtf_tan, mtf_sag keys"""
        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        required_keys = {"freq", "mtf_tan", "mtf_sag"}
        assert set(result.keys()) == required_keys

    def test_returned_arrays_are_numpy_arrays(self, analyzer):
        """Return values are numpy arrays"""
        result = analyzer.calculate_mtf()

        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_arrays_equal_length(self, analyzer):
        """freq, mtf_tan, mtf_sag arrays have equal length"""
        result = analyzer.calculate_mtf()

        assert len(result["freq"]) == len(result["mtf_tan"])
        assert len(result["freq"]) == len(result["mtf_sag"])

    def test_max_freq_filters_output(self, analyzer):
        """max_freq parameter filters frequencies to specified limit"""
        result_unlimited = analyzer.calculate_mtf(max_freq=200.0)
        result_limited = analyzer.calculate_mtf(max_freq=10.0)

        if len(result_limited["freq"]) > 0:
            assert np.all(result_limited["freq"] <= 10.0)

        # Limited result has fewer or equal points
        assert len(result_limited["freq"]) <= len(result_unlimited["freq"])

    def test_field_angle_deg_parameter(self, analyzer):
        """field_angle_deg parameter accepted and produces valid output"""
        result_0 = analyzer.calculate_mtf(field_angle_deg=0.0)
        result_5 = analyzer.calculate_mtf(field_angle_deg=5.0)

        for result in [result_0, result_5]:
            assert isinstance(result, dict)
            assert all(k in result for k in ["freq", "mtf_tan", "mtf_sag"])

    def test_wavelength_nm_parameter(self, analyzer):
        """wavelength_nm parameter accepted and produces valid output"""
        result_green = analyzer.calculate_mtf(wavelength_nm=550.0)
        result_red = analyzer.calculate_mtf(wavelength_nm=632.8)

        for result in [result_green, result_red]:
            assert isinstance(result, dict)
            assert "freq" in result

    def test_focus_shift_mm_parameter(self, analyzer):
        """focus_shift_mm parameter accepted without error"""
        result_0 = analyzer.calculate_mtf(focus_shift_mm=0.0)
        result_defocus = analyzer.calculate_mtf(focus_shift_mm=0.05)

        assert isinstance(result_0, dict)
        assert isinstance(result_defocus, dict)

    def test_legacy_field_angle_kwarg_alias(self, analyzer):
        """Legacy 'field_angle' kwarg handled as field_angle_deg"""
        result = analyzer.calculate_mtf(field_angle=5.0)

        assert isinstance(result, dict)
        assert "freq" in result

    def test_unexpected_kwargs_raise_typeerror(self, analyzer):
        """Unexpected kwargs raise TypeError with 'Unexpected kwargs' message"""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_parameter=42)

    def test_mtf_values_normalized_between_0_and_1(self, analyzer):
        """MTF values are normalized between 0.0 and 1.0"""
        result = analyzer.calculate_mtf()

        if len(result["mtf_tan"]) > 0:
            assert np.all(result["mtf_tan"] >= 0.0)
            assert np.all(result["mtf_tan"] <= 1.0)

        if len(result["mtf_sag"]) > 0:
            assert np.all(result["mtf_sag"] >= 0.0)
            assert np.all(result["mtf_sag"] <= 1.0)

    def test_frequency_array_monotonic_increasing(self, analyzer):
        """Frequency array is monotonically increasing and non-negative"""
        result = analyzer.calculate_mtf()

        if len(result["freq"]) > 1:
            assert np.all(result["freq"] >= 0.0)
            diffs = np.diff(result["freq"])
            assert np.all(diffs >= 0.0)

    def test_use_diffraction_parameter_true(self, analyzer):
        """use_diffraction=True produces valid output"""
        result = analyzer.calculate_mtf(use_diffraction=True)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_use_diffraction_parameter_false(self, analyzer):
        """use_diffraction=False produces valid output"""
        result = analyzer.calculate_mtf(use_diffraction=False)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_multiple_parameters_combined(self, analyzer):
        """Multiple parameters work together without conflict"""
        result = analyzer.calculate_mtf(
            field_angle_deg=3.0,
            wavelength_nm=587.6,
            focus_shift_mm=0.01,
            max_freq=50.0,
            use_diffraction=False,
        )

        assert isinstance(result, dict)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 50.0)
