import pytest
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestImageQualityAnalyzerCalculateMTF:
    """Unit tests for ImageQualityAnalyzer.calculate_mtf()"""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with one lens."""
        system = OpticalSystem()
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system.add_lens(lens)
        return system

    @pytest.fixture
    def analyzer(self, simple_system):
        """Create an ImageQualityAnalyzer."""
        return ImageQualityAnalyzer(simple_system)

    # --- Return structure and types ---

    def test_calculate_mtf_returns_dict_with_required_keys(self, analyzer):
        """Test that calculate_mtf returns a dict with freq, mtf_tan, mtf_sag."""
        result = analyzer.calculate_mtf()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_calculate_mtf_returns_numpy_arrays(self, analyzer):
        """Test that all return values are numpy arrays."""
        result = analyzer.calculate_mtf()
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_calculate_mtf_arrays_equal_length(self, analyzer):
        """Test that all output arrays have equal length."""
        result = analyzer.calculate_mtf()
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])

    # --- Empty system edge case ---

    def test_calculate_mtf_empty_system(self):
        """Test that empty system returns empty arrays for both geometric and diffraction."""
        empty_system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(empty_system)

        result_geom = analyzer.calculate_mtf()
        result_diff = analyzer.calculate_mtf(use_diffraction=True)

        for result in [result_geom, result_diff]:
            assert result["freq"].size == 0
            assert result["mtf_tan"].size == 0
            assert result["mtf_sag"].size == 0

    # --- Frequency array properties ---

    def test_calculate_mtf_frequency_starts_at_zero(self, analyzer):
        """Test that frequency array begins at 0 Hz."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0

    def test_calculate_mtf_frequency_monotonic_increasing(self, analyzer):
        """Test that frequency array is monotonically non-decreasing."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 1:
            assert np.all(np.diff(result["freq"]) >= -1e-9)

    def test_calculate_mtf_respects_max_freq_limit(self, analyzer):
        """Test that all frequencies are <= max_freq."""
        max_freq = 50.0
        result = analyzer.calculate_mtf(max_freq=max_freq)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= max_freq + 1e-9)

    # --- MTF value range validation ---

    def test_calculate_mtf_values_in_valid_range(self, analyzer):
        """Test that MTF values lie in [0, 1]."""
        result = analyzer.calculate_mtf()
        assert np.all(result["mtf_tan"] >= -1e-9)
        assert np.all(result["mtf_tan"] <= 1.0 + 1e-9)
        assert np.all(result["mtf_sag"] >= -1e-9)
        assert np.all(result["mtf_sag"] <= 1.0 + 1e-9)

    def test_calculate_mtf_dc_component_unity(self, analyzer):
        """Test that MTF at DC (zero frequency) is approximately 1.0."""
        result = analyzer.calculate_mtf()
        if len(result["mtf_tan"]) > 0:
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-6)
            assert np.isclose(result["mtf_sag"][0], 1.0, atol=1e-6)

    # --- max_freq edge cases ---

    def test_calculate_mtf_max_freq_zero(self, analyzer):
        """Test that max_freq=0 returns only the DC component."""
        result = analyzer.calculate_mtf(max_freq=0)
        assert len(result["freq"]) == 1
        assert result["freq"][0] == 0.0
        assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-6)

    def test_calculate_mtf_max_freq_negative(self, analyzer):
        """Test that negative max_freq returns empty arrays."""
        result = analyzer.calculate_mtf(max_freq=-1.0)
        assert len(result["freq"]) == 0
        assert len(result["mtf_tan"]) == 0
        assert len(result["mtf_sag"]) == 0

    # --- Kwargs handling ---

    def test_calculate_mtf_field_angle_legacy_alias(self, analyzer):
        """Test that field_angle kwarg works as an alias for field_angle_deg."""
        result1 = analyzer.calculate_mtf(field_angle_deg=5.0)
        result2 = analyzer.calculate_mtf(field_angle=5.0)
        # Same field angle should produce same-length results
        assert len(result1["freq"]) == len(result2["freq"])

    def test_calculate_mtf_field_angle_alias_overrides(self, analyzer):
        """Test that field_angle kwarg overrides field_angle_deg when both provided."""
        result = analyzer.calculate_mtf(field_angle_deg=1.0, field_angle=5.0)
        result_expected = analyzer.calculate_mtf(field_angle=5.0)
        # field_angle=5.0 should be the effective value
        assert len(result["freq"]) == len(result_expected["freq"])

    def test_calculate_mtf_unexpected_kwargs_raises_typeerror(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_param=42)

    def test_calculate_mtf_multiple_unexpected_kwargs_raises(self, analyzer):
        """Test that multiple unexpected kwargs are caught and reported."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(foo=1, bar=2)

    # --- Parameter variations ---

    def test_calculate_mtf_with_focus_shift(self, analyzer):
        """Test that focus_shift_mm parameter is accepted."""
        result = analyzer.calculate_mtf(focus_shift_mm=2.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == len(result["mtf_tan"])

    def test_calculate_mtf_diffraction_mode(self, analyzer):
        """Test that diffraction mode executes and returns valid structure."""
        result = analyzer.calculate_mtf(use_diffraction=True)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])
        # Check that frequencies respect the limit
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0
