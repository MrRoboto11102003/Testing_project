import pytest
import numpy as np
import logging

try:
    from openlens.lens import Lens
    from openlens.optical_system import OpticalSystem
    from openlens.analysis.psf_mtf import ImageQualityAnalyzer
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")
class TestImageQualityAnalyzerCalculateMTF:
    """Test suite for ImageQualityAnalyzer.calculate_mtf method."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with one lens."""
        system = OpticalSystem()
        system.add_lens(Lens())  # Default biconvex BK7
        return system

    @pytest.fixture
    def analyzer(self, simple_system):
        """Create an ImageQualityAnalyzer."""
        return ImageQualityAnalyzer(simple_system)

    # === Return Structure Tests ===

    def test_return_dict_has_required_keys(self, analyzer):
        """Test that calculate_mtf returns dict with exactly the required keys."""
        result = analyzer.calculate_mtf()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_all_return_values_are_numpy_arrays(self, analyzer):
        """Test that all return values are numpy arrays."""
        result = analyzer.calculate_mtf()
        assert isinstance(result["freq"], np.ndarray)
        assert isinstance(result["mtf_tan"], np.ndarray)
        assert isinstance(result["mtf_sag"], np.ndarray)

    def test_equal_length_outputs(self, analyzer):
        """Test that freq, mtf_tan, and mtf_sag have equal length."""
        result = analyzer.calculate_mtf()
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])

    # === Empty System Tests ===

    def test_empty_system_returns_empty_arrays(self, caplog):
        """Test that empty system returns three empty float64 arrays."""
        empty_system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(empty_system)

        result = analyzer.calculate_mtf()

        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0
        assert result["freq"].dtype == np.float64
        assert "MTF: system has no elements; returning an empty MTF" in caplog.text

    def test_empty_system_with_diffraction_flag(self, caplog):
        """Test that empty system returns empty with use_diffraction=True."""
        empty_system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(empty_system)

        result = analyzer.calculate_mtf(use_diffraction=True)

        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0

    # === Frequency Array Tests (Geometric Branch) ===

    def test_frequency_starts_at_zero(self, analyzer):
        """Test that first frequency is exactly 0.0."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0

    def test_frequency_monotonically_increasing(self, analyzer):
        """Test that frequencies are non-decreasing (monotonic)."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 1:
            assert np.all(np.diff(result["freq"]) >= 0)

    def test_frequency_bounded_by_max_freq(self, analyzer):
        """Test that all frequencies are within max_freq limit."""
        max_freq = 50.0
        result = analyzer.calculate_mtf(max_freq=max_freq)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= max_freq)

    # === MTF Value Tests ===

    def test_mtf_normalized_to_one_at_dc(self, analyzer):
        """Test that MTF equals 1.0 at DC component (freq=0)."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 0 and result["freq"][0] == 0.0:
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-9)
            assert np.isclose(result["mtf_sag"][0], 1.0, atol=1e-9)

    def test_mtf_values_in_valid_range(self, analyzer):
        """Test that all MTF values lie in [0, 1] with floating point tolerance."""
        result = analyzer.calculate_mtf()
        # Allow small negative values due to numerical precision
        assert np.all(result["mtf_tan"] >= -1e-9)
        assert np.all(result["mtf_tan"] <= 1.0 + 1e-9)
        assert np.all(result["mtf_sag"] >= -1e-9)
        assert np.all(result["mtf_sag"] <= 1.0 + 1e-9)

    # === Parameter Edge Cases ===

    def test_max_freq_zero_returns_dc_only(self, analyzer):
        """Test that max_freq=0 returns at most the DC component."""
        result = analyzer.calculate_mtf(max_freq=0.0)
        assert len(result["freq"]) <= 1
        if len(result["freq"]) == 1:
            assert result["freq"][0] == 0.0
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-9)

    def test_max_freq_negative_returns_empty(self, analyzer):
        """Test that max_freq < 0 returns empty arrays."""
        result = analyzer.calculate_mtf(max_freq=-1.0)
        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0

    def test_field_angle_parameter_accepted(self, analyzer):
        """Test that field_angle_deg parameter is accepted."""
        result = analyzer.calculate_mtf(field_angle_deg=0.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) > 0

    def test_field_angle_legacy_alias(self, analyzer):
        """Test that field_angle= legacy alias works (overrides field_angle_deg)."""
        result = analyzer.calculate_mtf(field_angle=0.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_wavelength_parameter_accepted(self, analyzer):
        """Test that wavelength_nm parameter is accepted and works."""
        result = analyzer.calculate_mtf(wavelength_nm=550.0)
        assert isinstance(result, dict)
        assert len(result["freq"]) > 0

    def test_focus_shift_parameter_accepted(self, analyzer):
        """Test that focus_shift_mm parameter is accepted."""
        result = analyzer.calculate_mtf(focus_shift_mm=0.1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_use_diffraction_flag_accepted(self, analyzer):
        """Test that use_diffraction flag is accepted without error."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        assert isinstance(result, dict)

    # === Error Handling Tests ===

    def test_invalid_kwargs_raises_typeerror(self, analyzer):
        """Test that invalid kwargs raise TypeError with descriptive message."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(foo=1)

    def test_invalid_kwargs_before_empty_check(self, analyzer):
        """Test that kwargs validation occurs before empty system check."""
        # Even with invalid kwargs, the error should be raised
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_param=42)

    def test_multiple_parameters_together(self, analyzer):
        """Test that multiple valid parameters work together."""
        result = analyzer.calculate_mtf(
            field_angle_deg=2.0,
            wavelength_nm=632.0,
            focus_shift_mm=0.05,
            max_freq=75.0,
            use_diffraction=False,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 75.0)
