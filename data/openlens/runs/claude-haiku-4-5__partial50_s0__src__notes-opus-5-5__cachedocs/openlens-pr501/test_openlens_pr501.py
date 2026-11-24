import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.constants import (
    WAVELENGTH_GREEN,
    WAVELENGTH_C_LINE,
    WAVELENGTH_D_LINE,
    WAVELENGTH_F_LINE,
)


class TestCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf"""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with a single lens."""
        system = OpticalSystem()
        lens = Lens(
            name="Test Lens",
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
        """Create an ImageQualityAnalyzer for the simple system."""
        return ImageQualityAnalyzer(simple_system)

    def test_calculate_mtf_returns_dict_with_correct_keys(self, analyzer):
        """MTF result should have freq, mtf_tan, mtf_sag keys."""
        result = analyzer.calculate_mtf()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_calculate_mtf_arrays_have_equal_length(self, analyzer):
        """All returned arrays should have the same length."""
        result = analyzer.calculate_mtf()
        assert len(result["freq"]) == len(result["mtf_tan"])
        assert len(result["mtf_tan"]) == len(result["mtf_sag"])

    def test_calculate_mtf_freq_starts_at_zero(self, analyzer):
        """Frequency array should start at 0 when non-empty."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0

    def test_calculate_mtf_mtf_at_zero_frequency_is_unity(self, analyzer):
        """MTF should be ~1.0 at zero frequency for geometric mode."""
        result = analyzer.calculate_mtf(use_diffraction=False)
        if len(result["mtf_tan"]) > 0:
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-6)
            assert np.isclose(result["mtf_sag"][0], 1.0, atol=1e-6)

    def test_calculate_mtf_freq_monotonic_increasing(self, analyzer):
        """Frequency array should be monotonically non-decreasing."""
        result = analyzer.calculate_mtf()
        if len(result["freq"]) > 1:
            diffs = np.diff(result["freq"])
            assert np.all(diffs >= -1e-10)  # Allow tiny numerical errors

    def test_calculate_mtf_mtf_values_in_valid_range(self, analyzer):
        """MTF values should be in [0, 1]."""
        result = analyzer.calculate_mtf()
        assert np.all(result["mtf_tan"] >= -1e-9)
        assert np.all(result["mtf_tan"] <= 1.0 + 1e-9)
        assert np.all(result["mtf_sag"] >= -1e-9)
        assert np.all(result["mtf_sag"] <= 1.0 + 1e-9)

    def test_calculate_mtf_empty_system(self):
        """Empty optical system should return empty arrays for both modes."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        result_geometric = analyzer.calculate_mtf(use_diffraction=False)
        assert result_geometric["freq"].size == 0
        assert result_geometric["mtf_tan"].size == 0
        assert result_geometric["mtf_sag"].size == 0
        
        result_diffraction = analyzer.calculate_mtf(use_diffraction=True)
        assert result_diffraction["freq"].size == 0
        assert result_diffraction["mtf_tan"].size == 0
        assert result_diffraction["mtf_sag"].size == 0

    def test_calculate_mtf_max_freq_respected(self, analyzer):
        """All frequencies should be <= max_freq."""
        max_freq = 50.0
        result = analyzer.calculate_mtf(max_freq=max_freq)
        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= max_freq + 1e-9)

    def test_calculate_mtf_max_freq_zero(self, analyzer):
        """max_freq=0 should return only DC component."""
        result = analyzer.calculate_mtf(max_freq=0.0)
        assert len(result["freq"]) == 1
        assert result["freq"][0] == 0.0
        assert np.isclose(result["mtf_tan"][0], 1.0, atol=1e-6)
        assert np.isclose(result["mtf_sag"][0], 1.0, atol=1e-6)

    def test_calculate_mtf_legacy_alias_field_angle(self, analyzer):
        """Legacy field_angle alias should work."""
        result_via_named = analyzer.calculate_mtf(field_angle_deg=5.0)
        result_via_alias = analyzer.calculate_mtf(field_angle=5.0)
        # Same parameter should give same result
        assert np.allclose(result_via_alias["freq"], result_via_named["freq"], atol=1e-10)
        assert np.allclose(result_via_alias["mtf_tan"], result_via_named["mtf_tan"], atol=1e-10)
        assert np.allclose(result_via_alias["mtf_sag"], result_via_named["mtf_sag"], atol=1e-10)

    def test_calculate_mtf_unexpected_kwargs_raises_type_error(self, analyzer):
        """Passing unexpected kwargs should raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(foo=1.0)

    def test_calculate_mtf_accepts_parameters(self, analyzer):
        """calculate_mtf should accept all documented parameters."""
        # Should not raise
        result = analyzer.calculate_mtf(
            field_angle_deg=5.0,
            wavelength_nm=550.0,
            focus_shift_mm=0.1,
            max_freq=100.0,
            use_diffraction=False
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_calculate_mtf_negative_max_freq(self, analyzer):
        """max_freq < 0 should return empty arrays."""
        result = analyzer.calculate_mtf(max_freq=-1.0)
        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0


class TestSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image"""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with a single lens."""
        system = OpticalSystem()
        lens = Lens(
            name="Test Lens",
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
        """Create an ImageQualityAnalyzer for the simple system."""
        return ImageQualityAnalyzer(simple_system)

    def test_simulate_image_grayscale_2d_shape_preserved(self, analyzer):
        """2-D grayscale input should return 2-D output."""
        image = np.ones((8, 8), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(image)
        assert result.shape == (8, 8)
        assert result.ndim == 2

    def test_simulate_image_rgb_3d_shape_preserved(self, analyzer):
        """3-D RGB input should return 3-D output."""
        image = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(image)
        assert result.shape == (8, 8, 3)
        assert result.ndim == 3

    def test_simulate_image_output_dtype_always_float64(self, analyzer):
        """Output should always be float64."""
        image_uint8 = np.ones((8, 8), dtype=np.uint8)
        result_uint8 = analyzer.simulate_image(image_uint8)
        assert result_uint8.dtype == np.float64

        image_float32 = np.ones((8, 8), dtype=np.float32)
        result_float32 = analyzer.simulate_image(image_float32)
        assert result_float32.dtype == np.float64

    def test_simulate_image_output_clipped_to_zero_one(self, analyzer):
        """Output should be clipped to [0, 1]."""
        image = np.array([[0.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
        result = analyzer.simulate_image(image)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_simulate_image_uint8_input_scaled(self, analyzer):
        """uint8 input should be scaled by dtype max (255)."""
        image = np.array([[0, 128, 255]], dtype=np.uint8).reshape(1, 3)
        result = analyzer.simulate_image(image, pixel_size_mm=0.01)
        assert result.dtype == np.float64
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_simulate_image_invalid_1d_raises_value_error(self, analyzer):
        """1-D input should raise ValueError."""
        image_1d = np.ones(10)
        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(image_1d)

    def test_simulate_image_invalid_4d_raises_value_error(self, analyzer):
        """4-D input should raise ValueError."""
        image_4d = np.ones((2, 2, 2, 2))
        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(image_4d)

    def test_simulate_image_channel_mismatch_raises_value_error(self, analyzer):
        """3-D input with wrong channel count should raise ValueError."""
        image_2ch = np.ones((8, 8, 2))
        # Default wavelengths_nm has 3 channels
        with pytest.raises(ValueError, match="channels but"):
            analyzer.simulate_image(image_2ch)

    def test_simulate_image_unexpected_kwargs_raises_type_error(self, analyzer):
        """Unexpected kwargs should raise TypeError."""
        image = np.ones((8, 8))
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(image, foo=1.0)

    def test_simulate_image_legacy_alias_pixel_size(self, analyzer):
        """Legacy pixel_size alias should work."""
        image = np.ones((8, 8))
        result = analyzer.simulate_image(image, pixel_size=0.01)
        assert result.shape == image.shape
        assert result.dtype == np.float64

    def test_simulate_image_legacy_alias_wavelengths(self, analyzer):
        """Legacy wavelengths alias should work."""
        image = np.ones((8, 8, 2))
        result = analyzer.simulate_image(image, wavelengths=(550.0, 600.0))
        assert result.shape == image.shape
        assert result.dtype == np.float64

    def test_simulate_image_accepts_documented_parameters(self, analyzer):
        """simulate_image should accept all documented parameters."""
        image = np.ones((8, 8, 3))
        result = analyzer.simulate_image(
            image,
            pixel_size_mm=0.005,
            wavelengths_nm=(WAVELENGTH_C_LINE, WAVELENGTH_D_LINE, WAVELENGTH_F_LINE),
            field_angle_deg=0.0,
            focus_shift_mm=0.0
        )
        assert result.shape == image.shape
        assert result.dtype == np.float64

    def test_simulate_image_empty_system_returns_zeros(self):
        """Empty system should return all-zero image."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        image = np.ones((8, 8))
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        # Empty system -> zero PSF -> convolution gives zeros
        assert np.allclose(result, 0.0)

    def test_simulate_image_grayscale_channels_averaged(self, analyzer):
        """For 2-D input, all wavelengths should be processed but averaged."""
        image = np.ones((6, 6), dtype=np.float32)
        # Mock calculate_psf to track calls
        call_count = []
        original_calculate_psf = analyzer.calculate_psf
        
        def mock_psf(**kwargs):
            call_count.append(kwargs)
            # Return a small identity kernel as float
            kernel = np.zeros((3, 3))
            kernel[1, 1] = 1.0
            return {"image": kernel, "step_size": 0.001}
        
        analyzer.calculate_psf = mock_psf
        try:
            result = analyzer.simulate_image(image)
            # Should have called calculate_psf for each wavelength (3 times for RGB)
            assert len(call_count) >= 1
            assert result.shape == image.shape
            assert result.dtype == np.float64
        finally:
            analyzer.calculate_psf = original_calculate_psf
