import pytest
import numpy as np
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.lens import Lens


@pytest.fixture
def simple_lens():
    """Create a minimal test lens for analysis."""
    return Lens(
        name="TestLens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7"
    )


@pytest.fixture
def analyzer(simple_lens):
    """Create ImageQualityAnalyzer instance."""
    return ImageQualityAnalyzer(simple_lens)


class TestSimulateImageInputValidation:
    """Test input validation and error handling."""

    def test_1d_array_raises_valueerror(self, analyzer):
        """1-D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="2-D|3-D"):
            analyzer.simulate_image(np.array([1, 2, 3]))

    def test_4d_array_raises_valueerror(self, analyzer):
        """4-D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="2-D|3-D"):
            analyzer.simulate_image(np.ones((10, 10, 3, 2)))

    def test_channel_mismatch_raises_valueerror(self, analyzer):
        """3-D array with wrong channel count should raise ValueError."""
        # Default wavelengths_nm has 3 channels (C, D, F lines)
        with pytest.raises(ValueError, match="channels.*wavelengths"):
            analyzer.simulate_image(np.ones((10, 10, 4), dtype=np.uint8))

    def test_unexpected_kwargs_raise_typeerror(self, analyzer):
        """Unexpected kwargs should raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(
                np.ones((10, 10), dtype=np.uint8),
                invalid_param=123
            )


class TestSimulateImageInputTypes:
    """Test various input array types and dtypes."""

    def test_uint8_grayscale(self, analyzer):
        """uint8 grayscale image should scale from [0, 255] to [0, 1]."""
        img = np.ones((32, 32), dtype=np.uint8) * 200
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert result.shape == (32, 32)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_uint16_grayscale(self, analyzer):
        """uint16 grayscale image should scale correctly."""
        img = np.ones((32, 32), dtype=np.uint16) * 50000
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_float32_grayscale(self, analyzer):
        """float32 grayscale image should work."""
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert result.shape == (32, 32)

    def test_float64_grayscale(self, analyzer):
        """float64 grayscale image should work."""
        img = np.ones((32, 32), dtype=np.float64) * 0.5
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64

    def test_rgb_3channel(self, analyzer):
        """3-D RGB image with 3 channels should work."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 150
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert result.shape == (32, 32, 3)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestSimulateImageOutputProperties:
    """Test output array properties and normalization."""

    def test_output_always_float64(self, analyzer):
        """Output should always be float64 regardless of input dtype."""
        for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            img = np.ones((32, 32), dtype=dtype) * 0.5
            result = analyzer.simulate_image(img)
            assert result.dtype == np.float64

    def test_output_shape_matches_grayscale(self, analyzer):
        """Output 2-D shape should match input grayscale."""
        for shape in [(32, 32), (64, 64), (100, 50)]:
            img = np.ones(shape, dtype=np.uint8)
            result = analyzer.simulate_image(img)
            assert result.shape == shape

    def test_output_shape_matches_rgb(self, analyzer):
        """Output 3-D shape should match input RGB."""
        for shape in [(32, 32, 3), (64, 64, 3)]:
            img = np.ones(shape, dtype=np.uint8)
            result = analyzer.simulate_image(img)
            assert result.shape == shape

    def test_output_clipped_to_unit_range(self, analyzer):
        """Output should be clipped to [0, 1]."""
        # Extreme input values that might overflow after convolution
        img = np.array([[0.0, 2.0], [-1.0, 10.0]], dtype=np.float64)
        result = analyzer.simulate_image(img)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestSimulateImageLegacyKwargs:
    """Test backward compatibility with legacy kwarg names."""

    def test_legacy_pixel_size_kwarg(self, analyzer):
        """Legacy 'pixel_size' kwarg should work alongside 'pixel_size_mm'."""
        img = np.ones((32, 32), dtype=np.uint8)
        result = analyzer.simulate_image(img, pixel_size=0.01)
        assert result.dtype == np.float64

    def test_legacy_wavelengths_kwarg(self, analyzer):
        """Legacy 'wavelengths' kwarg should work alongside 'wavelengths_nm'."""
        img = np.ones((32, 32), dtype=np.uint8)
        result = analyzer.simulate_image(
            img,
            wavelengths=(656.3, 587.6, 486.1)
        )
        assert result.dtype == np.float64


class TestSimulateImageParameters:
    """Test parameter variations and special cases."""

    def test_with_field_angle(self, analyzer):
        """Should correctly handle non-zero field angles."""
        img = np.ones((32, 32), dtype=np.uint8) * 128
        result = analyzer.simulate_image(img, field_angle_deg=5.0)
        assert result.dtype == np.float64
        assert result.shape == (32, 32)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_with_focus_shift(self, analyzer):
        """Should correctly handle defocus shifts."""
        img = np.ones((32, 32), dtype=np.uint8) * 128
        result = analyzer.simulate_image(img, focus_shift_mm=0.1)
        assert result.dtype == np.float64
        assert result.shape == (32, 32)

    def test_with_custom_pixel_size(self, analyzer):
        """Should accept custom pixel size in millimeters."""
        img = np.ones((32, 32), dtype=np.uint8) * 128
        result = analyzer.simulate_image(img, pixel_size_mm=0.01)
        assert result.dtype == np.float64

    def test_with_custom_wavelengths(self, analyzer):
        """Should accept custom wavelength tuple."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        custom_wl = (700.0, 550.0, 450.0)
        result = analyzer.simulate_image(img, wavelengths_nm=custom_wl)
        assert result.dtype == np.float64
        assert result.shape == img.shape

    def test_all_zeros_input(self, analyzer):
        """Should handle all-zero input gracefully."""
        img = np.zeros((32, 32), dtype=np.uint8)
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_all_ones_input_uint8(self, analyzer):
        """Should handle all-maximum uint8 input."""
        img = np.ones((32, 32), dtype=np.uint8) * 255
        result = analyzer.simulate_image(img)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)
