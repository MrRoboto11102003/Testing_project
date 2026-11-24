import pytest
import numpy as np
from openlens.analysis.psf_mtf import ImageQualityAnalyzer


class TestImageQualityAnalyzerSimulateImage:
    """Comprehensive test suite for ImageQualityAnalyzer.simulate_image method."""

    @pytest.fixture
    def analyzer(self):
        """Create an ImageQualityAnalyzer instance for testing."""
        return ImageQualityAnalyzer()

    # Basic functionality tests
    def test_grayscale_float_input(self, analyzer):
        """Test with 2D grayscale float image."""
        image = np.random.rand(32, 32).astype(np.float64)
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_rgb_float_input(self, analyzer):
        """Test with 3D RGB float image (H, W, 3)."""
        image = np.random.rand(32, 32, 3).astype(np.float64)
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_grayscale_uint8_input(self, analyzer):
        """Test with 2D grayscale uint8 image (scaled by max value)."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_rgb_uint8_input(self, analyzer):
        """Test with 3D RGB uint8 image."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_uint16_input(self, analyzer):
        """Test with uint16 integer input."""
        image = np.random.randint(0, 65536, (32, 32), dtype=np.uint16)
        result = analyzer.simulate_image(image)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    # Error handling tests
    def test_error_1d_array(self, analyzer):
        """Test that 1D array raises ValueError."""
        image = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image)

    def test_error_4d_array(self, analyzer):
        """Test that 4D array raises ValueError."""
        image = np.random.rand(32, 32, 3, 3)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image)

    def test_error_channel_mismatch(self, analyzer):
        """Test that channel count mismatch raises ValueError."""
        image = np.random.rand(32, 32, 3)
        with pytest.raises(ValueError, match="channels.*wavelengths"):
            analyzer.simulate_image(image, wavelengths_nm=(450.0, 550.0))

    def test_error_unexpected_kwargs(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        image = np.random.rand(32, 32)
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(image, invalid_parameter=123)

    # Parameter variation tests
    def test_pixel_size_parameter(self, analyzer):
        """Test that pixel_size_mm parameter is accepted."""
        image = np.random.rand(32, 32)
        result = analyzer.simulate_image(image, pixel_size_mm=0.01)
        assert result.shape == image.shape
        assert result.dtype == np.float64

    def test_field_angle_parameter(self, analyzer):
        """Test that field_angle_deg parameter is accepted."""
        image = np.random.rand(32, 32)
        result = analyzer.simulate_image(image, field_angle_deg=15.0)
        assert result.shape == image.shape

    def test_focus_shift_parameter(self, analyzer):
        """Test that focus_shift_mm parameter is accepted."""
        image = np.random.rand(32, 32)
        result = analyzer.simulate_image(image, focus_shift_mm=0.1)
        assert result.shape == image.shape

    def test_custom_wavelengths(self, analyzer):
        """Test that custom wavelengths tuple works with matching channels."""
        image = np.random.rand(32, 32, 3)
        custom_wavelengths = (470.0, 550.0, 650.0)
        result = analyzer.simulate_image(image, wavelengths_nm=custom_wavelengths)
        assert result.shape == image.shape
        assert result.dtype == np.float64

    # Legacy kwarg compatibility tests
    def test_legacy_pixel_size_kwarg(self, analyzer):
        """Test backward compatibility: legacy 'pixel_size' kwarg."""
        image = np.random.rand(32, 32)
        result = analyzer.simulate_image(image, pixel_size=0.01)
        assert result.shape == image.shape
        assert result.dtype == np.float64

    def test_legacy_wavelengths_kwarg(self, analyzer):
        """Test backward compatibility: legacy 'wavelengths' kwarg."""
        image = np.random.rand(32, 32, 3)
        result = analyzer.simulate_image(image, wavelengths=(450.0, 550.0, 650.0))
        assert result.shape == image.shape

    # Edge cases and boundary conditions
    def test_black_image(self, analyzer):
        """Test with all-black (zero) image."""
        image = np.zeros((32, 32))
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0)

    def test_white_image(self, analyzer):
        """Test with all-white (ones) image."""
        image = np.ones((32, 32))
        result = analyzer.simulate_image(image)
        assert result.shape == image.shape
        assert result.dtype == np.float64
        assert np.all(result <= 1)

    def test_max_intensity_uint8(self, analyzer):
        """Test that maximum uint8 value (255) is correctly normalized."""
        image = np.full((32, 32), 255, dtype=np.uint8)
        result = analyzer.simulate_image(image)
        assert result.dtype == np.float64
        assert np.all(result <= 1.0)
