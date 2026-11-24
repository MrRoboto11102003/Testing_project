import pytest
import numpy as np
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.lens import Lens


class TestSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image method."""

    @pytest.fixture
    def analyzer(self):
        """Create a test ImageQualityAnalyzer with a simple lens."""
        lens = Lens(
            name="Test lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        return ImageQualityAnalyzer(lens)

    # --- Input Format Tests ---

    def test_2d_grayscale_float_accepted(self, analyzer):
        """Test that 2D float grayscale image is accepted and returns 2D output."""
        image = np.random.rand(64, 64).astype(np.float64)
        result = analyzer.simulate_image(image)
        assert result.shape == (64, 64)
        assert result.ndim == 2
        assert result.dtype == np.float64

    def test_3d_rgb_float_accepted(self, analyzer):
        """Test that 3D float RGB image is accepted and returns 3D output."""
        image = np.random.rand(64, 64, 3).astype(np.float64)
        result = analyzer.simulate_image(image)
        assert result.shape == (64, 64, 3)
        assert result.ndim == 3
        assert result.dtype == np.float64

    def test_uint8_grayscale_scaled(self, analyzer):
        """Test that uint8 grayscale image is scaled correctly to 0-1."""
        image = np.full((32, 32), 128, dtype=np.uint8)
        result = analyzer.simulate_image(image)
        assert result.shape == (32, 32)
        assert result.dtype == np.float64
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_uint8_rgb_scaled(self, analyzer):
        """Test that uint8 RGB image is scaled correctly."""
        image = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = analyzer.simulate_image(image)
        assert result.shape == (32, 32, 3)
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_int16_image_scaled(self, analyzer):
        """Test that other integer types like int16 are scaled."""
        image = np.full((32, 32), 100, dtype=np.int16)
        result = analyzer.simulate_image(image)
        assert result.dtype == np.float64
        assert 0.0 <= result.min() <= result.max() <= 1.0

    # --- Dimension Validation ---

    def test_1d_array_raises_valueerror(self, analyzer):
        """Test that 1D array raises ValueError with appropriate message."""
        image = np.random.rand(64)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image)

    def test_4d_array_raises_valueerror(self, analyzer):
        """Test that 4D array raises ValueError."""
        image = np.random.rand(2, 64, 64, 3)
        with pytest.raises(ValueError, match="2-D.*3-D"):
            analyzer.simulate_image(image)

    def test_mismatched_channels_raises_valueerror(self, analyzer):
        """Test that channel count mismatch with wavelengths raises ValueError."""
        image = np.random.rand(32, 32, 4)
        wavelengths = (450.0, 550.0, 650.0)  # 3 wavelengths, 4 channels
        with pytest.raises(ValueError, match="channels.*wavelengths"):
            analyzer.simulate_image(image, wavelengths_nm=wavelengths)

    # --- Legacy Kwargs ---

    def test_legacy_pixel_size_kwarg(self, analyzer):
        """Test that legacy 'pixel_size' kwarg is supported."""
        image = np.random.rand(32, 32).astype(np.float64)
        result = analyzer.simulate_image(image, pixel_size=0.005)
        assert result.shape == (32, 32)

    def test_legacy_wavelengths_kwarg(self, analyzer):
        """Test that legacy 'wavelengths' kwarg is supported."""
        image = np.random.rand(32, 32, 3).astype(np.float64)
        result = analyzer.simulate_image(image, wavelengths=(450.0, 550.0, 650.0))
        assert result.shape == (32, 32, 3)

    def test_unexpected_kwargs_raise_typeerror(self, analyzer):
        """Test that unexpected kwargs raise TypeError."""
        image = np.random.rand(32, 32).astype(np.float64)
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(image, invalid_param=123)

    # --- Output Properties ---

    def test_output_always_float64(self, analyzer):
        """Test that output is always float64 regardless of input dtype."""
        uint8_img = np.full((32, 32), 128, dtype=np.uint8)
        float32_img = np.ones((32, 32), dtype=np.float32)
        int32_img = np.zeros((32, 32), dtype=np.int32)

        assert analyzer.simulate_image(uint8_img).dtype == np.float64
        assert analyzer.simulate_image(float32_img).dtype == np.float64
        assert analyzer.simulate_image(int32_img).dtype == np.float64

    def test_output_normalized_0_to_1(self, analyzer):
        """Test that output is clipped to [0, 1] range."""
        image = np.random.rand(32, 32).astype(np.float64)
        result = analyzer.simulate_image(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_shape_preserved(self, analyzer):
        """Test that output shape exactly matches input shape."""
        test_shapes = [(16, 16), (32, 48), (64, 32, 3), (8, 8, 3)]
        for shape in test_shapes:
            image = np.random.rand(*shape).astype(np.float64)
            result = analyzer.simulate_image(image)
            assert result.shape == shape

    # --- Parameter Variations ---

    def test_field_angle_and_focus_shift_accepted(self, analyzer):
        """Test that field_angle_deg and focus_shift_mm parameters work."""
        image = np.random.rand(32, 32).astype(np.float64)
        result1 = analyzer.simulate_image(image, field_angle_deg=5.0)
        result2 = analyzer.simulate_image(image, focus_shift_mm=0.5)
        result3 = analyzer.simulate_image(
            image, field_angle_deg=5.0, focus_shift_mm=-0.5
        )
        assert all(r.shape == (32, 32) for r in [result1, result2, result3])
        assert all(0.0 <= r.min() <= r.max() <= 1.0 for r in [result1, result2, result3])

    def test_various_pixel_sizes(self, analyzer):
        """Test that various pixel_size_mm values are accepted."""
        image = np.random.rand(32, 32).astype(np.float64)
        for pixel_size in [0.001, 0.005, 0.01, 0.05]:
            result = analyzer.simulate_image(image, pixel_size_mm=pixel_size)
            assert result.shape == (32, 32)
