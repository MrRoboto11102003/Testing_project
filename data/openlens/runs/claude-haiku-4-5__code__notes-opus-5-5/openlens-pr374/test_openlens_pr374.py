import pytest
import numpy as np
from unittest.mock import patch
import re

from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.system import OpticalSystem


class TestSimulateImage:
    """Test suite for ImageQualityAnalyzer.simulate_image"""

    @pytest.fixture
    def iqa(self):
        """Create an ImageQualityAnalyzer with an empty optical system."""
        return ImageQualityAnalyzer(OpticalSystem())

    def _make_identity_psf(self, size=3):
        """Helper: create an identity PSF kernel."""
        kernel = np.zeros((size, size), dtype=np.float64)
        kernel[size // 2, size // 2] = 1.0
        return kernel

    def _make_zero_psf(self, size=3):
        """Helper: create a zero PSF kernel."""
        return np.zeros((size, size), dtype=np.float64)

    def test_grayscale_2d_identity_kernel(self, iqa):
        """Test 2D grayscale image with identity kernel."""
        image = np.array([[0.1, 0.5], [0.8, 0.9]], dtype=np.float64)
        
        def mock_psf(**kwargs):
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        assert result.dtype == np.float64
        assert result.shape == image.shape
        assert np.allclose(result, image, atol=1e-9)
        assert result is not image

    def test_rgb_3d_identity_kernel(self, iqa):
        """Test 3D RGB image with identity kernels."""
        image = np.array([
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
            [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        ], dtype=np.float64)
        
        def mock_psf(**kwargs):
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        assert result.dtype == np.float64
        assert result.shape == image.shape
        assert np.allclose(result, image, atol=1e-9)

    def test_integer_input_uint8(self, iqa):
        """Test uint8 input is scaled by 255."""
        image = np.array([[0, 127], [255, 64]], dtype=np.uint8)
        
        def mock_psf(**kwargs):
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        expected = image.astype(np.float64) / 255.0
        assert result.dtype == np.float64
        assert np.allclose(result, expected, atol=1e-9)

    def test_clipping_extreme_values(self, iqa):
        """Test clipping of values outside [0, 1]."""
        image = np.array([[-0.5, 0.5], [1.5, 2.0]], dtype=np.float64)
        
        def mock_psf(**kwargs):
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert result.dtype == np.float64

    def test_error_1d_array(self, iqa):
        """Test ValueError for 1D array."""
        image = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        
        with pytest.raises(ValueError, match=re.escape(
            "image_array must be 2-D (H, W) or 3-D (H, W, C); got shape (3,)"
        )):
            iqa.simulate_image(image)

    def test_error_channel_mismatch_too_few(self, iqa):
        """Test ValueError when channels < wavelengths."""
        image = np.zeros((4, 4, 2), dtype=np.float64)
        
        with pytest.raises(ValueError, match=re.escape(
            "image_array has 2 channels but 3 wavelengths were given"
        )):
            iqa.simulate_image(image)

    def test_error_channel_mismatch_too_many(self, iqa):
        """Test ValueError when channels > wavelengths."""
        image = np.zeros((4, 4, 3), dtype=np.float64)
        
        with pytest.raises(ValueError, match=re.escape(
            "image_array has 3 channels but 1 wavelengths were given"
        )):
            iqa.simulate_image(image, wavelengths_nm=(550.0,))

    def test_error_unexpected_kwargs(self, iqa):
        """Test TypeError for unexpected kwargs."""
        image = np.zeros((4, 4), dtype=np.float64)
        
        with pytest.raises(TypeError, match=re.escape(
            "Unexpected kwargs: ['foo']"
        )):
            iqa.simulate_image(image, foo=1)

    def test_error_unexpected_kwargs_after_alias(self, iqa):
        """Test TypeError with valid alias and invalid kwarg."""
        image = np.zeros((4, 4), dtype=np.float64)
        
        with pytest.raises(TypeError, match=re.escape(
            "Unexpected kwargs: ['foo']"
        )):
            iqa.simulate_image(image, pixel_size=0.01, foo=1)

    def test_legacy_alias_pixel_size(self, iqa):
        """Test pixel_size alias overrides pixel_size_mm."""
        image = np.ones((8, 8), dtype=np.float64)
        calls = []
        
        def mock_psf(**kwargs):
            calls.append(kwargs)
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            iqa.simulate_image(image, pixel_size=0.01)
        
        assert len(calls) == 3
        for call in calls:
            assert call["sensor_size_mm"] == pytest.approx(
                call["pixels"] * 0.01, rel=1e-6
            )
            assert call["pixels"] % 2 == 1

    def test_legacy_alias_wavelengths(self, iqa):
        """Test wavelengths alias overrides wavelengths_nm."""
        image = np.zeros((4, 4, 2), dtype=np.float64)
        calls = []
        
        def mock_psf(**kwargs):
            calls.append(kwargs)
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            iqa.simulate_image(image, wavelengths=(500.0, 600.0))
        
        assert len(calls) == 2
        wls = [call["wavelength_nm"] for call in calls]
        assert wls == [500.0, 600.0]

    def test_psf_calls_use_diffraction_true(self, iqa):
        """Test PSF is called with use_diffraction=True."""
        image = np.ones((8, 8), dtype=np.float64)
        calls = []
        
        def mock_psf(**kwargs):
            calls.append(kwargs)
            return {"image": self._make_identity_psf(3)}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            iqa.simulate_image(image)
        
        assert len(calls) == 3
        for call in calls:
            assert call["use_diffraction"] is True

    def test_grayscale_mean_of_wavelengths(self, iqa):
        """Test grayscale output is mean of per-wavelength convolutions."""
        image = np.ones((4, 4), dtype=np.float64)
        kernels = [
            self._make_identity_psf(3),
            self._make_zero_psf(3),
            self._make_zero_psf(3),
        ]
        kernel_index = [0]
        
        def mock_psf(**kwargs):
            k = kernels[kernel_index[0]].copy()
            kernel_index[0] += 1
            return {"image": k}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        expected = np.ones((4, 4)) / 3.0
        assert np.allclose(result, expected, atol=1e-9)

    def test_rgb_channels_separate(self, iqa):
        """Test 3D channels are convolved separately."""
        image = np.array([
            [[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ], dtype=np.float64)
        
        kernels = [
            self._make_identity_psf(3),
            self._make_zero_psf(3),
            self._make_zero_psf(3),
        ]
        kernel_index = [0]
        
        def mock_psf(**kwargs):
            k = kernels[kernel_index[0]].copy()
            kernel_index[0] += 1
            return {"image": k}
        
        with patch.object(iqa, "calculate_psf", side_effect=mock_psf):
            result = iqa.simulate_image(image)
        
        expected = np.zeros_like(image)
        expected[:, :, 0] = image[:, :, 0]
        assert np.allclose(result, expected, atol=1e-9)

    def test_empty_system_produces_zeros(self, iqa):
        """Test empty OpticalSystem produces all-zero output."""
        image = np.ones((4, 4), dtype=np.float64)
        result = iqa.simulate_image(image)
        
        assert result.dtype == np.float64
        assert result.shape == image.shape
        assert np.allclose(result, 0.0, atol=1e-9)
