import pytest
import numpy as np
from unittest.mock import Mock, patch
from openlens.analysis.diffraction_psf import DiffractionPSFCalculator


class TestDiffractionPSFCalculatorCalculatePSF:
    """Test suite for DiffractionPSFCalculator.calculate_psf static method."""

    def test_basic_psf_calculation_returns_2d_array(self):
        """Test that basic PSF calculation returns a 2D normalized array."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (256, 256)  # 64 * 4 (default pad_factor)

    def test_psf_normalized_maximum(self):
        """Test that PSF maximum intensity is normalized to 1.0."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        # Perfect wavefront should have peak normalized to 1.0
        assert np.max(result) <= 1.0 + 1e-10
        assert np.max(result) > 0.9

    def test_psf_all_values_non_negative(self):
        """Test that all PSF values are non-negative (physical intensity)."""
        wavefront = Mock()
        wavefront.W = np.random.randn(64, 64) * 0.05

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        assert np.all(result >= 0)

    def test_pad_factor_determines_output_size(self):
        """Test that pad_factor correctly sets the output array size."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        for pad_factor in [1, 2, 4, 8]:
            result = DiffractionPSFCalculator.calculate_psf(
                wavefront, pad_factor=pad_factor
            )
            expected_size = 64 * pad_factor
            assert result.shape == (expected_size, expected_size)

    def test_psf_with_intensity_map_apodization(self):
        """Test PSF calculation with pupil apodization (intensity_map)."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))
        intensity_map = np.ones((64, 64)) * 0.5

        result = DiffractionPSFCalculator.calculate_psf(
            wavefront, intensity_map=intensity_map
        )

        assert isinstance(result, np.ndarray)
        assert np.max(result) <= 1.0
        assert np.all(result >= 0)

    def test_psf_with_gaussian_vignetting_map(self):
        """Test PSF with realistic Gaussian vignetting intensity map."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        intensity_map = np.exp(-2 * R**2)

        result = DiffractionPSFCalculator.calculate_psf(
            wavefront, intensity_map=intensity_map
        )

        assert np.max(result) <= 1.0
        assert np.all(result >= 0)

    def test_psf_handles_nan_outside_aperture(self):
        """Test that NaN values (outside aperture) are properly handled."""
        wavefront = Mock()
        W = np.zeros((64, 64))
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        W[R > 1.0] = np.nan
        wavefront.W = W

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        # NaNs should be replaced, not propagated
        assert not np.any(np.isnan(result))
        assert np.max(result) <= 1.0

    def test_psf_with_phase_aberration(self):
        """Test PSF calculation with non-zero phase (aberrated wavefront)."""
        wavefront = Mock()
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        wavefront.W = 0.1 * (X**2 + Y**2)  # Defocus-like aberration

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        assert isinstance(result, np.ndarray)
        assert np.max(result) <= 1.0
        assert np.all(result >= 0)

    def test_psf_different_wavefront_input_sizes(self):
        """Test PSF calculation with various input array dimensions."""
        for size in [32, 64, 128]:
            wavefront = Mock()
            wavefront.W = np.zeros((size, size))

            result = DiffractionPSFCalculator.calculate_psf(
                wavefront, pad_factor=2
            )

            assert result.shape == (size * 2, size * 2)

    def test_psf_with_intensity_map_larger_than_one(self):
        """Test that intensity_map values > 1 are properly handled."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))
        intensity_map = np.ones((64, 64)) * 2.5

        result = DiffractionPSFCalculator.calculate_psf(
            wavefront, intensity_map=intensity_map
        )

        # PSF should normalize correctly regardless of intensity_map magnitude
        assert np.max(result) <= 1.0 + 1e-10

    def test_psf_intensity_concentration_at_center(self):
        """Test that PSF concentrates energy in central region for perfect wavefront."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=4)

        center = result.shape[0] // 2
        center_region = result[center-20:center+20, center-20:center+20]
        edge_region = result[0:20, 0:20]

        assert np.mean(center_region) > np.mean(edge_region) * 10

    def test_psf_with_zero_intensity_map(self):
        """Test PSF with completely blocked pupil (zero intensity everywhere)."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))
        intensity_map = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(
            wavefront, intensity_map=intensity_map
        )

        # With zero amplitude, PSF should be all zeros or negligible
        assert isinstance(result, np.ndarray)
        assert np.max(result) < 1e-10 or np.allclose(result, 0)

    def test_default_pad_factor_is_four(self):
        """Test that default pad_factor parameter is 4."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        # With default pad_factor=4, output should be 64*4 = 256
        assert result.shape == (256, 256)

    @patch('openlens.analysis.diffraction_psf.NUMPY_AVAILABLE', False)
    def test_numpy_unavailable_returns_empty_list(self):
        """Test that empty list is returned when numpy is not available."""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))

        result = DiffractionPSFCalculator.calculate_psf(wavefront)

        assert result == []
        assert isinstance(result, list)

    def test_psf_circular_aperture_symmetry(self):
        """Test PSF with circular aperture has approximate symmetry."""
        wavefront = Mock()
        W = np.zeros((64, 64))
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        W[R > 1.0] = np.nan
        wavefront.W = W

        result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=2)

        # Center should have highest intensity
        center = result.shape[0] // 2
        center_val = result[center, center]
        
        # Radially outward, intensity should decrease
        assert result[center, center] > result[center+10, center]
