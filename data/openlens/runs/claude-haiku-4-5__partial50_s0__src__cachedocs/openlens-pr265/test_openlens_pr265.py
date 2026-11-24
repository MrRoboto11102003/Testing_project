import pytest
import numpy as np
from unittest.mock import Mock, patch


try:
    from openlens.analysis.diffraction_psf import DiffractionPSFCalculator
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestDiffractionPSFCalculatorCalculatePSF:
    """Test suite for DiffractionPSFCalculator.calculate_psf"""

    def _create_wavefront_mock(self, W):
        """Helper to create a mock WavefrontError with given W array"""
        wavefront = Mock()
        wavefront.W = W
        return wavefront

    def test_basic_calculation_returns_numpy_array(self):
        """Test that calculate_psf returns a numpy array"""
        wavefront = self._create_wavefront_mock(np.zeros((64, 64)))
        result = DiffractionPSFCalculator.calculate_psf(wavefront)
        assert isinstance(result, np.ndarray)

    def test_output_shape_with_default_pad_factor(self):
        """Test output shape with default pad_factor=4"""
        wavefront = self._create_wavefront_mock(np.zeros((64, 64)))
        result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=4)
        assert result.shape == (256, 256)

    def test_output_shape_scales_with_pad_factor(self):
        """Test that output size = input_size * pad_factor"""
        for size in [8, 16, 32, 64]:
            for pad_factor in [1, 2, 4]:
                wavefront = self._create_wavefront_mock(np.zeros((size, size)))
                result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=pad_factor)
                expected = size * pad_factor
                assert result.shape == (expected, expected)

    def test_psf_normalization_max_equals_one(self):
        """Test that PSF is normalized with max value equal to 1"""
        wavefront = self._create_wavefront_mock(np.zeros((64, 64)))
        result = DiffractionPSFCalculator.calculate_psf(wavefront)
        assert np.max(result) == 1.0

    def test_psf_values_are_non_negative(self):
        """Test that PSF intensity values are non-negative"""
        wavefront = self._create_wavefront_mock(np.random.randn(64, 64) * 0.5)
        result = DiffractionPSFCalculator.calculate_psf(wavefront)
        assert np.all(result >= 0)

    def test_nan_regions_outside_aperture_handled(self):
        """Test that NaN regions (outside aperture) are handled gracefully"""
        W = np.zeros((64, 64))
        W[0:8, :] = np.nan
        W[-8:, :] = np.nan
        wavefront = self._create_wavefront_mock(W)
        
        result = DiffractionPSFCalculator.calculate_psf(wavefront)
        
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert 0 <= np.max(result) <= 1.0

    def test_completely_masked_wavefront(self):
        """Test PSF calculation with completely masked (all-NaN) wavefront"""
        wavefront = self._create_wavefront_mock(np.full((64, 64), np.nan))
        result = DiffractionPSFCalculator.calculate_psf(wavefront)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256)
        assert not np.any(np.isnan(result))

    def test_with_uniform_intensity_apodization(self):
        """Test PSF with uniform intensity map apodization"""
        W = np.zeros((64, 64))
        intensity = np.ones((64, 64))
        wavefront = self._create_wavefront_mock(W)
        
        result = DiffractionPSFCalculator.calculate_psf(wavefront, intensity_map=intensity)
        
        assert result.shape == (256, 256)
        assert np.max(result) == 1.0

    def test_with_gaussian_intensity_apodization(self):
        """Test PSF with Gaussian intensity map apodization"""
        W = np.zeros((64, 64))
        x = np.linspace(-3, 3, 64)
        X, Y = np.meshgrid(x, x)
        intensity = np.exp(-(X**2 + Y**2))
        wavefront = self._create_wavefront_mock(W)
        
        result = DiffractionPSFCalculator.calculate_psf(wavefront, intensity_map=intensity)
        
        assert result.shape == (256, 256)
        assert 0 < np.max(result) <= 1.0

    def test_phase_pattern_affects_psf_result(self):
        """Test that different phase patterns produce different PSF results"""
        # Zero phase baseline
        W_zero = np.zeros((64, 64))
        wavefront_zero = self._create_wavefront_mock(W_zero)
        result_zero = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        
        # Linear phase perturbation
        x = np.linspace(0, 2*np.pi, 64)
        W_linear = x.reshape(64, 1) * np.ones((1, 64)) * 0.2
        wavefront_linear = self._create_wavefront_mock(W_linear)
        result_linear = DiffractionPSFCalculator.calculate_psf(wavefront_linear)
        
        # Results should differ
        assert not np.allclose(result_zero, result_linear)

    def test_pad_factor_parameter_effect(self):
        """Test that pad_factor correctly affects output resolution"""
        W = np.sin(np.linspace(0, 4*np.pi, 32).reshape(32, 1) * np.ones((1, 32)))
        
        result_pad1 = DiffractionPSFCalculator.calculate_psf(
            self._create_wavefront_mock(W), pad_factor=1
        )
        result_pad4 = DiffractionPSFCalculator.calculate_psf(
            self._create_wavefront_mock(W), pad_factor=4
        )
        
        assert result_pad1.shape == (32, 32)
        assert result_pad4.shape == (128, 128)

    def test_numpy_unavailable_returns_empty_list(self):
        """Test that function returns empty list when numpy unavailable"""
        wavefront = self._create_wavefront_mock(np.zeros((64, 64)))
        
        with patch('openlens.analysis.diffraction_psf.NUMPY_AVAILABLE', False):
            result = DiffractionPSFCalculator.calculate_psf(wavefront)
            assert result == []

    def test_small_array_processing(self):
        """Test processing of small wavefront arrays"""
        wavefront = self._create_wavefront_mock(np.zeros((4, 4)))
        result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=2)
        assert result.shape == (8, 8)
        assert np.max(result) == 1.0

    def test_large_array_processing(self):
        """Test processing of larger wavefront arrays"""
        wavefront = self._create_wavefront_mock(np.zeros((128, 128)))
        result = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=2)
        assert result.shape == (256, 256)
        assert np.max(result) == 1.0
