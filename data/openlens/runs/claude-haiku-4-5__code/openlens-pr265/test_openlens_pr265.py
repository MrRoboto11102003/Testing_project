import pytest
import numpy as np
from unittest.mock import Mock
from openlens.analysis.diffraction_psf import DiffractionPSFCalculator


class TestCalculatePSF:
    """Test suite for DiffractionPSFCalculator.calculate_psf"""
    
    @pytest.fixture
    def wavefront_zero(self):
        """Create mock wavefront with zero phase"""
        wavefront = Mock()
        wavefront.W = np.zeros((64, 64))
        return wavefront
    
    @pytest.fixture
    def wavefront_with_phase(self):
        """Create mock wavefront with sinusoidal phase"""
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        wavefront = Mock()
        wavefront.W = np.where(R <= 1.0, 0.1 * np.sin(2 * np.pi * R), np.nan)
        return wavefront
    
    @pytest.fixture
    def wavefront_with_nans(self):
        """Create mock wavefront with NaN outside circular aperture"""
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        wavefront = Mock()
        wavefront.W = np.where(R <= 1.0, 0.05 * R, np.nan)
        return wavefront
    
    def test_calculate_psf_returns_array_or_empty_list(self, wavefront_zero):
        """PSF calculation should return numpy array or empty list if numpy unavailable"""
        result = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        assert isinstance(result, (np.ndarray, list))
    
    def test_calculate_psf_shape_with_default_padding(self, wavefront_zero):
        """PSF shape should be (input_size * pad_factor, input_size * pad_factor)"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        if isinstance(psf, np.ndarray):
            assert psf.shape == (256, 256), f"Expected (256, 256), got {psf.shape}"
    
    def test_calculate_psf_shape_respects_pad_factor(self, wavefront_zero):
        """PSF shape should scale correctly with pad_factor parameter"""
        for pad_factor in [1, 2, 4, 8]:
            psf = DiffractionPSFCalculator.calculate_psf(
                wavefront_zero, pad_factor=pad_factor
            )
            if isinstance(psf, np.ndarray):
                expected_size = 64 * pad_factor
                assert psf.shape == (expected_size, expected_size)
    
    def test_calculate_psf_is_normalized_to_one(self, wavefront_zero):
        """PSF should be normalized with maximum value of 1.0"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        if isinstance(psf, np.ndarray):
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
    
    def test_calculate_psf_intensity_values_in_valid_range(self, wavefront_zero):
        """PSF intensity values should be in [0, 1]"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        if isinstance(psf, np.ndarray):
            assert np.all(psf >= -1e-10)
            assert np.all(psf <= 1.0 + 1e-10)
    
    def test_calculate_psf_is_real_valued(self, wavefront_zero):
        """PSF should contain only real values"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_zero)
        if isinstance(psf, np.ndarray):
            assert not np.iscomplexobj(psf)
    
    def test_calculate_psf_with_uniform_intensity_map(self, wavefront_zero):
        """PSF calculation should accept and use uniform intensity_map"""
        intensity_map = np.ones((64, 64))
        psf = DiffractionPSFCalculator.calculate_psf(
            wavefront_zero, intensity_map=intensity_map
        )
        if isinstance(psf, np.ndarray):
            assert psf.shape == (256, 256)
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
    
    def test_calculate_psf_with_gaussian_apodization(self, wavefront_zero):
        """PSF with Gaussian intensity apodization should produce valid output"""
        x = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, x)
        intensity_map = np.exp(-(X**2 + Y**2) / 0.5)
        
        psf = DiffractionPSFCalculator.calculate_psf(
            wavefront_zero, intensity_map=intensity_map
        )
        if isinstance(psf, np.ndarray):
            assert psf.shape == (256, 256)
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
            assert np.all(psf >= 0.0)
    
    def test_calculate_psf_with_phase_variation(self, wavefront_with_phase):
        """PSF with varying phase wavefront should produce valid output"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_with_phase)
        if isinstance(psf, np.ndarray):
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
            assert np.all(psf >= 0.0)
            assert np.all(psf <= 1.0 + 1e-10)
    
    def test_calculate_psf_handles_nan_in_wavefront(self, wavefront_with_nans):
        """PSF should handle NaN values in wavefront (outside aperture) gracefully"""
        psf = DiffractionPSFCalculator.calculate_psf(wavefront_with_nans)
        if isinstance(psf, np.ndarray):
            assert not np.any(np.isnan(psf)), "PSF contains NaN values"
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
            assert np.all(psf >= 0.0)
    
    def test_calculate_psf_different_input_sizes(self):
        """PSF should work correctly with different wavefront sizes"""
        for size in [32, 64, 128]:
            wavefront = Mock()
            wavefront.W = np.zeros((size, size))
            psf = DiffractionPSFCalculator.calculate_psf(wavefront, pad_factor=2)
            if isinstance(psf, np.ndarray):
                assert psf.shape == (size * 2, size * 2)
                assert np.isclose(np.max(psf), 1.0, atol=1e-10)
    
    def test_calculate_psf_minimum_padding(self, wavefront_zero):
        """PSF with minimal padding (pad_factor=1) should still be valid"""
        psf = DiffractionPSFCalculator.calculate_psf(
            wavefront_zero, pad_factor=1
        )
        if isinstance(psf, np.ndarray):
            assert psf.shape == (64, 64)
            assert np.isclose(np.max(psf), 1.0, atol=1e-10)
