import numpy as np
import pytest
from openlens.analysis.diffraction_psf import DiffractionPSFCalculator
from openlens.optics.wavefront import WavefrontError


class TestDiffractionPSFCalculatorCalculatePSF:
    """Test suite for DiffractionPSFCalculator.calculate_psf"""
    
    def test_basic_flat_phase_uniform_pupil(self):
        """Test basic case: flat phase (W=0) with uniform pupil"""
        N = 8
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert psf.size > 0
        assert psf.shape == (N * 4, N * 4)  # default pad_factor=4
        assert np.isclose(psf.max(), 1.0)
        
    def test_output_shape_with_different_pad_factors(self):
        """Test that output shape is (N*pad_factor, N*pad_factor)"""
        N = 4
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        for pad_factor in [1, 2, 4]:
            psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=pad_factor)
            assert psf.shape == (N * pad_factor, N * pad_factor)
    
    def test_output_normalization(self):
        """Test that PSF is normalized to max=1.0 when non-zero"""
        N = 8
        W = np.ones((N, N)) * 0.5  # Non-zero constant phase
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert np.isclose(psf.max(), 1.0)
    
    def test_all_nan_pupil_returns_zeros(self):
        """Test that all-NaN input returns all zeros without error"""
        N = 4
        W = np.full((N, N), np.nan)
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert np.allclose(psf, 0.0)
        assert psf.max() == 0.0
    
    def test_with_intensity_map(self):
        """Test PSF calculation with intensity apodization"""
        N = 8
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        intensity_map = np.ones((N, N))
        psf = DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity_map)
        
        assert psf.shape == (N * 4, N * 4)
        assert np.isclose(psf.max(), 1.0)
    
    def test_piston_invariance(self):
        """Test that constant phase (piston) doesn't change PSF"""
        N = 8
        W1 = np.zeros((N, N))
        W2 = np.ones((N, N)) * 0.3
        W3 = np.ones((N, N)) * 1.0  # Integer multiple of wave
        
        wf1 = WavefrontError(None, None, W1)
        wf2 = WavefrontError(None, None, W2)
        wf3 = WavefrontError(None, None, W3)
        
        psf1 = DiffractionPSFCalculator.calculate_psf(wf1)
        psf2 = DiffractionPSFCalculator.calculate_psf(wf2)
        psf3 = DiffractionPSFCalculator.calculate_psf(wf3)
        
        assert np.allclose(psf1, psf2, atol=1e-9)
        assert np.allclose(psf1, psf3, atol=1e-9)
    
    def test_nan_handling(self):
        """Test that NaNs (outside aperture) are handled correctly"""
        N = 8
        W = np.zeros((N, N))
        W[0:2, 0:2] = np.nan
        W[-2:, -2:] = np.nan
        
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert not np.isnan(psf).any()
        assert np.isclose(psf.max(), 1.0)
    
    def test_intensity_map_not_mutated(self):
        """Test that intensity_map is not modified by calculate_psf"""
        N = 8
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        intensity_map = np.ones((N, N)) * 2.0
        intensity_map_copy = intensity_map.copy()
        
        DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity_map)
        
        assert np.array_equal(intensity_map, intensity_map_copy)
    
    def test_wavefront_not_mutated(self):
        """Test that wavefront W is not modified by calculate_psf"""
        N = 8
        W = np.zeros((N, N))
        W[0, 0] = np.nan
        wf = WavefrontError(None, None, W)
        
        W_copy = W.copy()
        DiffractionPSFCalculator.calculate_psf(wf)
        
        assert np.array_equal(W, W_copy, equal_nan=True)
    
    def test_parseval_flat_phase(self):
        """Test Parseval property: sum ≈ M²/K for flat phase"""
        N = 4
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        pad_factor = 4
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=pad_factor)
        
        M = N * pad_factor
        K = N * N  # All samples valid (no NaN)
        expected_sum = (M * M) / K
        
        assert np.isclose(psf.sum(), expected_sum, rtol=1e-6)
    
    def test_central_peak_for_flat_phase(self):
        """Test that peak is at center for flat-phase uniform pupil"""
        N = 8
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        pad_factor = 4
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=pad_factor)
        
        M = N * pad_factor
        center = M // 2
        
        peak_idx = np.unravel_index(psf.argmax(), psf.shape)
        assert peak_idx[0] == center
        assert peak_idx[1] == center
    
    def test_all_values_nonnegative(self):
        """Test that all PSF values are >= 0"""
        N = 8
        np.random.seed(42)
        W = np.random.randn(N, N) * 0.1  # Small random phase
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert np.all(psf >= 0.0)
    
    def test_output_dtype(self):
        """Test that output dtype is float64"""
        N = 4
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf)
        
        assert psf.dtype == np.float64
    
    def test_pad_factor_one_delta(self):
        """Test that pad_factor=1 with uniform pupil concentrates PSF at center"""
        N = 4
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=1)
        
        center = N // 2
        assert psf[center, center] == 1.0
        assert np.isclose(psf.sum(), 1.0, atol=1e-9)
    
    def test_intensity_map_scaling_invariance(self):
        """Test that scaling intensity_map doesn't change normalized PSF"""
        N = 8
        W = np.zeros((N, N))
        wf = WavefrontError(None, None, W)
        
        intensity_map = np.ones((N, N))
        psf1 = DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity_map)
        
        intensity_map_scaled = np.ones((N, N)) * 4.0
        psf2 = DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity_map_scaled)
        
        assert np.allclose(psf1, psf2, atol=1e-9)
