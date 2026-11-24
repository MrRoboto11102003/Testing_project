import pytest
import numpy as np
from openlens.analysis.diffraction_psf import DiffractionPSFCalculator
from openlens.aberrations import WavefrontError


class TestDiffractionPSFCalculatorCalculatePSF:
    """Test suite for DiffractionPSFCalculator.calculate_psf"""

    def test_basic_uniform_pupil_zero_phase(self):
        """Test basic PSF calculation with uniform pupil and zero phase."""
        W = np.zeros((8, 8), dtype=float)
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)

        # Check shape: N * pad_factor = 8 * 4 = 32
        assert psf.shape == (32, 32)
        # Check normalization: peak = 1.0
        assert np.isclose(psf.max(), 1.0, atol=1e-10)
        # Check non-negative
        assert np.all(psf >= 0)

    def test_output_shape_various_pad_factors(self):
        """Test that output shape is correct for different pad factors."""
        W = np.zeros((4, 4), dtype=float)
        wf = WavefrontError(None, None, W)

        for pad in [1, 2, 4, 8]:
            psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=pad)
            assert psf.shape == (4 * pad, 4 * pad)

    def test_normalization_to_unity_peak(self):
        """Test that PSF peak is normalized to 1.0 for non-zero pupil."""
        W = np.random.rand(8, 8) * 0.1
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)

        assert np.isclose(psf.max(), 1.0, atol=1e-9)

    def test_all_nan_wavefront_gives_zeros(self):
        """Test that all-NaN wavefront returns zeros (not NaN)."""
        W = np.full((8, 8), np.nan, dtype=float)
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)

        assert psf.shape == (32, 32)
        assert np.allclose(psf, 0.0)
        assert not np.any(np.isnan(psf))

    def test_zero_phase_produces_centered_peak(self):
        """Test that zero phase produces peak at center."""
        W = np.zeros((8, 8), dtype=float)
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=4)

        center = 16
        peak_row, peak_col = np.unravel_index(np.argmax(psf), psf.shape)
        assert peak_row == center
        assert peak_col == center

    def test_zero_phase_symmetry(self):
        """Test symmetry of flat-phase PSF."""
        W = np.zeros((8, 8), dtype=float)
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=4)

        center = 16
        # Check 4-fold symmetry around center
        assert np.allclose(psf[center + 1, center], psf[center - 1, center])
        assert np.allclose(psf[center, center + 1], psf[center, center - 1])
        assert np.allclose(psf[center + 2, center], psf[center - 2, center])
        assert np.allclose(psf[center, center + 2], psf[center, center - 2])

    def test_uniform_intensity_map_same_as_none(self):
        """Test that None intensity_map equals uniform intensity_map."""
        W = np.random.rand(8, 8) * 0.1
        wf1 = WavefrontError(None, None, W.copy())
        wf2 = WavefrontError(None, None, W.copy())

        psf_none = DiffractionPSFCalculator.calculate_psf(wf1, intensity_map=None)
        psf_uniform = DiffractionPSFCalculator.calculate_psf(
            wf2, intensity_map=np.ones((8, 8))
        )

        assert np.allclose(psf_none, psf_uniform, atol=1e-10)

    def test_intensity_scaling_invariant_normalized(self):
        """Test that scaling intensity_map doesn't change normalized PSF."""
        W = np.random.rand(8, 8) * 0.1
        intensity1 = np.ones((8, 8))
        intensity4 = np.ones((8, 8)) * 4.0

        wf1 = WavefrontError(None, None, W.copy())
        wf2 = WavefrontError(None, None, W.copy())

        psf1 = DiffractionPSFCalculator.calculate_psf(wf1, intensity_map=intensity1)
        psf4 = DiffractionPSFCalculator.calculate_psf(wf2, intensity_map=intensity4)

        assert np.allclose(psf1, psf4, atol=1e-10)

    def test_nan_handling_zero_amplitude(self):
        """Test that NaN in W produces zero amplitude at that location."""
        W = np.zeros((8, 8), dtype=float)
        # Create circular aperture
        y = np.linspace(-1, 1, 8)
        Y, Z = np.meshgrid(y, y)
        W[Y**2 + Z**2 > 1.0] = np.nan

        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)

        # Result should have no NaN
        assert not np.any(np.isnan(psf))
        # Result should be normalized
        assert np.isclose(psf.max(), 1.0, atol=1e-10)

    def test_piston_phase_invariance(self):
        """Test that adding constant phase (piston) is invariant."""
        W0 = np.zeros((8, 8), dtype=float)
        W1 = np.ones((8, 8)) * 0.5
        W2 = np.ones((8, 8)) * 1.0
        W3 = np.ones((8, 8)) * 2.0

        psf0 = DiffractionPSFCalculator.calculate_psf(WavefrontError(None, None, W0))
        psf1 = DiffractionPSFCalculator.calculate_psf(WavefrontError(None, None, W1))
        psf2 = DiffractionPSFCalculator.calculate_psf(WavefrontError(None, None, W2))
        psf3 = DiffractionPSFCalculator.calculate_psf(WavefrontError(None, None, W3))

        assert np.allclose(psf0, psf1, atol=1e-10)
        assert np.allclose(psf1, psf2, atol=1e-10)
        assert np.allclose(psf2, psf3, atol=1e-10)

    def test_dtype_float64(self):
        """Test that output dtype is float64."""
        W = np.zeros((8, 8), dtype=float)
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf)

        assert psf.dtype == np.float64

    def test_all_values_non_negative(self):
        """Test that all PSF values are non-negative."""
        W = np.random.rand(16, 16) * 2.0
        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf, pad_factor=2)

        assert np.all(psf >= -1e-10)  # Small tolerance for roundoff

    def test_input_w_not_mutated(self):
        """Test that input wavefront W is not mutated."""
        W_orig = np.random.rand(8, 8) * 0.1
        W = W_orig.copy()
        wf = WavefrontError(None, None, W)

        DiffractionPSFCalculator.calculate_psf(wf)

        assert np.array_equal(W, W_orig)

    def test_intensity_map_not_mutated(self):
        """Test that input intensity_map is not mutated."""
        W = np.zeros((8, 8), dtype=float)
        intensity_orig = np.ones((8, 8)) * 2.0
        intensity = intensity_orig.copy()

        wf = WavefrontError(None, None, W)
        DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity)

        assert np.array_equal(intensity, intensity_orig)

    def test_non_uniform_intensity_map(self):
        """Test that non-uniform intensity map produces reasonable PSF."""
        W = np.zeros((8, 8), dtype=float)
        # Create a Gaussian intensity profile
        y = np.linspace(-1, 1, 8)
        Y, Z = np.meshgrid(y, y)
        intensity = np.exp(-(Y**2 + Z**2) / 0.5)

        wf = WavefrontError(None, None, W)
        psf = DiffractionPSFCalculator.calculate_psf(wf, intensity_map=intensity)

        # PSF should be normalized
        assert np.isclose(psf.max(), 1.0, atol=1e-9)
        # PSF should have no NaN
        assert not np.any(np.isnan(psf))
