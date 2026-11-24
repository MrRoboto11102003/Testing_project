import pytest
import numpy as np
from unittest.mock import Mock
import re

from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.optical_system import OpticalSystem
from openlens.constants import WAVELENGTH_C_LINE, WAVELENGTH_D_LINE, WAVELENGTH_F_LINE


class TestImageQualityAnalyzerSimulateImage:
    """Test suite for ImageQualityAnalyzer.simulate_image."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with empty optical system."""
        return ImageQualityAnalyzer(OpticalSystem())

    @staticmethod
    def mock_psf_identity(**kwargs):
        """Return identity kernel as diffraction PSF (fresh copy each call)."""
        k = np.zeros((3, 3), dtype=np.float64)
        k[1, 1] = 1.0
        return {"image": k}

    # === Input validation ===

    def test_plain_list_raises_attribute_error(self, analyzer):
        """Plain list (no .shape) raises AttributeError before np.asarray."""
        with pytest.raises(AttributeError):
            analyzer.simulate_image([[0.1]])

    def test_invalid_dimensions_1d_raises_value_error(self, analyzer, monkeypatch):
        """1-D array raises ValueError for wrong dimensionality."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        with pytest.raises(
            ValueError, match=r"must be 2-D \(H, W\) or 3-D \(H, W, C\)"
        ):
            analyzer.simulate_image(np.zeros(5))

    def test_invalid_dimensions_0d_raises_value_error(self, analyzer, monkeypatch):
        """0-D scalar array raises ValueError."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        with pytest.raises(
            ValueError, match=r"must be 2-D \(H, W\) or 3-D \(H, W, C\)"
        ):
            analyzer.simulate_image(np.array(1.0))

    def test_invalid_dimensions_4d_raises_value_error(self, analyzer, monkeypatch):
        """4-D array raises ValueError."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        with pytest.raises(
            ValueError, match=r"must be 2-D \(H, W\) or 3-D \(H, W, C\)"
        ):
            analyzer.simulate_image(np.zeros((2, 2, 2, 2)))

    def test_3d_channel_mismatch_raises_value_error(self, analyzer, monkeypatch):
        """3-D array with wrong channel count raises ValueError."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        with pytest.raises(
            ValueError, match=r"image_array has 2 channels but 3 wavelengths"
        ):
            analyzer.simulate_image(np.zeros((4, 4, 2)))

    def test_unexpected_kwargs_raises_type_error(self, analyzer):
        """Unexpected kwarg raises TypeError."""
        with pytest.raises(TypeError, match=r"Unexpected kwargs: \['foo'\]"):
            analyzer.simulate_image(np.zeros((4, 4)), foo=1)

    def test_legacy_pixel_size_kwarg_works(self, analyzer, monkeypatch):
        """Legacy pixel_size kwarg overrides pixel_size_mm."""
        calls = []

        def mock_psf(**kw):
            calls.append(kw)
            return self.mock_psf_identity(**kw)

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        analyzer.simulate_image(np.zeros((4, 4)), pixel_size=0.01)
        # With 0.01: psf_size_mm = max(0.2, 0.64) = 0.64, psf_pixels = 65
        assert calls[0]["sensor_size_mm"] == pytest.approx(0.65, rel=1e-9)

    def test_legacy_wavelengths_kwarg_works(self, analyzer, monkeypatch):
        """Legacy wavelengths kwarg overrides wavelengths_nm."""
        calls = []

        def mock_psf(**kw):
            calls.append(kw)
            return self.mock_psf_identity(**kw)

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        analyzer.simulate_image(np.zeros((4, 4, 2)), wavelengths=(500.0, 600.0))
        assert len(calls) == 2
        assert calls[0]["wavelength_nm"] == 500.0
        assert calls[1]["wavelength_nm"] == 600.0

    # === Output properties ===

    def test_2d_grayscale_shape_and_dtype(self, analyzer, monkeypatch):
        """2-D input produces 2-D float64 output with same shape."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        result = analyzer.simulate_image(np.zeros((8, 8)))
        assert result.shape == (8, 8)
        assert result.dtype == np.float64
        assert result is not np.zeros((8, 8))

    def test_3d_rgb_shape_and_dtype(self, analyzer, monkeypatch):
        """3-D input produces 3-D float64 output with same shape."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        result = analyzer.simulate_image(np.zeros((8, 8, 3)))
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.float64

    def test_uint8_integer_input_scaled_by_max(self, analyzer, monkeypatch):
        """uint8 input 255 scaled to 1.0 before convolution."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        img = np.array([[0, 255], [51, 102]], dtype=np.uint8)
        result = analyzer.simulate_image(img)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-9)
        assert result[1, 0] == pytest.approx(0.2, abs=1e-6)
        assert result[1, 1] == pytest.approx(0.4, abs=1e-6)

    def test_float_input_not_scaled(self, analyzer, monkeypatch):
        """Float input used as-is without scaling."""
        monkeypatch.setattr(analyzer, "calculate_psf", self.mock_psf_identity)
        img = np.array([[0.0, 1.0, 0.5]], dtype=np.float32)
        result = analyzer.simulate_image(img)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-9)
        assert result[0, 2] == pytest.approx(0.5, abs=1e-9)

    # === PSF calculation behavior ===

    def test_calls_calculate_psf_once_per_wavelength(self, analyzer, monkeypatch):
        """calculate_psf called exactly once per wavelength."""
        calls = []

        def mock_psf(**kw):
            calls.append(kw)
            return self.mock_psf_identity(**kw)

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        analyzer.simulate_image(np.zeros((4, 4)))
        assert len(calls) == 3

    def test_all_psf_calls_have_use_diffraction_true(self, analyzer, monkeypatch):
        """Every calculate_psf call has use_diffraction=True."""
        calls = []

        def mock_psf(**kw):
            calls.append(kw)
            return self.mock_psf_identity(**kw)

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        analyzer.simulate_image(np.zeros((4, 4)), field_angle_deg=5.0, focus_shift_mm=0.1)
        for call in calls:
            assert call["use_diffraction"] is True
            assert call["field_angle_deg"] == 5.0
            assert call["focus_shift_mm"] == 0.1

    def test_psf_size_is_odd_and_matches_sensor_size(self, analyzer, monkeypatch):
        """PSF pixels is always odd; sensor_size_mm = pixels * pixel_size."""
        calls = []

        def mock_psf(**kw):
            calls.append(kw)
            return self.mock_psf_identity(**kw)

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        analyzer.simulate_image(np.zeros((4, 4)), pixel_size_mm=0.005)
        for call in calls:
            pixels = call["pixels"]
            assert pixels % 2 == 1
            sensor_size = call["sensor_size_mm"]
            assert sensor_size == pytest.approx(pixels * 0.005, rel=1e-9)

    # === Channel processing ===

    def test_grayscale_uses_mean_of_wavelength_convolutions(self, analyzer, monkeypatch):
        """Grayscale (2-D) output is mean of per-wavelength convolutions."""
        kernels = [
            np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, 0]], dtype=np.float64),
            np.array([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]], dtype=np.float64),
            np.array([[0, 0, 0], [0, 3.0, 0], [0, 0, 0]], dtype=np.float64),
        ]
        idx = [0]

        def mock_psf(**kw):
            k = kernels[idx[0]].copy()
            idx[0] += 1
            return {"image": k}

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf)
        img = np.ones((5, 5), dtype=np.float32)
        result = analyzer.simulate_image(img)
        assert result.shape == (5, 5)
        # Center pixel: mean of 1.0, 2.0, 3.0 = 2.0 (after normalization and scaling)
        assert result.dtype == np.float64

    def test_output_clipped_to_0_to_1(self, analyzer, monkeypatch):
        """Output values outside [0, 1] are clipped."""
        def mock_psf_unnormalized(**kw):
            k = np.ones((3, 3), dtype=np.float64) * 2.0
            return {"image": k}

        monkeypatch.setattr(analyzer, "calculate_psf", mock_psf_unnormalized)
        img = np.ones((5, 5), dtype=np.float32)
        result = analyzer.simulate_image(img)
        assert np.all(result <= 1.0)
        assert np.all(result >= 0.0)

    # === Empty system edge case ===

    def test_empty_optical_system_returns_all_zeros(self, analyzer):
        """Empty OpticalSystem produces all-zero output."""
        result = analyzer.simulate_image(np.ones((8, 8)))
        assert np.all(result == 0.0)
        assert result.dtype == np.float64

    def test_empty_system_3d_rgb_returns_zeros(self, analyzer):
        """Empty system with 3-D input produces all-zero 3-D output."""
        result = analyzer.simulate_image(np.ones((8, 8, 3)))
        assert result.shape == (8, 8, 3)
        assert np.all(result == 0.0)
