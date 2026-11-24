import pytest
import numpy as np
import math
from unittest.mock import patch

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.vector3 import Vector3


class TestBeamSynthesisPropagatorPropagateToImage:
    """Tests for BeamSynthesisPropagator.propagate_to_image"""

    def test_empty_system_returns_empty_arrays(self):
        """Empty system should return three empty arrays"""
        system = OpticalSystem()
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, Intensity = propagator.propagate_to_image(
            grid_size=8, detector_pixels=16
        )

        assert DY.shape == (0,)
        assert DZ.shape == (0,)
        assert Intensity.shape == (0,)

    def test_single_lens_returns_valid_shaped_arrays(self):
        """Single lens should return arrays of correct shape"""
        lens = Lens(diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        propagator = BeamSynthesisPropagator(system)

        DY, DZ, I = propagator.propagate_to_image(grid_size=8, detector_pixels=16)

        assert DY.shape == (16, 16)
        assert DZ.shape == (16, 16)
        assert I.shape == (16, 16)
        assert I.dtype == np.float64
        assert np.all(I >= 0)
        assert np.all(np.isfinite(I))

    def test_detector_grid_spans_correct_range(self):
        """Detector grids should span from -size/2 to +size/2"""
        lens = Lens(diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        propagator = BeamSynthesisPropagator(system)

        detector_size = 0.1
        DY, DZ, _ = propagator.propagate_to_image(
            grid_size=8, detector_pixels=32, detector_size=detector_size
        )

        expected_min = -detector_size / 2
        expected_max = detector_size / 2
        assert np.isclose(DY.min(), expected_min, atol=1e-10)
        assert np.isclose(DY.max(), expected_max, atol=1e-10)
        assert np.isclose(DZ.min(), expected_min, atol=1e-10)
        assert np.isclose(DZ.max(), expected_max, atol=1e-10)


class TestImageQualityAnalyzerCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf"""

    def test_empty_system_returns_empty_mtf(self):
        """Empty system should return empty MTF arrays"""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert result["freq"].size == 0
        assert result["mtf_tan"].size == 0
        assert result["mtf_sag"].size == 0

    def test_lens_mtf_has_correct_structure(self):
        """MTF result should have correct structure and properties"""
        lens = Lens()
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf(max_freq=20.0)

        # All arrays should have equal length
        assert len(result["freq"]) == len(result["mtf_tan"]) == len(result["mtf_sag"])

        if len(result["freq"]) > 0:
            # First frequency should be 0
            assert result["freq"][0] == 0.0
            # DC gain should be ~1.0
            assert np.isclose(result["mtf_tan"][0], 1.0, atol=0.01)
            assert np.isclose(result["mtf_sag"][0], 1.0, atol=0.01)
            # All frequencies should respect max_freq
            assert np.all(result["freq"] <= 20.0)
            # MTF values should be in valid range
            assert np.all(result["mtf_tan"] >= -1e-9) and np.all(
                result["mtf_tan"] <= 1 + 1e-9
            )
            assert np.all(result["mtf_sag"] >= -1e-9) and np.all(
                result["mtf_sag"] <= 1 + 1e-9
            )

    def test_unexpected_kwargs_raises_typeerror(self):
        """Unexpected keyword arguments should raise TypeError"""
        lens = Lens()
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = ImageQualityAnalyzer(system)

        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_param=123)


class TestImageQualityAnalyzerSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image"""

    def test_grayscale_image_simulation(self):
        """Grayscale (2-D) image should return grayscale output"""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)

        image = np.ones((16, 16), dtype=np.float64)
        result = analyzer.simulate_image(image)

        assert result.dtype == np.float64
        assert result.ndim == 2
        assert result.shape == (16, 16)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_rgb_image_simulation(self):
        """RGB (3-D) image should return RGB output with same shape"""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)

        image = np.ones((16, 16, 3), dtype=np.float64)
        result = analyzer.simulate_image(image)

        assert result.dtype == np.float64
        assert result.ndim == 3
        assert result.shape == (16, 16, 3)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_dimension_validation_error(self):
        """Invalid dimensions should raise ValueError"""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)

        image_4d = np.ones((4, 4, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(image_4d)

    def test_channel_count_mismatch_error(self):
        """Channel count mismatch should raise ValueError"""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)

        # 2 channels but default wavelengths_nm has 3
        image = np.ones((16, 16, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="channels but"):
            analyzer.simulate_image(image)


class TestVector3Normalize:
    """Tests for Vector3.normalize"""

    def test_zero_vector_normalization(self):
        """Zero vector should return zero vector with int components"""
        v = Vector3(0, 0, 0)
        result = v.normalize()

        assert result.to_tuple() == (0, 0, 0)
        assert type(result.x) is int
        assert type(result.y) is int
        assert type(result.z) is int
        assert result is not v

    def test_pythagorean_triple_normalization(self):
        """3-4-5 triangle should normalize to 0.6-0.8-0"""
        v = Vector3(3, 4, 0)
        result = v.normalize()

        assert np.isclose(result.x, 0.6, atol=1e-15)
        assert np.isclose(result.y, 0.8, atol=1e-15)
        assert np.isclose(result.z, 0.0, atol=1e-15)
        # Original should be unchanged
        assert v.to_tuple() == (3, 4, 0)
        # Result should be a new object
        assert result is not v

    def test_normalized_vector_has_unit_magnitude(self):
        """Normalized vector should have magnitude ~1.0"""
        v = Vector3(1, 1, 1)
        result = v.normalize()

        mag = math.sqrt(result.x**2 + result.y**2 + result.z**2)
        assert np.isclose(mag, 1.0, atol=1e-14)

    def test_very_small_vector_becomes_zero(self):
        """Vector with magnitude < 1e-15 should return zero vector"""
        v = Vector3(1e-16, 1e-16, 0)
        result = v.normalize()

        assert result.to_tuple() == (0, 0, 0)
        assert type(result.x) is int

    def test_negative_component_direction_preservation(self):
        """Normalization should preserve direction including negative signs"""
        v = Vector3(-3, 0, -4)
        result = v.normalize()

        assert np.isclose(result.x, -0.6, atol=1e-15)
        assert np.isclose(result.y, 0.0, atol=1e-15)
        assert np.isclose(result.z, -0.8, atol=1e-15)

    def test_single_axis_vector_normalization(self):
        """Single non-zero component should normalize to ±1.0"""
        v = Vector3(0, -2, 0)
        result = v.normalize()

        assert np.isclose(result.x, 0.0, atol=1e-15)
        assert np.isclose(result.y, -1.0, atol=1e-15)
        assert np.isclose(result.z, 0.0, atol=1e-15)
