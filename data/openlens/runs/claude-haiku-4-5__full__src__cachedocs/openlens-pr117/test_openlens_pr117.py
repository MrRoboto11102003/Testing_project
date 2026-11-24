import pytest
import numpy as np

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.vector3 import Vector3


class TestVector3Normalize:
    """Tests for Vector3.normalize()"""

    def test_normalize_unit_vector(self):
        """Already-unit vector should normalize to itself"""
        v = Vector3(1, 0, 0)
        normalized = v.normalize()
        assert isinstance(normalized, Vector3)
        assert abs(normalized.x - 1.0) < 1e-6
        assert abs(normalized.y) < 1e-6
        assert abs(normalized.z) < 1e-6

    def test_normalize_arbitrary_vector(self):
        """Arbitrary vector should normalize to magnitude ~1"""
        v = Vector3(3, 4, 0)
        normalized = v.normalize()
        # magnitude should be 5, so (0.6, 0.8, 0)
        assert abs(normalized.x - 0.6) < 1e-6
        assert abs(normalized.y - 0.8) < 1e-6
        assert abs(normalized.z) < 1e-6
        magnitude = np.sqrt(normalized.x**2 + normalized.y**2 + normalized.z**2)
        assert abs(magnitude - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """Zero vector normalizes to zero vector"""
        v = Vector3(0, 0, 0)
        normalized = v.normalize()
        assert normalized.x == 0
        assert normalized.y == 0
        assert normalized.z == 0

    def test_normalize_very_small_magnitude(self):
        """Vectors with magnitude < 1e-15 should return zero vector"""
        v = Vector3(1e-16, 1e-16, 1e-16)
        normalized = v.normalize()
        assert normalized.x == 0
        assert normalized.y == 0
        assert normalized.z == 0

    def test_normalize_negative_components(self):
        """Normalize preserves direction of negative components"""
        v = Vector3(-3, -4, 0)
        normalized = v.normalize()
        assert abs(normalized.x - (-0.6)) < 1e-6
        assert abs(normalized.y - (-0.8)) < 1e-6


class TestBeamSynthesisPropagator:
    """Tests for BeamSynthesisPropagator.propagate_to_image()"""

    def test_propagate_empty_system(self):
        """Empty system should return empty arrays"""
        try:
            from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        except ImportError:
            pytest.skip("beam_synthesis module not available")

        system = OpticalSystem(name="Empty")
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image()

        assert len(dy) == 0
        assert len(dz) == 0
        assert len(intensity) == 0

    def test_propagate_single_lens(self):
        """Single lens system should produce output arrays"""
        try:
            from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        except ImportError:
            pytest.skip("beam_synthesis module not available")

        lens = Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Test System")
        system.add_lens(lens)

        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image(grid_size=16)

        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)

    def test_propagate_detector_size_effect(self):
        """Detector size parameter should affect output shape"""
        try:
            from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
        except ImportError:
            pytest.skip("beam_synthesis module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        propagator = BeamSynthesisPropagator(system)
        dy1, dz1, int1 = propagator.propagate_to_image(
            grid_size=8, detector_pixels=32
        )
        dy2, dz2, int2 = propagator.propagate_to_image(
            grid_size=8, detector_pixels=64
        )

        # Larger detector_pixels should give larger arrays (or same if empty)
        if len(dy1) > 0 and len(dy2) > 0:
            assert dy1.size <= dy2.size


class TestImageQualityAnalyzerMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf()"""

    def test_mtf_empty_system(self):
        """Empty system should return dict with empty arrays"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        system = OpticalSystem(name="Empty")
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert "freq" in result
        assert "mtf_tan" in result
        assert "mtf_sag" in result
        assert len(result["freq"]) == 0

    def test_mtf_returns_correct_keys(self):
        """MTF should return dict with freq, mtf_tan, mtf_sag keys"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf(use_diffraction=False)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}

    def test_mtf_max_freq_filtering(self):
        """Frequencies should not exceed max_freq parameter"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf(max_freq=50.0, use_diffraction=False)

        if len(result["freq"]) > 0:
            assert np.all(result["freq"] <= 50.0)

    def test_mtf_field_angle_variation(self):
        """Different field angles should complete without error"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        for field_angle in [0.0, 5.0, 10.0]:
            result = analyzer.calculate_mtf(
                field_angle_deg=field_angle, use_diffraction=False
            )
            assert isinstance(result, dict)
            assert "freq" in result


class TestImageQualityAnalyzerSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image()"""

    def test_simulate_grayscale_float_image(self):
        """2-D float grayscale image should produce float64 output"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        result = analyzer.simulate_image(img)

        assert result.shape == (32, 32)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_rgb_float_image(self):
        """3-D float RGB image should maintain shape"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.7
        result = analyzer.simulate_image(img)

        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_uint8_input(self):
        """Integer uint8 input should scale and normalize"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        img = np.ones((32, 32), dtype=np.uint8) * 128
        result = analyzer.simulate_image(img)

        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_channel_mismatch_raises(self):
        """ValueError if 3-D image channel count != wavelengths length"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        img = np.ones((32, 32, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="channels"):
            analyzer.simulate_image(img, wavelengths_nm=(405, 550, 700))

    def test_simulate_invalid_dimensions_raises(self):
        """ValueError if image is not 2-D or 3-D"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        img_1d = np.ones(32, dtype=np.float32)

        with pytest.raises(ValueError, match="2-D|3-D"):
            analyzer.simulate_image(img_1d)

    def test_simulate_output_always_normalized(self):
        """Output should always be clipped to [0, 1]"""
        try:
            from openlens.analysis.psf_mtf import ImageQualityAnalyzer
        except ImportError:
            pytest.skip("psf_mtf module not available")

        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        analyzer = ImageQualityAnalyzer(system)
        # High-value input that could saturate
        img = np.ones((32, 32), dtype=np.float64) * 10.0
        result = analyzer.simulate_image(img)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
