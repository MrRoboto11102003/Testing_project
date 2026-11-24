import pytest
import numpy as np
from unittest.mock import Mock

try:
    from openlens.vector3 import Vector3
    from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
    from openlens.analysis.psf_mtf import ImageQualityAnalyzer
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")
class TestVector3Normalize:
    """Test Vector3.normalize() method."""

    def test_normalize_known_magnitude(self):
        """Test normalizing vector with known magnitude (3-4-5 triangle)."""
        v = Vector3(3, 4, 0)
        result = v.normalize()
        assert result.x == pytest.approx(0.6)
        assert result.y == pytest.approx(0.8)
        assert result.z == pytest.approx(0.0)

    def test_normalize_result_magnitude_is_one(self):
        """Test that normalized vector has magnitude 1."""
        v = Vector3(2, 3, 6)
        result = v.normalize()
        magnitude = np.sqrt(result.x**2 + result.y**2 + result.z**2)
        assert magnitude == pytest.approx(1.0)

    def test_normalize_zero_vector_returns_zero(self):
        """Test that normalizing zero vector returns zero (edge case)."""
        v = Vector3(0, 0, 0)
        result = v.normalize()
        assert result.x == 0 and result.y == 0 and result.z == 0

    def test_normalize_very_small_vector_threshold(self):
        """Test that vector with magnitude < 1e-15 returns zero."""
        v = Vector3(1e-16, 1e-16, 1e-16)
        result = v.normalize()
        assert result.x == 0 and result.y == 0 and result.z == 0


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")
class TestBeamSynthesisPropagatorPropagateToImage:
    """Test BeamSynthesisPropagator.propagate_to_image() method."""

    def test_propagate_empty_system_returns_empty_arrays(self):
        """Test that empty system returns three empty numpy arrays."""
        propagator = BeamSynthesisPropagator()
        propagator.system = Mock()
        propagator.system.elements = []

        dy, dz, intensity = propagator.propagate_to_image()

        assert isinstance(dy, np.ndarray) and len(dy) == 0
        assert isinstance(dz, np.ndarray) and len(dz) == 0
        assert isinstance(intensity, np.ndarray) and len(intensity) == 0

    def test_propagate_returns_three_arrays(self):
        """Test that propagate_to_image returns exactly 3 numpy arrays."""
        propagator = BeamSynthesisPropagator()

        mock_lens = Mock()
        mock_lens.diameter = 10.0
        mock_elem = Mock()
        mock_elem.lens = mock_lens
        mock_elem.position = 0.0

        propagator.system = Mock()
        propagator.system.elements = [mock_elem]
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=50.0)

        result = propagator.propagate_to_image()

        assert isinstance(result, tuple) and len(result) == 3
        dy, dz, intensity = result
        assert all(isinstance(arr, np.ndarray) for arr in result)


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")
class TestImageQualityAnalyzerCalculateMTF:
    """Test ImageQualityAnalyzer.calculate_mtf() method."""

    def test_calculate_mtf_empty_system(self):
        """Test that empty system returns empty MTF arrays."""
        analyzer = ImageQualityAnalyzer()
        analyzer.system = Mock()
        analyzer.system.elements = []

        result = analyzer.calculate_mtf()

        assert isinstance(result, dict)
        assert all(key in result for key in ["freq", "mtf_tan", "mtf_sag"])
        assert all(len(result[key]) == 0 for key in ["freq", "mtf_tan", "mtf_sag"])

    def test_calculate_mtf_legacy_kwargs(self):
        """Test that calculate_mtf accepts legacy field_angle and max_freq kwargs."""
        analyzer = ImageQualityAnalyzer()
        analyzer.system = Mock()
        analyzer.system.elements = []

        # Should not raise TypeError
        result = analyzer.calculate_mtf(field_angle=15.0, max_freq=200.0)
        assert isinstance(result, dict)
        assert "freq" in result and "mtf_tan" in result and "mtf_sag" in result

    def test_calculate_mtf_rejects_unexpected_kwargs(self):
        """Test that calculate_mtf raises TypeError for unexpected kwargs."""
        analyzer = ImageQualityAnalyzer()
        analyzer.system = Mock()
        analyzer.system.elements = []

        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.calculate_mtf(invalid_param=True)


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not installed")
class TestImageQualityAnalyzerSimulateImage:
    """Test ImageQualityAnalyzer.simulate_image() method."""

    def test_simulate_image_grayscale_2d(self):
        """Test simulate_image accepts and returns grayscale 2D array."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(side_effect=lambda img, kern: img)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.random.rand(100, 100).astype(np.float64)
        result = analyzer.simulate_image(image)

        assert result.shape == (100, 100)
        assert result.dtype == np.float64

    def test_simulate_image_rgb_3d(self):
        """Test simulate_image accepts and returns RGB 3D array."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(side_effect=lambda img, kern: img)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.random.rand(100, 100, 3).astype(np.float64)
        result = analyzer.simulate_image(image)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float64

    def test_simulate_image_uint8_input_scaling(self):
        """Test that integer input (uint8) is scaled to [0, 1]."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(side_effect=lambda img, kern: img)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = analyzer.simulate_image(image)

        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_output_clipped_0_to_1(self):
        """Test that output is clipped to [0, 1] range."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(return_value=np.ones((50, 50)) * 5.0)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.ones((50, 50), dtype=np.float64)
        result = analyzer.simulate_image(image)

        assert np.all(result >= 0) and np.all(result <= 1)

    def test_simulate_image_rejects_1d_input(self):
        """Test that 1D input raises ValueError."""
        analyzer = ImageQualityAnalyzer()

        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(np.array([1, 2, 3]))

    def test_simulate_image_rejects_4d_input(self):
        """Test that 4D input raises ValueError."""
        analyzer = ImageQualityAnalyzer()

        with pytest.raises(ValueError, match="must be 2-D"):
            analyzer.simulate_image(np.random.rand(10, 10, 3, 3))

    def test_simulate_image_channel_count_mismatch(self):
        """Test that 3D image with wrong channel count raises ValueError."""
        analyzer = ImageQualityAnalyzer()

        image = np.random.rand(50, 50, 4)
        with pytest.raises(ValueError, match="channels"):
            analyzer.simulate_image(image, wavelengths_nm=(450, 550, 650))

    def test_simulate_image_legacy_pixel_size_kwarg(self):
        """Test that legacy pixel_size kwarg is accepted."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(side_effect=lambda img, kern: img)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.random.rand(50, 50)
        result = analyzer.simulate_image(image, pixel_size=0.005)

        assert result.shape == (50, 50)

    def test_simulate_image_legacy_wavelengths_kwarg(self):
        """Test that legacy wavelengths kwarg is accepted."""
        analyzer = ImageQualityAnalyzer()
        analyzer._numpy_fftconvolve = Mock(side_effect=lambda img, kern: img)
        analyzer.calculate_psf = Mock(
            return_value={"image": np.ones((32, 32)), "step_size": 0.005}
        )

        image = np.random.rand(50, 50, 3)
        result = analyzer.simulate_image(image, wavelengths=(450, 550, 650))

        assert result.shape == (50, 50, 3)

    def test_simulate_image_rejects_unexpected_kwargs(self):
        """Test that unexpected kwargs raise TypeError."""
        analyzer = ImageQualityAnalyzer()

        with pytest.raises(TypeError, match="Unexpected kwargs"):
            analyzer.simulate_image(np.random.rand(50, 50), invalid_param=True)
