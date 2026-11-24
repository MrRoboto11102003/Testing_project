import pytest
import numpy as np
from unittest.mock import Mock

from openlens.vector3 import Vector3
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
from openlens.analysis.psf_mtf import ImageQualityAnalyzer
from openlens.constants import WAVELENGTH_GREEN, NM_TO_MM


class TestVector3Normalize:
    """Tests for Vector3.normalize() method."""

    def test_normalize_basic_vector(self):
        """Test normalization of a simple 3-4-5 triangle."""
        v = Vector3(3, 4, 0)
        result = v.normalize()
        assert result.to_tuple() == (0.6, 0.8, 0.0)
        assert result is not v
        assert v.to_tuple() == (3, 4, 0)  # Original unchanged

    def test_normalize_zero_vector_returns_int_zeros(self):
        """Test zero vector returns zero with int components."""
        v = Vector3(0, 0, 0)
        result = v.normalize()
        assert result.to_tuple() == (0, 0, 0)
        assert type(result.x) is int

    def test_normalize_below_threshold(self):
        """Test that magnitude < 1e-15 returns zero vector."""
        v = Vector3(1e-16, 0, 0)
        result = v.normalize()
        assert result.to_tuple() == (0, 0, 0)
        assert type(result.x) is int

    def test_normalize_above_threshold(self):
        """Test that magnitude >= 1e-15 is normalized."""
        v = Vector3(1e-14, 0, 0)
        result = v.normalize()
        assert result.x == pytest.approx(1.0)
        assert result.y == 0.0
        assert result.z == 0.0

    def test_normalize_preserves_sign(self):
        """Test that normalization preserves sign of components."""
        v = Vector3(-3, 0, -4)
        result = v.normalize()
        assert result.to_tuple() == pytest.approx((-0.6, 0.0, -0.8))


class TestBeamSynthesisPropagatorPropagateToImage:
    """Tests for BeamSynthesisPropagator.propagate_to_image()."""

    def test_empty_system_returns_empty_arrays(self):
        """Test empty system returns three empty arrays."""
        system = OpticalSystem()
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        assert DY.shape == (0,)
        assert DZ.shape == (0,)
        assert I.shape == (0,)

    def test_valid_system_returns_correct_shapes(self):
        """Test valid system returns arrays of correct shape."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        assert DY.shape == (16, 16)
        assert DZ.shape == (16, 16)
        assert I.shape == (16, 16)

    def test_intensity_non_negative_and_finite(self):
        """Test that intensity is non-negative, finite, and real."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=6, detector_pixels=16)
        assert np.isrealobj(I)
        assert (I >= 0).all()
        assert np.isfinite(I).all()

    def test_detector_grid_values(self):
        """Test that detector grid has correct values."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        propagator = BeamSynthesisPropagator(system)
        detector_size = 0.1
        detector_pixels = 16
        DY, DZ, I = propagator.propagate_to_image(
            grid_size=6, detector_size=detector_size, detector_pixels=detector_pixels
        )
        expected_y = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
        expected_z = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
        assert np.allclose(DY[0, :], expected_y)
        assert np.allclose(DZ[:, 0], expected_z)


class TestImageQualityAnalyzerCalculateMTF:
    """Tests for ImageQualityAnalyzer.calculate_mtf()."""

    def test_empty_system_returns_empty_arrays(self):
        """Test empty system returns empty arrays for all MTF components."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf()
        assert result["freq"].shape == (0,)
        assert result["mtf_tan"].shape == (0,)
        assert result["mtf_sag"].shape == (0,)

    def test_mtf_returns_dict_with_correct_structure(self):
        """Test MTF returns dict with correct keys and equal-length arrays."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf(max_freq=50.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"freq", "mtf_tan", "mtf_sag"}
        assert len(result["freq"]) == len(result["mtf_tan"])
        assert len(result["freq"]) == len(result["mtf_sag"])

    def test_mtf_valid_values_and_frequency_properties(self):
        """Test MTF has valid values and frequency properties."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        analyzer = ImageQualityAnalyzer(system)
        result = analyzer.calculate_mtf(max_freq=50.0)
        if len(result["freq"]) > 0:
            assert result["freq"][0] == 0.0
            assert np.isclose(result["mtf_tan"][0], 1.0)
            assert np.isclose(result["mtf_sag"][0], 1.0)
            assert np.all(result["freq"] <= 50.0)
            assert np.all(result["mtf_tan"] >= 0)
            assert np.all(result["mtf_tan"] <= 1 + 1e-9)
            assert np.all(result["mtf_sag"] >= 0)
            assert np.all(result["mtf_sag"] <= 1 + 1e-9)


class TestImageQualityAnalyzerSimulateImage:
    """Tests for ImageQualityAnalyzer.simulate_image()."""

    def test_simulate_image_grayscale_2d(self):
        """Test 2D grayscale input returns 2D float64 output in [0,1]."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        def mock_psf(**kwargs):
            kernel = np.zeros((3, 3))
            kernel[1, 1] = 1.0
            return {"image": kernel}
        
        analyzer.calculate_psf = mock_psf
        input_img = np.ones((16, 16), dtype=np.float32)
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.shape == (16, 16)
        assert result.dtype == np.float64
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_simulate_image_rgb_3d(self):
        """Test 3D RGB input returns 3D float64 output."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        def mock_psf(**kwargs):
            kernel = np.zeros((3, 3))
            kernel[1, 1] = 1.0
            return {"image": kernel}
        
        analyzer.calculate_psf = mock_psf
        input_img = np.ones((16, 16, 3), dtype=np.float32)
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.float64

    def test_simulate_image_raises_on_wrong_dimensions(self):
        """Test ValueError is raised for 1D or 4D input."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        with pytest.raises(ValueError, match="must be 2-D .* or 3-D"):
            analyzer.simulate_image(np.ones(16), pixel_size_mm=0.005)
        
        with pytest.raises(ValueError, match="must be 2-D .* or 3-D"):
            analyzer.simulate_image(np.ones((2, 2, 2, 2)), pixel_size_mm=0.005)

    def test_simulate_image_raises_on_channel_mismatch(self):
        """Test ValueError is raised when channel count does not match wavelengths."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        with pytest.raises(ValueError, match="channels but.*wavelengths"):
            analyzer.simulate_image(
                np.ones((16, 16, 2)),
                pixel_size_mm=0.005,
                wavelengths_nm=(550.0, 600.0, 650.0)
            )

    def test_simulate_image_uint8_scaling(self):
        """Test uint8 input is scaled and output is normalized."""
        system = OpticalSystem()
        analyzer = ImageQualityAnalyzer(system)
        
        def mock_psf(**kwargs):
            kernel = np.zeros((3, 3))
            kernel[1, 1] = 1.0
            return {"image": kernel}
        
        analyzer.calculate_psf = mock_psf
        input_img = np.array([[255, 0]], dtype=np.uint8).reshape((1, 2))
        result = analyzer.simulate_image(input_img, pixel_size_mm=0.005)
        
        assert result.dtype == np.float64
        assert np.all(result >= 0)
        assert np.all(result <= 1)
