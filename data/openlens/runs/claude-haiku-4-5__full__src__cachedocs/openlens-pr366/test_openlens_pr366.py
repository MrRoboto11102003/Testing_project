import pytest
import numpy as np
from unittest.mock import patch
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.constants import WAVELENGTH_GREEN, NM_TO_MM


class TestBeamSynthesisPropagatorPropagateToImage:
    """Tests for BeamSynthesisPropagator.propagate_to_image method."""

    @pytest.fixture
    def simple_lens(self):
        """Create a simple converging lens for testing."""
        return Lens(
            name="test_singlet",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def simple_system(self, simple_lens):
        """Create a minimal optical system with one element."""
        system = OpticalSystem(name="test_system")
        system.add_lens(simple_lens, position=0.0)
        return system

    @pytest.fixture
    def propagator(self, simple_system):
        """Create a BeamSynthesisPropagator instance."""
        return BeamSynthesisPropagator(simple_system)

    def test_returns_tuple_of_three_numpy_arrays(self, propagator):
        """Test that method returns (dy, dz, intensity) as numpy arrays."""
        result = propagator.propagate_to_image()
        assert isinstance(result, tuple)
        assert len(result) == 3
        dy, dz, intensity = result
        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)

    def test_empty_system_returns_empty_arrays(self):
        """Test that system with no elements returns empty arrays."""
        empty_system = OpticalSystem(name="empty")
        propagator = BeamSynthesisPropagator(empty_system)
        dy, dz, intensity = propagator.propagate_to_image()
        assert dy.size == 0
        assert dz.size == 0
        assert intensity.size == 0

    def test_default_detector_pixels_is_64x64(self, propagator):
        """Test that default detector_pixels parameter creates 64x64 output."""
        dy, dz, intensity = propagator.propagate_to_image()
        assert dy.shape == (64, 64)
        assert dz.shape == (64, 64)
        assert intensity.shape == (64, 64)

    def test_custom_detector_pixels_applied(self, propagator):
        """Test that custom detector_pixels parameter changes output shape."""
        for pixels in [16, 32, 48, 96]:
            dy, dz, intensity = propagator.propagate_to_image(detector_pixels=pixels)
            assert dy.shape == (pixels, pixels)
            assert dz.shape == (pixels, pixels)
            assert intensity.shape == (pixels, pixels)

    def test_detector_grid_coordinate_range(self, propagator):
        """Test that detector coordinates span [-size/2, size/2]."""
        detector_size = 0.4
        dy, dz, _ = propagator.propagate_to_image(
            detector_size=detector_size, detector_pixels=10
        )
        half_size = detector_size / 2
        assert np.isclose(dy.min(), -half_size, atol=1e-6)
        assert np.isclose(dy.max(), half_size, atol=1e-6)
        assert np.isclose(dz.min(), -half_size, atol=1e-6)
        assert np.isclose(dz.max(), half_size, atol=1e-6)

    def test_intensity_values_non_negative(self, propagator):
        """Test that intensity is always non-negative (physically valid)."""
        _, _, intensity = propagator.propagate_to_image()
        assert np.all(intensity >= 0.0)

    def test_intensity_dtype_is_float(self, propagator):
        """Test that intensity array contains floating-point values."""
        _, _, intensity = propagator.propagate_to_image()
        assert np.issubdtype(intensity.dtype, np.floating)

    def test_different_wavelengths_produce_valid_output(self, propagator):
        """Test method with different wavelengths."""
        wavelengths = [
            WAVELENGTH_GREEN * NM_TO_MM,
            450.0 * NM_TO_MM,  # blue
            650.0 * NM_TO_MM,  # red
        ]
        for wl in wavelengths:
            dy, dz, intensity = propagator.propagate_to_image(wavelength_mm=wl)
            assert intensity.shape == (64, 64)
            assert np.all(intensity >= 0.0)

    def test_different_grid_sizes_accepted(self, propagator):
        """Test that different beamlet grid sizes are accepted."""
        for grid_size in [4, 8, 16, 32]:
            dy, dz, intensity = propagator.propagate_to_image(
                grid_size=grid_size, detector_pixels=16
            )
            assert intensity.shape == (16, 16)
            assert np.all(intensity >= 0.0)

    def test_custom_image_plane_position_accepted(self, propagator):
        """Test that custom image_plane_x parameter is accepted."""
        for image_plane_x in [50.0, 100.0, 150.0]:
            dy, dz, intensity = propagator.propagate_to_image(
                image_plane_x=image_plane_x, detector_pixels=16
            )
            assert intensity.shape == (16, 16)
            assert np.all(intensity >= 0.0)

    def test_detector_grids_are_meshgrid_format(self, propagator):
        """Test that dy, dz follow numpy meshgrid convention."""
        pixels = 8
        dy, dz, _ = propagator.propagate_to_image(detector_pixels=pixels)
        # In meshgrid format, rows of dz should be identical (DZ constant along row)
        for row in range(pixels):
            assert np.allclose(dz[row, :], dz[row, 0])
        # Columns of dy should be identical (DY constant along column)
        for col in range(pixels):
            assert np.allclose(dy[:, col], dy[0, col])

    def test_deterministic_output(self, propagator):
        """Test that repeated calls produce identical results."""
        params = dict(detector_pixels=16, grid_size=8)
        _, _, intensity1 = propagator.propagate_to_image(**params)
        _, _, intensity2 = propagator.propagate_to_image(**params)
        assert np.allclose(intensity1, intensity2)

    def test_raises_import_error_when_numpy_unavailable(self, propagator):
        """Test that ImportError is raised if numpy is not available."""
        with patch("openlens.analysis.beam_synthesis.NUMPY_AVAILABLE", False):
            with pytest.raises(ImportError, match="numpy is required"):
                propagator.propagate_to_image()
