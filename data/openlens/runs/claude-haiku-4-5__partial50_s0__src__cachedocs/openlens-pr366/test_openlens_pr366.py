import pytest
import numpy as np
from unittest.mock import patch

try:
    from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
    from openlens.optical_system import OpticalSystem
    from openlens.lens import Lens
    HAS_OPENLENS = True
except ImportError:
    HAS_OPENLENS = False


@pytest.fixture
def simple_optical_system():
    """Create a minimal optical system with one lens for testing."""
    if not HAS_OPENLENS:
        pytest.skip("openlens not available")
    
    lens = Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7"
    )
    system = OpticalSystem()
    system.add_element(lens, air_gap_before=0)
    return system


@pytest.fixture
def propagator(simple_optical_system):
    """Create a BeamSynthesisPropagator instance for testing."""
    return BeamSynthesisPropagator(simple_optical_system)


@pytest.mark.skipif(not HAS_OPENLENS, reason="openlens not available")
class TestBeamSynthesisPropagatorPropagateToImage:
    """Comprehensive test suite for propagate_to_image method."""

    def test_return_type_is_tuple_of_three_ndarrays(self, propagator):
        """Test that return value is a tuple of exactly 3 numpy arrays."""
        result = propagator.propagate_to_image()
        assert isinstance(result, tuple)
        assert len(result) == 3
        Y, Z, Intensity = result
        assert isinstance(Y, np.ndarray)
        assert isinstance(Z, np.ndarray)
        assert isinstance(Intensity, np.ndarray)

    def test_default_parameters_produce_64x64_output(self, propagator):
        """Test default detector_pixels=64 produces 64x64 output grids."""
        Y, Z, Intensity = propagator.propagate_to_image()
        assert Y.shape == (64, 64)
        assert Z.shape == (64, 64)
        assert Intensity.shape == (64, 64)

    def test_custom_detector_pixels_respects_shape(self, propagator):
        """Test that detector_pixels parameter controls output grid shape."""
        for pixels in [8, 16, 32, 128]:
            Y, Z, Intensity = propagator.propagate_to_image(
                detector_pixels=pixels
            )
            assert Y.shape == (pixels, pixels)
            assert Z.shape == (pixels, pixels)
            assert Intensity.shape == (pixels, pixels)

    def test_detector_grids_span_detector_size_range(self, propagator):
        """Test that Y and Z grids span [-detector_size/2, detector_size/2]."""
        detector_size = 0.2
        Y, Z, Intensity = propagator.propagate_to_image(
            detector_size=detector_size,
            detector_pixels=20
        )
        half_size = detector_size / 2
        # Allow small tolerance for numerical discretization
        assert np.allclose(Y.min(), -half_size, atol=1e-6)
        assert np.allclose(Y.max(), half_size, atol=1e-6)
        assert np.allclose(Z.min(), -half_size, atol=1e-6)
        assert np.allclose(Z.max(), half_size, atol=1e-6)

    def test_intensity_values_are_non_negative(self, propagator):
        """Test that intensity (|E_field|^2) is always non-negative."""
        Y, Z, Intensity = propagator.propagate_to_image()
        assert np.all(Intensity >= 0.0)
        assert Intensity.dtype in [np.float64, np.float32, float]

    def test_intensity_contains_no_nan_or_inf(self, propagator):
        """Test that intensity contains only finite values."""
        Y, Z, Intensity = propagator.propagate_to_image()
        assert np.all(np.isfinite(Intensity))

    def test_empty_system_returns_empty_arrays(self):
        """Test that system with no elements returns empty result arrays."""
        if not HAS_OPENLENS:
            pytest.skip("openlens not available")
        empty_system = OpticalSystem()
        propagator = BeamSynthesisPropagator(empty_system)
        Y, Z, Intensity = propagator.propagate_to_image()
        assert Y.size == 0
        assert Z.size == 0
        assert Intensity.size == 0

    def test_all_parameters_simultaneously(self, propagator):
        """Test that all custom parameters work together without error."""
        Y, Z, Intensity = propagator.propagate_to_image(
            wavelength_mm=0.00055,
            grid_size=16,
            image_plane_x=100.0,
            detector_size=0.25,
            detector_pixels=48
        )
        assert Y.shape == (48, 48)
        assert Z.shape == (48, 48)
        assert Intensity.shape == (48, 48)
        assert np.all(Intensity >= 0.0)

    def test_different_wavelengths_are_accepted(self, propagator):
        """Test that method accepts various wavelength values."""
        wavelengths = [0.0004, 0.00055, 0.00070]
        for wavelength in wavelengths:
            Y, Z, Intensity = propagator.propagate_to_image(
                wavelength_mm=wavelength,
                detector_pixels=24
            )
            assert Intensity.shape == (24, 24)
            assert np.all(Intensity >= 0.0)

    def test_grid_size_parameter_affects_beamlet_sampling(self, propagator):
        """Test that grid_size parameter is accepted (affects internal sampling)."""
        for grid_size in [4, 8, 16, 32, 64]:
            Y, Z, Intensity = propagator.propagate_to_image(
                grid_size=grid_size,
                detector_pixels=32
            )
            # Output shape depends on detector_pixels, not grid_size
            assert Intensity.shape == (32, 32)

    def test_explicit_image_plane_x_parameter(self, propagator):
        """Test that explicit image_plane_x parameter is accepted."""
        Y, Z, Intensity = propagator.propagate_to_image(
            image_plane_x=50.0,
            detector_pixels=32
        )
        assert Intensity.shape == (32, 32)
        assert np.all(Intensity >= 0.0)

    def test_detector_grids_match_meshgrid_convention(self, propagator):
        """Test that Y, Z grids follow np.meshgrid convention."""
        detector_size = 0.1
        detector_pixels = 10
        Y, Z, Intensity = propagator.propagate_to_image(
            detector_size=detector_size,
            detector_pixels=detector_pixels
        )
        
        # Verify grids behave like np.meshgrid output
        expected_coords = np.linspace(
            -detector_size / 2,
            detector_size / 2,
            detector_pixels
        )
        expected_Y, expected_Z = np.meshgrid(expected_coords, expected_coords)
        
        np.testing.assert_allclose(Y, expected_Y, rtol=1e-10)
        np.testing.assert_allclose(Z, expected_Z, rtol=1e-10)

    @patch('openlens.analysis.beam_synthesis.NUMPY_AVAILABLE', False)
    def test_numpy_availability_check_raises_import_error(self):
        """Test that ImportError is raised when numpy is not available."""
        if not HAS_OPENLENS:
            pytest.skip("openlens not available")
        
        system = OpticalSystem()
        lens = Lens(diameter=25.0)
        system.add_element(lens)
        propagator = BeamSynthesisPropagator(system)
        
        with pytest.raises(ImportError, match="numpy is required"):
            propagator.propagate_to_image()

    def test_output_arrays_all_same_shape(self, propagator):
        """Test that Y, Z, and Intensity arrays have identical shapes."""
        for pixels in [16, 32, 64]:
            Y, Z, Intensity = propagator.propagate_to_image(
                detector_pixels=pixels
            )
            assert Y.shape == Z.shape == Intensity.shape
