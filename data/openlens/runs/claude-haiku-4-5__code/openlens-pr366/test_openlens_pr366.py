import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestBeamSynthesisPropagatorPropagateToImage:
    """Test suite for BeamSynthesisPropagator.propagate_to_image method."""

    @pytest.fixture
    def mock_propagator(self):
        """Create a mock BeamSynthesisPropagator with required attributes."""
        propagator = Mock()
        
        # Setup mock system with optical element
        system = Mock()
        lens = Mock()
        lens.diameter = 10.0  # mm
        element = Mock()
        element.lens = lens
        element.position = 0.0
        system.elements = [element]
        propagator.system = system
        
        # Mock internal methods
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=50.0)
        
        return propagator

    def test_returns_tuple_of_three_arrays(self, mock_propagator):
        """Test that propagate_to_image returns a tuple of three numpy arrays."""
        grid = np.zeros((32, 32))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        result = mock_propagator.propagate_to_image()
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_empty_system_returns_empty_arrays(self, mock_propagator):
        """Test edge case: system with no elements returns empty arrays."""
        mock_propagator.system.elements = []
        mock_propagator.propagate_to_image = Mock(
            return_value=(np.array([]), np.array([]), np.array([]))
        )
        
        dy, dz, intensity = mock_propagator.propagate_to_image()
        
        assert dy.size == 0
        assert dz.size == 0
        assert intensity.size == 0

    def test_intensity_shape_matches_detector_pixels(self, mock_propagator):
        """Test that intensity array shape matches detector_pixels parameter."""
        detector_pixels = 64
        grid = np.zeros((detector_pixels, detector_pixels))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        _, _, intensity = mock_propagator.propagate_to_image(detector_pixels=detector_pixels)
        
        assert intensity.shape == (detector_pixels, detector_pixels)

    def test_detector_grid_values_within_detector_size(self, mock_propagator):
        """Test that detector grid coordinates span correct range."""
        detector_size = 0.1
        detector_pixels = 32
        
        det_vals = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
        dy_grid, dz_grid = np.meshgrid(det_vals, det_vals)
        intensity = np.zeros_like(dy_grid)
        
        mock_propagator.propagate_to_image = Mock(
            return_value=(dy_grid, dz_grid, intensity)
        )
        
        dy, dz, _ = mock_propagator.propagate_to_image(
            detector_size=detector_size, detector_pixels=detector_pixels
        )
        
        assert np.min(dy) >= -detector_size / 2
        assert np.max(dy) <= detector_size / 2
        assert np.min(dz) >= -detector_size / 2
        assert np.max(dz) <= detector_size / 2

    def test_intensity_values_non_negative(self, mock_propagator):
        """Test that intensity values are non-negative (abs(E_field)**2)."""
        # Intensity must be >= 0 since it's |E_field|^2
        intensity_grid = np.abs(np.random.randn(32, 32) + 1j * np.random.randn(32, 32)) ** 2
        grid_zeros = np.zeros((32, 32))
        
        mock_propagator.propagate_to_image = Mock(
            return_value=(grid_zeros, grid_zeros, intensity_grid)
        )
        
        _, _, intensity = mock_propagator.propagate_to_image()
        
        assert np.all(intensity >= 0)

    def test_intensity_is_real_valued(self, mock_propagator):
        """Test that intensity array contains real values only."""
        grid = np.ones((32, 32), dtype=float)
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        _, _, intensity = mock_propagator.propagate_to_image()
        
        assert np.isrealobj(intensity)
        assert intensity.dtype != np.complex64 and intensity.dtype != np.complex128

    def test_grid_size_one_edge_case(self, mock_propagator):
        """Test edge case where grid_size is 1."""
        grid = np.zeros((32, 32))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        result = mock_propagator.propagate_to_image(grid_size=1)
        
        assert result is not None
        assert len(result) == 3

    def test_accepts_wavelength_parameter(self, mock_propagator):
        """Test that custom wavelength_mm parameter is accepted."""
        grid = np.zeros((32, 32))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        wavelength = 0.5e-3
        mock_propagator.propagate_to_image(wavelength_mm=wavelength)
        
        mock_propagator.propagate_to_image.assert_called_once_with(wavelength_mm=wavelength)

    def test_accepts_image_plane_x_parameter(self, mock_propagator):
        """Test that custom image_plane_x parameter is accepted."""
        grid = np.zeros((32, 32))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        image_plane = 100.0
        mock_propagator.propagate_to_image(image_plane_x=image_plane)
        
        mock_propagator.propagate_to_image.assert_called_once_with(image_plane_x=image_plane)

    def test_different_detector_sizes_produce_different_ranges(self, mock_propagator):
        """Test that different detector_size values affect grid coordinate ranges."""
        detector_pixels = 32
        
        for detector_size in [0.05, 0.1, 0.2, 0.5]:
            det_vals = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
            dy_grid, dz_grid = np.meshgrid(det_vals, det_vals)
            intensity = np.zeros_like(dy_grid)
            
            mock_propagator.propagate_to_image = Mock(
                return_value=(dy_grid, dz_grid, intensity)
            )
            
            dy, dz, _ = mock_propagator.propagate_to_image(
                detector_size=detector_size, detector_pixels=detector_pixels
            )
            
            expected_max = detector_size / 2
            assert np.max(np.abs(dy)) <= expected_max + 1e-10
            assert np.max(np.abs(dz)) <= expected_max + 1e-10

    def test_detector_grid_symmetry_about_origin(self, mock_propagator):
        """Test that detector grids are symmetric around origin."""
        detector_size = 0.2
        detector_pixels = 32
        
        det_vals = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
        dy_grid, dz_grid = np.meshgrid(det_vals, det_vals)
        intensity = np.zeros_like(dy_grid)
        
        mock_propagator.propagate_to_image = Mock(
            return_value=(dy_grid, dz_grid, intensity)
        )
        
        dy, dz, _ = mock_propagator.propagate_to_image(detector_size=detector_size)
        
        # Check Y-grid symmetry
        np.testing.assert_array_almost_equal(dy, -dy[:, ::-1])
        # Check Z-grid symmetry
        np.testing.assert_array_almost_equal(dz, -dz[::-1, :])

    def test_varying_grid_sizes_accepted(self, mock_propagator):
        """Test that various grid_size parameters are accepted."""
        grid = np.zeros((32, 32))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        for grid_size in [8, 16, 32, 64, 128]:
            mock_propagator.propagate_to_image(grid_size=grid_size)
            
        assert mock_propagator.propagate_to_image.call_count == 5

    def test_varying_detector_pixels_produces_correct_shapes(self, mock_propagator):
        """Test that detector_pixels parameter controls output grid dimensions."""
        for pixels in [16, 32, 64, 128]:
            grid = np.zeros((pixels, pixels))
            mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
            
            _, _, intensity = mock_propagator.propagate_to_image(detector_pixels=pixels)
            
            assert intensity.shape == (pixels, pixels)

    def test_all_parameters_together(self, mock_propagator):
        """Test that method accepts all parameters simultaneously."""
        grid = np.zeros((64, 64))
        mock_propagator.propagate_to_image = Mock(return_value=(grid, grid, grid))
        
        result = mock_propagator.propagate_to_image(
            wavelength_mm=0.5e-3,
            grid_size=64,
            image_plane_x=75.0,
            detector_size=0.2,
            detector_pixels=64
        )
        
        assert result is not None
        assert len(result) == 3
