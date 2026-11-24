import pytest
import numpy as np
from unittest.mock import patch
from openlens.analysis.diffraction_psf import WavefrontSensor, WavefrontError
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.constants import WAVELENGTH_GREEN


class TestWavefrontSensorGetPupilWavefront:
    """Tests for WavefrontSensor.get_pupil_wavefront following expert guidance."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with one lens."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50,
                             thickness=5, diameter=25))
        return system

    @pytest.fixture
    def multi_lens_system(self):
        """Create an optical system with multiple lenses."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50,
                             thickness=5, diameter=25))
        system.add_lens(Lens(radius_of_curvature_1=40, radius_of_curvature_2=-40,
                             thickness=3, diameter=20))
        return system

    def test_construction_with_optical_system(self, simple_system):
        """Test that WavefrontSensor constructs without error."""
        sensor = WavefrontSensor(simple_system)
        assert sensor is not None

    def test_get_pupil_wavefront_returns_wavefront_error(self, simple_system):
        """Test that get_pupil_wavefront returns a WavefrontError instance."""
        sensor = WavefrontSensor(simple_system)
        result = sensor.get_pupil_wavefront()
        assert isinstance(result, WavefrontError)

    def test_grid_shape_matches_requested_grid_size(self, simple_system):
        """Test that W, Y, Z grids have correct shape matching grid_size."""
        sensor = WavefrontSensor(simple_system)
        for grid_size in [9, 16, 32, 64]:
            result = sensor.get_pupil_wavefront(grid_size=grid_size)
            assert result.W.shape == (grid_size, grid_size)
            assert result.Y.shape == (grid_size, grid_size)
            assert result.Z.shape == (grid_size, grid_size)

    def test_y_z_meshgrid_corner_values(self, simple_system):
        """Test Y and Z meshgrid values at corners and edges."""
        diameter = 25.0
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=50, radius_of_curvature_2=-50,
                             thickness=5, diameter=diameter))
        sensor = WavefrontSensor(system)
        result = sensor.get_pupil_wavefront(grid_size=9)
        
        radius = diameter / 2
        # Y[i, j] == y[j]: Y[0, 0] should be -r, Y[0, -1] should be r
        assert np.isclose(result.Y[0, 0], -radius, atol=1e-10)
        assert np.isclose(result.Y[0, -1], radius, atol=1e-10)
        # Z[i, j] == z[i]: Z[-1, 0] should be r
        assert np.isclose(result.Z[-1, 0], radius, atol=1e-10)

    def test_lens_properties_restored_after_call(self, simple_system):
        """Test that lens refractive_index and wavelength are restored."""
        lens = simple_system.elements[0].lens
        original_index = lens.refractive_index
        original_wavelength = lens.wavelength
        
        sensor = WavefrontSensor(simple_system)
        sensor.get_pupil_wavefront(wavelength_nm=486.1)
        
        assert lens.refractive_index == original_index
        assert lens.wavelength == original_wavelength

    def test_lens_restoration_on_exception(self, simple_system):
        """Test that lenses are restored even when exception is raised."""
        lens = simple_system.elements[0].lens
        original_index = lens.refractive_index
        original_wavelength = lens.wavelength
        
        sensor = WavefrontSensor(simple_system)
        
        # Patch _compute_pupil_wavefront to raise an exception
        with patch.object(sensor, '_compute_pupil_wavefront',
                         side_effect=RuntimeError("Simulated error")):
            with pytest.raises(RuntimeError):
                sensor.get_pupil_wavefront(wavelength_nm=550.0)
        
        # Restoration must succeed despite exception
        assert lens.refractive_index == original_index
        assert lens.wavelength == original_wavelength

    def test_wavelength_applied_during_trace(self, simple_system):
        """Test that wavelength is set on lens during compute."""
        lens = simple_system.elements[0].lens
        sensor = WavefrontSensor(simple_system)
        
        test_wavelength = 486.1
        captured_state = {}
        original_compute = sensor._compute_pupil_wavefront
        
        def spy_compute(field_angle_deg=0.0, wavelength_nm=WAVELENGTH_GREEN,
                       grid_size=64):
            # Capture lens state when _compute_pupil_wavefront is called
            captured_state['wavelength'] = lens.wavelength
            captured_state['index'] = lens.refractive_index
            return original_compute(field_angle_deg=field_angle_deg,
                                   wavelength_nm=wavelength_nm,
                                   grid_size=grid_size)
        
        with patch.object(sensor, '_compute_pupil_wavefront',
                         side_effect=spy_compute):
            sensor.get_pupil_wavefront(wavelength_nm=test_wavelength)
        
        # Verify wavelength was applied during trace
        assert captured_state['wavelength'] == test_wavelength
        # Verify refractive index was set to value at that wavelength
        assert captured_state['index'] == lens.refractive_index_at(test_wavelength)

    def test_all_lenses_restored_in_multi_lens_system(self, multi_lens_system):
        """Test that all lenses are restored in a multi-lens system."""
        originals = []
        for element in multi_lens_system.elements:
            lens = element.lens
            originals.append((lens, lens.refractive_index, lens.wavelength))
        
        sensor = WavefrontSensor(multi_lens_system)
        sensor.get_pupil_wavefront(wavelength_nm=550.0)
        
        for lens, orig_index, orig_wavelength in originals:
            assert lens.refractive_index == orig_index
            assert lens.wavelength == orig_wavelength

    def test_empty_system_returns_empty_arrays(self):
        """Test that an empty system (no elements) returns empty arrays."""
        system = OpticalSystem()
        sensor = WavefrontSensor(system)
        result = sensor.get_pupil_wavefront()
        
        assert result.Y.size == 0
        assert result.Z.size == 0
        assert result.W.size == 0

    def test_corner_samples_are_nan(self, simple_system):
        """Test that corner samples are NaN (outside circular aperture)."""
        sensor = WavefrontSensor(simple_system)
        result = sensor.get_pupil_wavefront(grid_size=16)
        
        # All four corners are outside the circular aperture and must be NaN
        assert np.isnan(result.W[0, 0])
        assert np.isnan(result.W[0, -1])
        assert np.isnan(result.W[-1, 0])
        assert np.isnan(result.W[-1, -1])

    def test_plane_fit_mean_near_zero(self, simple_system):
        """Test that plane-fit orthogonality makes nanmean(W) ≈ 0."""
        sensor = WavefrontSensor(simple_system)
        result = sensor.get_pupil_wavefront(grid_size=16)
        
        # For valid (non-NaN) samples, the mean should be near zero after plane fit
        if not np.all(np.isnan(result.W)):
            mean_w = np.nanmean(result.W)
            assert abs(mean_w) < 1e-6

    def test_kwargs_forwarded_correctly(self, simple_system):
        """Test that field_angle_deg, wavelength_nm, grid_size are forwarded."""
        sensor = WavefrontSensor(simple_system)
        
        captured_kwargs = {}
        
        def spy_compute(field_angle_deg=0.0, wavelength_nm=WAVELENGTH_GREEN,
                       grid_size=64):
            captured_kwargs.update({
                'field_angle_deg': field_angle_deg,
                'wavelength_nm': wavelength_nm,
                'grid_size': grid_size
            })
            return WavefrontError(np.array([]), np.array([]), np.array([]))
        
        with patch.object(sensor, '_compute_pupil_wavefront',
                         side_effect=spy_compute):
            sensor.get_pupil_wavefront(field_angle_deg=5.0,
                                      wavelength_nm=589.0,
                                      grid_size=32)
        
        assert captured_kwargs['field_angle_deg'] == 5.0
        assert captured_kwargs['wavelength_nm'] == 589.0
        assert captured_kwargs['grid_size'] == 32

    def test_tracer_object_not_replaced(self, simple_system):
        """Test that self.tracer object identity is preserved."""
        sensor = WavefrontSensor(simple_system)
        original_tracer = sensor.tracer
        
        sensor.get_pupil_wavefront()
        
        # Tracer should not be replaced; must be the same object
        assert sensor.tracer is original_tracer

    def test_default_parameters(self, simple_system):
        """Test that default parameters produce valid results."""
        sensor = WavefrontSensor(simple_system)
        result = sensor.get_pupil_wavefront()
        
        assert isinstance(result, WavefrontError)
        # Default grid_size is 64
        assert result.W.shape == (64, 64)
