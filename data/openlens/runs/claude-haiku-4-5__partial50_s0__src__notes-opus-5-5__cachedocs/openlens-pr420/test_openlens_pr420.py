import pytest
import numpy as np
from unittest.mock import patch

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.diffraction_psf import WavefrontSensor, WavefrontError
from openlens.constants import WAVELENGTH_GREEN


class TestWavefrontSensorGetPupilWavefront:
    """Tests for WavefrontSensor.get_pupil_wavefront() from diffraction_psf module."""

    @pytest.fixture
    def simple_system(self):
        """Create a system with one biconvex lens."""
        system = OpticalSystem()
        system.add_lens(Lens(
            radius_of_curvature_1=50,
            radius_of_curvature_2=-50,
            thickness=5,
            diameter=25,
            material="BK7"
        ))
        return system

    @pytest.fixture
    def sensor(self, simple_system):
        """Create a WavefrontSensor for the simple system."""
        return WavefrontSensor(simple_system)

    def test_returns_wavefront_error_instance(self, sensor):
        """Result is an instance of WavefrontError."""
        result = sensor.get_pupil_wavefront()
        assert isinstance(result, WavefrontError)

    def test_result_has_required_attributes(self, sensor):
        """WavefrontError has Y, Z, and W attributes."""
        result = sensor.get_pupil_wavefront()
        assert hasattr(result, 'Y')
        assert hasattr(result, 'Z')
        assert hasattr(result, 'W')

    def test_output_grid_shape_matches_grid_size(self, sensor):
        """Y, Z, W shapes are (grid_size, grid_size)."""
        grid_size = 9
        result = sensor.get_pupil_wavefront(grid_size=grid_size)
        assert result.Y.shape == (grid_size, grid_size)
        assert result.Z.shape == (grid_size, grid_size)
        assert result.W.shape == (grid_size, grid_size)

    def test_grid_coordinates_span_aperture_diameter(self, sensor, simple_system):
        """Y and Z grids span [-radius, radius] with radius = diameter/2."""
        grid_size = 9
        result = sensor.get_pupil_wavefront(grid_size=grid_size)
        
        lens = simple_system.elements[0].lens
        radius = lens.diameter / 2
        
        assert np.isclose(result.Y[0, 0], -radius)
        assert np.isclose(result.Y[0, -1], radius)
        assert np.isclose(result.Z[0, 0], -radius)
        assert np.isclose(result.Z[-1, 0], radius)

    def test_aperture_corners_produce_nan(self, sensor):
        """All four corners (outside circular aperture) are NaN."""
        result = sensor.get_pupil_wavefront(grid_size=9)
        assert np.isnan(result.W[0, 0])
        assert np.isnan(result.W[0, -1])
        assert np.isnan(result.W[-1, 0])
        assert np.isnan(result.W[-1, -1])

    def test_lens_refractive_index_restored(self, simple_system):
        """Lens refractive_index is restored to original after call."""
        lens = simple_system.elements[0].lens
        original_index = lens.refractive_index
        
        sensor = WavefrontSensor(simple_system)
        sensor.get_pupil_wavefront(wavelength_nm=486.1)
        
        assert lens.refractive_index == original_index

    def test_lens_wavelength_restored(self, simple_system):
        """Lens wavelength is restored to original after call."""
        lens = simple_system.elements[0].lens
        original_wavelength = lens.wavelength
        
        sensor = WavefrontSensor(simple_system)
        sensor.get_pupil_wavefront(wavelength_nm=656.3)
        
        assert lens.wavelength == original_wavelength

    def test_lens_properties_restored_on_exception(self, simple_system):
        """Lens properties are restored even when computation raises."""
        lens = simple_system.elements[0].lens
        original_index = lens.refractive_index
        original_wavelength = lens.wavelength
        
        sensor = WavefrontSensor(simple_system)
        with patch.object(sensor, '_compute_pupil_wavefront',
                         side_effect=RuntimeError("Injected test error")):
            with pytest.raises(RuntimeError):
                sensor.get_pupil_wavefront()
        
        assert lens.refractive_index == original_index
        assert lens.wavelength == original_wavelength

    def test_wavelength_applied_to_lens_during_computation(self, simple_system):
        """Requested wavelength is applied to lens before _compute_pupil_wavefront."""
        lens = simple_system.elements[0].lens
        sensor = WavefrontSensor(simple_system)
        
        test_wavelength = 486.1
        captured_wavelengths = []
        
        def spy_compute(*args, **kwargs):
            captured_wavelengths.append(lens.wavelength)
            return WavefrontError(np.array([]), np.array([]), np.array([]))
        
        with patch.object(sensor, '_compute_pupil_wavefront', side_effect=spy_compute):
            sensor.get_pupil_wavefront(wavelength_nm=test_wavelength)
        
        assert len(captured_wavelengths) == 1
        assert captured_wavelengths[0] == test_wavelength

    def test_empty_system_returns_empty_arrays(self):
        """Empty system returns WavefrontError with size-0 arrays."""
        empty_system = OpticalSystem()
        sensor = WavefrontSensor(empty_system)
        result = sensor.get_pupil_wavefront()
        
        assert result.Y.size == 0
        assert result.Z.size == 0
        assert result.W.size == 0

    def test_grid_size_one_edge_case(self, sensor):
        """grid_size=1 produces (1, 1) NaN at corner (outside aperture)."""
        result = sensor.get_pupil_wavefront(grid_size=1)
        assert result.W.shape == (1, 1)
        assert np.isnan(result.W[0, 0])

    def test_tracer_object_identity_preserved(self, sensor):
        """WavefrontSensor.tracer object identity is not changed by call."""
        original_tracer = sensor.tracer
        sensor.get_pupil_wavefront()
        assert sensor.tracer is original_tracer

    def test_plane_fit_removes_piston(self, sensor):
        """Valid (non-NaN) wavefront samples have mean ≈ 0 after plane fit."""
        result = sensor.get_pupil_wavefront(grid_size=15)
        
        valid_mask = ~np.isnan(result.W)
        if np.any(valid_mask):
            mean_value = np.nanmean(result.W)
            assert abs(mean_value) < 0.1

    def test_different_field_angles_produce_valid_results(self, sensor):
        """get_pupil_wavefront accepts different field_angle_deg values."""
        result_0 = sensor.get_pupil_wavefront(field_angle_deg=0.0)
        result_5 = sensor.get_pupil_wavefront(field_angle_deg=5.0)
        
        assert isinstance(result_0, WavefrontError)
        assert isinstance(result_5, WavefrontError)
        assert result_0.W.shape == result_5.W.shape

    def test_different_wavelengths_produce_valid_results(self, sensor):
        """get_pupil_wavefront accepts different wavelength_nm values."""
        result_green = sensor.get_pupil_wavefront(wavelength_nm=WAVELENGTH_GREEN)
        result_red = sensor.get_pupil_wavefront(wavelength_nm=656.3)
        result_blue = sensor.get_pupil_wavefront(wavelength_nm=486.1)
        
        assert isinstance(result_green, WavefrontError)
        assert isinstance(result_red, WavefrontError)
        assert isinstance(result_blue, WavefrontError)
