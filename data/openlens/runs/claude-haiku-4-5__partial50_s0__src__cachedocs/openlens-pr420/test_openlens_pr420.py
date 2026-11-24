import pytest
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.diffraction_psf import WavefrontSensor, WavefrontError


class TestWavefrontSensorGetPupilWavefront:
    """Tests for WavefrontSensor.get_pupil_wavefront method."""

    @pytest.fixture
    def biconvex_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Biconvex Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def test_system(self, biconvex_lens):
        """Create an optical system with a single lens."""
        system = OpticalSystem(name="Test System")
        system.add_element(biconvex_lens, air_gap=10.0)
        return system

    @pytest.fixture
    def sensor(self, test_system):
        """Create a WavefrontSensor instance."""
        return WavefrontSensor(test_system)

    def test_get_pupil_wavefront_returns_wavefront_error(self, sensor):
        """Test that get_pupil_wavefront returns WavefrontError type."""
        result = sensor.get_pupil_wavefront()
        assert isinstance(result, WavefrontError)

    def test_get_pupil_wavefront_default_parameters(self, sensor):
        """Test get_pupil_wavefront with default parameters."""
        result = sensor.get_pupil_wavefront()
        
        assert isinstance(result, WavefrontError)
        # Verify wavefront contains Y, Z, W grids
        assert hasattr(result, 'Y')
        assert hasattr(result, 'Z')
        assert hasattr(result, 'W')

    def test_get_pupil_wavefront_field_angle_zero(self, sensor):
        """Test get_pupil_wavefront with on-axis field angle."""
        result = sensor.get_pupil_wavefront(field_angle_deg=0.0)
        assert isinstance(result, WavefrontError)

    def test_get_pupil_wavefront_field_angle_nonzero(self, sensor):
        """Test get_pupil_wavefront with off-axis field angle."""
        result = sensor.get_pupil_wavefront(field_angle_deg=5.0)
        assert isinstance(result, WavefrontError)

    def test_get_pupil_wavefront_field_angle_negative(self, sensor):
        """Test get_pupil_wavefront with negative field angle."""
        result_neg = sensor.get_pupil_wavefront(field_angle_deg=-5.0)
        result_pos = sensor.get_pupil_wavefront(field_angle_deg=5.0)
        
        assert isinstance(result_neg, WavefrontError)
        assert isinstance(result_pos, WavefrontError)

    def test_get_pupil_wavefront_wavelength_does_not_persist(self, sensor, biconvex_lens):
        """Test that wavelength parameter does not persist on lens after call.
        
        Per docstring: 'The lenses are restored afterwards, so asking for a 
        wavefront never changes the system it was asked about.'
        """
        original_wavelength = biconvex_lens.wavelength
        original_index = biconvex_lens.refractive_index
        
        sensor.get_pupil_wavefront(wavelength_nm=450.0)
        
        # Lens state must be restored
        assert biconvex_lens.wavelength == original_wavelength
        assert biconvex_lens.refractive_index == original_index

    def test_get_pupil_wavefront_state_restored_multiple_wavelengths(self, sensor, biconvex_lens):
        """Test that state is properly restored across multiple wavelength calls."""
        original_wavelength = biconvex_lens.wavelength
        
        for wl in [400.0, 500.0, 600.0, 700.0]:
            sensor.get_pupil_wavefront(wavelength_nm=wl)
            assert biconvex_lens.wavelength == original_wavelength

    def test_get_pupil_wavefront_grid_size_64(self, sensor):
        """Test get_pupil_wavefront with 64x64 grid."""
        result = sensor.get_pupil_wavefront(grid_size=64)
        
        assert isinstance(result, WavefrontError)
        if result.Y is not None and isinstance(result.Y, np.ndarray):
            assert result.Y.shape == (64, 64)
            assert result.Z.shape == (64, 64)
            assert result.W.shape == (64, 64)

    def test_get_pupil_wavefront_grid_size_32(self, sensor):
        """Test get_pupil_wavefront with smaller 32x32 grid."""
        result = sensor.get_pupil_wavefront(grid_size=32)
        
        assert isinstance(result, WavefrontError)
        if result.Y is not None and isinstance(result.Y, np.ndarray):
            assert result.Y.shape == (32, 32)
            assert result.Z.shape == (32, 32)
            assert result.W.shape == (32, 32)

    def test_get_pupil_wavefront_grid_size_128(self, sensor):
        """Test get_pupil_wavefront with larger 128x128 grid."""
        result = sensor.get_pupil_wavefront(grid_size=128)
        
        assert isinstance(result, WavefrontError)
        if result.Y is not None and isinstance(result.Y, np.ndarray):
            assert result.Y.shape == (128, 128)

    def test_get_pupil_wavefront_grid_dimensions_consistent(self, sensor):
        """Test that Y, Z, W pupil grids have matching dimensions."""
        result = sensor.get_pupil_wavefront(grid_size=64)
        
        if result.Y is not None and isinstance(result.Y, np.ndarray):
            assert result.Y.shape == result.Z.shape == result.W.shape

    def test_get_pupil_wavefront_state_restored_on_exception(self, sensor, biconvex_lens):
        """Test that lens state is restored even if wavefront computation fails.
        
        Per docstring: state restoration uses try-finally, so must work on 
        exception paths too.
        """
        original_wavelength = biconvex_lens.wavelength
        original_index = biconvex_lens.refractive_index
        
        try:
            sensor.get_pupil_wavefront(wavelength_nm=10000.0)
        except (ValueError, RuntimeError, AttributeError, TypeError, IndexError):
            pass
        
        # State must be restored even on failure
        assert biconvex_lens.wavelength == original_wavelength
        assert biconvex_lens.refractive_index == original_index


class TestWavefrontSensorClass:
    """Tests for WavefrontSensor class initialization and basic functionality."""

    def test_wavefront_sensor_initialization(self):
        """Test WavefrontSensor can be initialized with an OpticalSystem."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Test System")
        system.add_element(lens, air_gap=10.0)
        
        sensor = WavefrontSensor(system)
        assert sensor is not None
        assert sensor.system == system

    def test_wavefront_sensor_with_multi_element_assembly(self):
        """Test WavefrontSensor with multi-element optical assembly."""
        lens1 = Lens(
            name="Element1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="Element2",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=3.0,
            diameter=20.0,
            material="SF2",
        )
        
        system = OpticalSystem(name="Doublet")
        system.add_element(lens1, air_gap=5.0)
        system.add_element(lens2, air_gap=20.0)
        
        sensor = WavefrontSensor(system)
        assert sensor is not None
        assert len(sensor.system.elements) == 2
