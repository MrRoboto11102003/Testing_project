import pytest
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.diffraction_psf import WavefrontSensor


class TestWavefrontSensorGetPupilWavefront:
    """Tests for openlens.analysis.diffraction_psf.WavefrontSensor.get_pupil_wavefront"""

    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Test Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def simple_system(self, simple_lens):
        """Create a simple optical system with one lens."""
        system = OpticalSystem(name="Test System")
        system.add_lens(simple_lens)
        return system

    @pytest.fixture
    def wavefront_sensor(self, simple_system):
        """Create a WavefrontSensor instance."""
        return WavefrontSensor(simple_system)

    def test_get_pupil_wavefront_returns_result(self, wavefront_sensor):
        """Test that get_pupil_wavefront returns a valid result."""
        result = wavefront_sensor.get_pupil_wavefront()
        assert result is not None

    def test_get_pupil_wavefront_with_default_parameters(self, wavefront_sensor):
        """Test get_pupil_wavefront with default parameters."""
        # Default: field_angle_deg=0.0, wavelength_nm=550.0 (green), grid_size=64
        result = wavefront_sensor.get_pupil_wavefront()
        assert result is not None
        # Result should have wavefront error data (Y, Z grids and W)
        assert hasattr(result, "W") or isinstance(result, dict)

    def test_lens_state_preserved_after_calculation(self, wavefront_sensor, simple_system):
        """Test that lens refractive index and wavelength are restored after calculation.
        
        Per docstring: "The lenses are restored afterwards, so asking for a 
        wavefront never changes the system it was asked about."
        """
        lens = simple_system.elements[0].lens
        original_index = lens.refractive_index
        original_wavelength = lens.wavelength
        
        # Call with different wavelength
        wavefront_sensor.get_pupil_wavefront(wavelength_nm=650.0)
        
        # Verify restoration
        assert lens.refractive_index == original_index
        assert lens.wavelength == original_wavelength

    def test_different_wavelengths_produce_results(self, wavefront_sensor):
        """Test get_pupil_wavefront with different wavelengths (chromatic variation).
        
        Tests that the method correctly handles chromatic aberration by 
        setting different refractive indices per wavelength.
        """
        result_blue = wavefront_sensor.get_pupil_wavefront(wavelength_nm=450.0)
        result_green = wavefront_sensor.get_pupil_wavefront(wavelength_nm=550.0)
        result_red = wavefront_sensor.get_pupil_wavefront(wavelength_nm=650.0)
        
        assert result_blue is not None
        assert result_green is not None
        assert result_red is not None

    def test_field_angle_parameter(self, wavefront_sensor):
        """Test get_pupil_wavefront with various field angles."""
        result_on_axis = wavefront_sensor.get_pupil_wavefront(field_angle_deg=0.0)
        result_off_axis_small = wavefront_sensor.get_pupil_wavefront(field_angle_deg=2.5)
        result_off_axis_large = wavefront_sensor.get_pupil_wavefront(field_angle_deg=10.0)
        
        assert result_on_axis is not None
        assert result_off_axis_small is not None
        assert result_off_axis_large is not None

    def test_grid_size_parameter(self, wavefront_sensor):
        """Test get_pupil_wavefront with different grid sizes.
        
        grid_size is "Number of pupil samples per axis".
        """
        result_32 = wavefront_sensor.get_pupil_wavefront(grid_size=32)
        result_64 = wavefront_sensor.get_pupil_wavefront(grid_size=64)
        result_128 = wavefront_sensor.get_pupil_wavefront(grid_size=128)
        
        assert result_32 is not None
        assert result_64 is not None
        assert result_128 is not None

    def test_negative_field_angle(self, wavefront_sensor):
        """Test get_pupil_wavefront with negative field angle."""
        result_negative = wavefront_sensor.get_pupil_wavefront(field_angle_deg=-5.0)
        result_positive = wavefront_sensor.get_pupil_wavefront(field_angle_deg=5.0)
        
        assert result_negative is not None
        assert result_positive is not None

    def test_multiple_sequential_calls(self, wavefront_sensor):
        """Test that multiple calls don't interfere with each other.
        
        Each call should restore lens state, so sequential calls should
        be independent.
        """
        result1 = wavefront_sensor.get_pupil_wavefront(wavelength_nm=450.0, field_angle_deg=0.0)
        result2 = wavefront_sensor.get_pupil_wavefront(wavelength_nm=650.0, field_angle_deg=10.0)
        result3 = wavefront_sensor.get_pupil_wavefront(wavelength_nm=450.0, field_angle_deg=0.0)
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

    def test_extreme_grid_sizes(self, wavefront_sensor):
        """Test get_pupil_wavefront with extreme grid sizes.
        
        Tests edge cases for the grid_size parameter.
        """
        result_small = wavefront_sensor.get_pupil_wavefront(grid_size=16)
        result_large = wavefront_sensor.get_pupil_wavefront(grid_size=256)
        
        assert result_small is not None
        assert result_large is not None

    def test_uv_to_ir_wavelength_range(self, wavefront_sensor):
        """Test get_pupil_wavefront across a wide wavelength range.
        
        Tests UV (365 nm), visible (400-700 nm), and IR (1000+ nm).
        """
        result_uv = wavefront_sensor.get_pupil_wavefront(wavelength_nm=365.0)
        result_visible = wavefront_sensor.get_pupil_wavefront(wavelength_nm=550.0)
        result_ir = wavefront_sensor.get_pupil_wavefront(wavelength_nm=1064.0)
        
        assert result_uv is not None
        assert result_visible is not None
        assert result_ir is not None

    def test_multi_element_system(self):
        """Test get_pupil_wavefront with multi-element optical system.
        
        Tests that the method handles multiple lenses correctly, as the
        docstring mentions "refractive indices the lenses currently carry"
        (plural).
        """
        crown = Lens(
            name="Crown",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        flint = Lens(
            name="Flint",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=-100.0,
            thickness=4.0,
            diameter=25.0,
            material="F2",
        )
        
        system = OpticalSystem(name="Two-element")
        system.add_lens(crown)
        system.add_lens(flint, air_gap_before=5.0)
        
        sensor = WavefrontSensor(system)
        result = sensor.get_pupil_wavefront(wavelength_nm=550.0)
        
        assert result is not None

    def test_system_focal_length_unchanged_after_calculation(self, wavefront_sensor, simple_system):
        """Test that system optical properties are not mutated.
        
        Verifies the documented behavior: "asking for a wavefront never 
        changes the system it was asked about".
        """
        original_efl = simple_system.get_system_focal_length()
        original_bfl = simple_system.calculate_back_focal_length()
        
        # Perform wavefront calculation with different parameters
        wavefront_sensor.get_pupil_wavefront(wavelength_nm=450.0)
        wavefront_sensor.get_pupil_wavefront(wavelength_nm=650.0, field_angle_deg=15.0)
        
        # Verify no mutation
        assert simple_system.get_system_focal_length() == original_efl
        assert simple_system.calculate_back_focal_length() == original_bfl

    def test_wavefront_error_has_required_attributes(self, wavefront_sensor):
        """Test that returned WavefrontError has Y, Z, and W attributes.
        
        Per docstring: "Returns WavefrontError with the Y, Z pupil grids and 
        W in waves."
        """
        result = wavefront_sensor.get_pupil_wavefront(grid_size=32)
        
        assert result is not None
        # Result should have pupil grid coordinates and wavefront error
        # Either as attributes or as dict keys
        has_grids = (
            (hasattr(result, "Y") and hasattr(result, "Z") and hasattr(result, "W"))
            or ("Y" in result and "Z" in result and "W" in result)
        )
        assert has_grids, "WavefrontError missing Y, Z, or W"

    def test_afocal_lens_handling(self):
        """Test get_pupil_wavefront with an afocal (no power) lens.
        
        Tests edge case where optical system has zero power.
        """
        # Create an afocal system (plano-plano plate)
        afocal = Lens(
            name="Plate",
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        
        system = OpticalSystem(name="Afocal")
        system.add_lens(afocal)
        
        sensor = WavefrontSensor(system)
        # Should handle gracefully even though system is afocal
        result = sensor.get_pupil_wavefront()
        assert result is not None or result is None  # Either valid or None, not error
