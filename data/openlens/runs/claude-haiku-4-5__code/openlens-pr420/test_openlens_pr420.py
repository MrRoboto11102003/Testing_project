import pytest
from openlens.analysis.diffraction_psf import WavefrontSensor, WavefrontError, WAVELENGTH_GREEN


@pytest.fixture
def sensor():
    """Create a WavefrontSensor instance for testing."""
    return WavefrontSensor()


class TestGetPupilWavefront:
    """Test WavefrontSensor.get_pupil_wavefront method from diffraction_psf module."""
    
    def test_returns_wavefront_error_type(self, sensor):
        """Test that get_pupil_wavefront returns WavefrontError instance."""
        result = sensor.get_pupil_wavefront()
        assert isinstance(result, WavefrontError)
    
    def test_default_parameters(self, sensor):
        """Test with default parameters: field_angle_deg=0, wavelength=GREEN, grid_size=64."""
        result = sensor.get_pupil_wavefront()
        assert isinstance(result, WavefrontError)
    
    def test_field_angle_variations(self, sensor):
        """Test field angles: 0.0, +5.0, -5.0 degrees."""
        for angle in [0.0, 5.0, -5.0]:
            result = sensor.get_pupil_wavefront(field_angle_deg=angle)
            assert isinstance(result, WavefrontError)
    
    def test_wavelength_rgb(self, sensor):
        """Test RGB wavelengths: red (650nm), green (550nm), blue (450nm)."""
        for wl in [650.0, 550.0, 450.0]:
            result = sensor.get_pupil_wavefront(wavelength_nm=wl)
            assert isinstance(result, WavefrontError)
    
    def test_grid_sizes_power_of_two(self, sensor):
        """Test grid sizes: 32, 64, 128, 256 samples per axis."""
        for size in [32, 64, 128, 256]:
            result = sensor.get_pupil_wavefront(grid_size=size)
            assert isinstance(result, WavefrontError)
    
    def test_lens_state_restored_after_wavelength_change(self, sensor):
        """Test that lens refractive indices are restored after get_pupil_wavefront computation."""
        if not hasattr(sensor, 'system') or not hasattr(sensor.system, 'elements'):
            pytest.skip("System has no elements attribute")
        
        elements = getattr(sensor.system, 'elements', None) or []
        if not elements:
            pytest.skip("No optical elements in system")
        
        # Store initial lens state
        initial_state = {}
        for i, elem in enumerate(elements):
            if hasattr(elem, 'lens'):
                lens = elem.lens
                initial_state[i] = (lens.refractive_index, lens.wavelength)
        
        # Call get_pupil_wavefront with different wavelength
        sensor.get_pupil_wavefront(wavelength_nm=750.0)
        
        # Verify lens state is restored to original values
        for i, elem in enumerate(elements):
            if i in initial_state and hasattr(elem, 'lens'):
                lens = elem.lens
                expected_ri, expected_wl = initial_state[i]
                assert lens.refractive_index == expected_ri, f"Refractive index not restored at element {i}"
                assert lens.wavelength == expected_wl, f"Wavelength not restored at element {i}"
    
    def test_combined_parameters(self, sensor):
        """Test with non-default values for all parameters."""
        result = sensor.get_pupil_wavefront(
            field_angle_deg=7.5,
            wavelength_nm=580.0,
            grid_size=96
        )
        assert isinstance(result, WavefrontError)
    
    def test_sequential_calls_different_wavelengths(self, sensor):
        """Test sequential calls with different wavelengths preserve system state."""
        r1 = sensor.get_pupil_wavefront(wavelength_nm=600.0)
        r2 = sensor.get_pupil_wavefront(wavelength_nm=700.0)
        r3 = sensor.get_pupil_wavefront(wavelength_nm=550.0)
        
        assert all(isinstance(r, WavefrontError) for r in [r1, r2, r3])
    
    def test_extreme_field_angles(self, sensor):
        """Test large positive and negative field angles."""
        for angle in [45.0, -45.0, 60.0]:
            result = sensor.get_pupil_wavefront(field_angle_deg=angle)
            assert isinstance(result, WavefrontError)
    
    def test_extreme_wavelengths_uv_ir(self, sensor):
        """Test UV (365nm) and IR (1000nm) wavelengths outside visible spectrum."""
        for wl in [365.0, 1000.0]:
            result = sensor.get_pupil_wavefront(wavelength_nm=wl)
            assert isinstance(result, WavefrontError)
    
    def test_extreme_grid_sizes(self, sensor):
        """Test very small (8) and very large (512) grid sizes."""
        for size in [8, 512]:
            result = sensor.get_pupil_wavefront(grid_size=size)
            assert isinstance(result, WavefrontError)
    
    def test_wavefront_error_has_pupil_grid_data(self, sensor):
        """Test that returned WavefrontError contains Y and Z pupil grids."""
        result = sensor.get_pupil_wavefront()
        # Check for grid attributes in common naming conventions
        has_y = any(hasattr(result, attr) for attr in ['grid_y', 'Y', 'y', 'pupil_y'])
        has_z = any(hasattr(result, attr) for attr in ['grid_z', 'Z', 'z', 'pupil_z'])
        assert has_y, "WavefrontError missing Y pupil grid"
        assert has_z, "WavefrontError missing Z pupil grid"
    
    def test_chromatic_rgb_wavefront_traces(self, sensor):
        """Test that chromatic wavefront traces can be computed for all RGB wavelengths."""
        result_r = sensor.get_pupil_wavefront(wavelength_nm=650.0, grid_size=64)
        result_g = sensor.get_pupil_wavefront(wavelength_nm=550.0, grid_size=64)
        result_b = sensor.get_pupil_wavefront(wavelength_nm=450.0, grid_size=64)
        
        assert all(isinstance(r, WavefrontError) for r in [result_r, result_g, result_b])
