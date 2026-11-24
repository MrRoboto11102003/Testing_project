import pytest
import math
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_2d import LensRayTracer, SystemRayTracer


class TestLensRayTracerParallelRays:
    """Tests for LensRayTracer.trace_parallel_rays"""
    
    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Test Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
    
    @pytest.fixture
    def tracer(self, simple_lens):
        """Create a ray tracer for the test lens."""
        return LensRayTracer(simple_lens)
    
    def test_returns_correct_number_of_rays(self, tracer):
        """Test that trace_parallel_rays returns the requested number of rays."""
        for num_rays in [1, 2, 5, 11]:
            rays = tracer.trace_parallel_rays(num_rays=num_rays)
            assert isinstance(rays, list)
            assert len(rays) == num_rays
    
    def test_single_ray_on_axis(self, tracer):
        """Test that num_rays=1 produces a single ray on optical axis."""
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        # Single ray should be at height 0 (on axis)
        assert rays[0].y_start == pytest.approx(0.0, abs=1e-6)
    
    def test_multiple_rays_span_full_aperture(self, tracer, simple_lens):
        """Test that rays span full aperture with default fill=1.0."""
        num_rays = 11
        rays = tracer.trace_parallel_rays(num_rays=num_rays)
        
        heights = [ray.y_start for ray in rays]
        expected_semi_aperture = simple_lens.diameter / 2
        
        assert min(heights) == pytest.approx(-expected_semi_aperture, abs=1e-6)
        assert max(heights) == pytest.approx(expected_semi_aperture, abs=1e-6)
    
    def test_ray_height_range_parameter_overrides_fill(self, tracer):
        """Test that explicit ray_height_range overrides fill calculation."""
        custom_range = (-8.0, 3.0)
        rays = tracer.trace_parallel_rays(num_rays=11, ray_height_range=custom_range)
        
        heights = [ray.y_start for ray in rays]
        assert min(heights) == pytest.approx(custom_range[0], abs=1e-6)
        assert max(heights) == pytest.approx(custom_range[1], abs=1e-6)
    
    def test_fill_parameter_scales_aperture(self, tracer, simple_lens):
        """Test that fill parameter properly scales the aperture."""
        fill = 0.5
        rays = tracer.trace_parallel_rays(num_rays=11, fill=fill)
        
        heights = [ray.y_start for ray in rays]
        expected_semi_aperture = simple_lens.diameter / 2 * fill
        
        assert max(heights) == pytest.approx(expected_semi_aperture, abs=1e-6)
        assert min(heights) == pytest.approx(-expected_semi_aperture, abs=1e-6)
    
    def test_angle_deg_parameter(self, tracer):
        """Test that angle_deg parameter sets ray direction correctly."""
        angle_degrees = 5.0
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_degrees)
        
        expected_angle_rad = math.radians(angle_degrees)
        for ray in rays:
            assert ray.direction_rad == pytest.approx(expected_angle_rad, abs=1e-6)
    
    def test_wavelength_mm_parameter(self, tracer):
        """Test that wavelength_mm parameter is set on traced rays."""
        wavelength = 0.000550  # 550 nm in mm
        rays = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=wavelength)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength, abs=1e-9)
    
    def test_default_parameters(self, tracer):
        """Test that default parameters produce valid ray traces."""
        rays = tracer.trace_parallel_rays()
        # Should return a list with at least one ray
        assert isinstance(rays, list)
        assert len(rays) > 0
        # All rays should be valid objects
        for ray in rays:
            assert hasattr(ray, 'y_start')
            assert hasattr(ray, 'direction_rad')


class TestSystemRayTracerParallelRays:
    """Tests for SystemRayTracer.trace_parallel_rays"""
    
    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens."""
        return Lens(
            name="Test Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
    
    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system."""
        return OpticalSystem(name="Empty System")
    
    @pytest.fixture
    def single_element_system(self, simple_lens):
        """Create a system with one lens."""
        system = OpticalSystem(name="Single Element")
        system.add_lens(simple_lens)
        return system
    
    @pytest.fixture
    def multi_element_system(self):
        """Create a system with two lenses and an air gap."""
        system = OpticalSystem(name="Multi Element")
        
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=100.0,
            thickness=4.0,
            diameter=20.0,
            material="BK7"
        )
        
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=5.0)
        return system
    
    def test_empty_system_returns_empty_list(self, empty_system):
        """Test that an empty system returns an empty list (documented behavior)."""
        tracer = SystemRayTracer(empty_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert rays == []
        assert isinstance(rays, list)
    
    def test_single_element_system_returns_rays(self, single_element_system):
        """Test basic ray tracing through single-element system."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=7)
        
        assert isinstance(rays, list)
        assert len(rays) == 7
    
    def test_single_ray_on_axis_system(self, single_element_system):
        """Test that num_rays=1 produces ray on optical axis."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=1)
        
        assert len(rays) == 1
        assert rays[0].y_start == pytest.approx(0.0, abs=1e-6)
    
    def test_fill_parameter_scales_entrance_pupil(self, single_element_system):
        """Test that fill parameter scales the entrance pupil."""
        tracer = SystemRayTracer(single_element_system)
        
        rays_full = tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=11, fill=0.5)
        
        heights_full = [ray.y_start for ray in rays_full]
        heights_half = [ray.y_start for ray in rays_half]
        
        # Half fill should produce roughly half the aperture
        full_range = max(heights_full) - min(heights_full)
        half_range = max(heights_half) - min(heights_half)
        assert half_range == pytest.approx(full_range / 2, rel=1e-5)
    
    def test_angle_deg_parameter_system(self, single_element_system):
        """Test off-axis field angle in system."""
        tracer = SystemRayTracer(single_element_system)
        angle_degrees = 8.0
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_degrees)
        
        expected_angle_rad = math.radians(angle_degrees)
        for ray in rays:
            assert ray.direction_rad == pytest.approx(expected_angle_rad, abs=1e-6)
    
    def test_wavelength_parameter_system(self, single_element_system):
        """Test that wavelength is set correctly in system tracing."""
        tracer = SystemRayTracer(single_element_system)
        wavelength = 0.000486  # 486 nm (F-line) in mm
        rays = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=wavelength)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength, abs=1e-9)
    
    def test_multi_element_system_tracing(self, multi_element_system):
        """Test ray tracing through multi-element system."""
        tracer = SystemRayTracer(multi_element_system)
        rays = tracer.trace_parallel_rays(num_rays=9)
        
        assert isinstance(rays, list)
        assert len(rays) == 9
