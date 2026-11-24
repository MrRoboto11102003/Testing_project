import pytest
import math
from typing import List

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_2d import LensRayTracer, SystemRayTracer, Ray


class TestLensRayTracerParallelRays:
    """Test suite for LensRayTracer.trace_parallel_rays()"""
    
    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
    
    @pytest.fixture
    def tracer(self, simple_lens):
        """Create a LensRayTracer instance."""
        return LensRayTracer(simple_lens)
    
    def test_returns_list_of_rays(self, tracer):
        """Test that trace_parallel_rays returns list of Ray objects."""
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 5
        assert all(isinstance(ray, Ray) for ray in rays)
    
    def test_single_ray_on_axis(self, tracer):
        """Test tracing single ray (num_rays=1 edge case)."""
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        assert isinstance(rays[0], Ray)
    
    def test_num_rays_parameter(self, tracer):
        """Test that num_rays parameter controls output count."""
        for n in [1, 3, 5, 11, 21]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n
    
    def test_default_fill_uses_full_aperture(self, tracer):
        """Test default fill=1.0 uses full lens diameter/2."""
        rays = tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        heights = sorted([r.y for r in rays])
        
        expected_max = tracer.D / 2
        assert heights[-1] <= expected_max + 0.01
    
    def test_custom_ray_height_range(self, tracer):
        """Test custom ray_height_range parameter."""
        min_h, max_h = -2.0, 2.0
        rays = tracer.trace_parallel_rays(num_rays=5, ray_height_range=(min_h, max_h))
        
        assert len(rays) == 5
        heights = [r.y for r in rays]
        assert min(heights) >= min_h - 0.01
        assert max(heights) <= max_h + 0.01
    
    def test_angled_rays(self, tracer):
        """Test tracing rays at a non-zero angle."""
        angle_deg = 5.0
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_deg)
        
        assert len(rays) == 5
        expected_angle_rad = math.radians(angle_deg)
        for ray in rays:
            assert hasattr(ray, 'angle')
            assert abs(ray.angle - expected_angle_rad) < 1e-6
    
    def test_wavelength_parameter(self, tracer):
        """Test wavelength_mm parameter is set correctly."""
        wavelength_mm = 0.00055  # 550 nm in mm
        rays = tracer.trace_parallel_rays(num_rays=3, wavelength_mm=wavelength_mm)
        
        assert len(rays) == 3
        for ray in rays:
            assert hasattr(ray, 'wavelength_mm')
            assert abs(ray.wavelength_mm - wavelength_mm) < 1e-9
    
    def test_fill_parameter_scales_height(self, tracer):
        """Test that fill parameter properly scales ray heights."""
        rays_full = tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=11, fill=0.5)
        
        max_full = max(abs(r.y) for r in rays_full)
        max_half = max(abs(r.y) for r in rays_half)
        
        assert max_half < max_full
        # With fill=0.5, max height should be approximately half
        assert max_half / max_full < 0.6


class TestSystemRayTracerParallelRays:
    """Test suite for SystemRayTracer.trace_parallel_rays()"""
    
    @pytest.fixture
    def single_element_system(self):
        """Create optical system with one lens element."""
        lens = Lens(
            name="sys_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(lens, position=0.0, air_gap_before=0.0)
        return system
    
    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system."""
        return OpticalSystem()
    
    @pytest.fixture
    def system_tracer(self, single_element_system):
        """Create SystemRayTracer for single element system."""
        return SystemRayTracer(single_element_system)
    
    @pytest.fixture
    def empty_tracer(self, empty_system):
        """Create SystemRayTracer for empty system."""
        return SystemRayTracer(empty_system)
    
    def test_empty_system_returns_empty_list(self, empty_tracer):
        """Test that empty system returns empty ray list."""
        rays = empty_tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 0
    
    def test_returns_list_of_rays(self, system_tracer):
        """Test that trace_parallel_rays returns ray list."""
        rays = system_tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 5
        assert all(isinstance(ray, Ray) for ray in rays)
    
    def test_single_ray_through_system(self, system_tracer):
        """Test single ray through system (num_rays=1 edge case)."""
        rays = system_tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        assert isinstance(rays[0], Ray)
    
    def test_num_rays_parameter_system(self, system_tracer):
        """Test num_rays parameter controls output count."""
        for n in [1, 3, 7, 11]:
            rays = system_tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n
    
    def test_angled_rays_system(self, system_tracer):
        """Test tracing angled rays through system."""
        angle_deg = 7.5
        rays = system_tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_deg)
        
        assert len(rays) == 5
        expected_angle_rad = math.radians(angle_deg)
        for ray in rays:
            assert hasattr(ray, 'angle')
            assert abs(ray.angle - expected_angle_rad) < 1e-6
    
    def test_wavelength_parameter_system(self, system_tracer):
        """Test wavelength_mm parameter for system tracer."""
        wavelength_mm = 0.00050  # 500 nm in mm
        rays = system_tracer.trace_parallel_rays(num_rays=3, wavelength_mm=wavelength_mm)
        
        assert len(rays) == 3
        for ray in rays:
            assert hasattr(ray, 'wavelength_mm')
            assert abs(ray.wavelength_mm - wavelength_mm) < 1e-9
    
    def test_fill_parameter_system(self, system_tracer):
        """Test fill parameter for system tracer."""
        rays_full = system_tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        rays_reduced = system_tracer.trace_parallel_rays(num_rays=11, fill=0.95)
        
        assert len(rays_full) == 11
        assert len(rays_reduced) == 11
        assert all(isinstance(r, Ray) for r in rays_full)
        assert all(isinstance(r, Ray) for r in rays_reduced)
