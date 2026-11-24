import pytest
import math
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer, SystemRayTracer
from openlens.optical_system import OpticalSystem
from openlens.ray import Ray


class TestLensRayTracerParallelRays:
    """Tests for LensRayTracer.trace_parallel_rays"""

    @pytest.fixture
    def symmetric_lens(self):
        """Create a symmetric biconvex lens for testing."""
        return Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )

    def test_ray_count_and_types(self, symmetric_lens):
        """Test that trace_parallel_rays returns correct number and types."""
        tracer = LensRayTracer(symmetric_lens)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert len(rays) == 5
        assert all(isinstance(ray, Ray) for ray in rays)

    def test_symmetry_and_convergence(self, symmetric_lens):
        """Test ray symmetry at angle_deg=0 and proper convergence."""
        tracer = LensRayTracer(symmetric_lens)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        
        # Symmetric rays about optical axis
        for i in range(len(rays) // 2):
            assert rays[i].y == pytest.approx(-rays[-(i+1)].y, abs=1e-9)
            assert rays[i].angle == pytest.approx(-rays[-(i+1)].angle, abs=1e-9)
        
        # Middle ray on-axis
        middle = rays[2]
        assert middle.y == pytest.approx(0.0, abs=1e-9)
        assert middle.angle == pytest.approx(0.0, abs=1e-9)
        
        # Top ray bends downward, bottom ray bends upward (convergence)
        assert rays[-1].angle < 0
        assert rays[0].angle > 0

    def test_edge_cases_empty_and_single_ray(self, symmetric_lens):
        """Test zero/negative rays and single ray edge cases."""
        tracer = LensRayTracer(symmetric_lens)
        
        assert tracer.trace_parallel_rays(num_rays=0) == []
        assert tracer.trace_parallel_rays(num_rays=-5) == []
        
        # Single ray should be on-axis even with custom range
        rays = tracer.trace_parallel_rays(num_rays=1, ray_height_range=(2, 4))
        assert len(rays) == 1
        assert rays[0].angle == pytest.approx(0.0, abs=1e-9)

    def test_fill_and_explicit_range(self, symmetric_lens):
        """Test fill parameter and that explicit range overrides it."""
        tracer = LensRayTracer(symmetric_lens)
        
        # Explicit range should override fill
        rays_explicit = tracer.trace_parallel_rays(
            num_rays=3, ray_height_range=(-10, 10), fill=0.1
        )
        rays_no_fill = tracer.trace_parallel_rays(
            num_rays=3, ray_height_range=(-10, 10)
        )
        
        assert len(rays_explicit) == len(rays_no_fill) == 3

    def test_wavelength_independence(self, symmetric_lens):
        """Test wavelength_mm is stored but doesn't affect refraction."""
        tracer = LensRayTracer(symmetric_lens)
        
        rays_wl1 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0005)
        rays_wl2 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0007)
        
        # Tracing results identical (wavelength doesn't change refraction)
        for r1, r2 in zip(rays_wl1, rays_wl2):
            assert r1.angle == pytest.approx(r2.angle, abs=1e-9)
            assert r1.y == pytest.approx(r2.y, abs=1e-9)
        
        # But wavelengths stored should differ
        assert rays_wl1[0].wavelength_mm == pytest.approx(0.0005, rel=1e-6)
        assert rays_wl2[0].wavelength_mm == pytest.approx(0.0007, rel=1e-6)

    def test_offset_invariance_with_angle(self, symmetric_lens):
        """Test x_offset is properly accounted for (referencing bug fix)."""
        tracer_0 = LensRayTracer(symmetric_lens, x_offset=0.0)
        tracer_50 = LensRayTracer(symmetric_lens, x_offset=50.0)
        
        rays_0 = tracer_0.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        rays_50 = tracer_50.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        
        # Final angles and y should match
        for r0, r50 in zip(rays_0, rays_50):
            assert r0.angle == pytest.approx(r50.angle, abs=1e-6)
            assert r0.y == pytest.approx(r50.y, abs=1e-6)
            # Final x should differ by offset
            assert r0.x == pytest.approx(r50.x - 50.0, abs=1e-6)

    def test_angled_ray_fan(self, symmetric_lens):
        """Test angled rays maintain angle through lens."""
        tracer = LensRayTracer(symmetric_lens)
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=10.0)
        
        angle_rad = math.radians(10.0)
        # All rays should start with this angle
        for ray in rays:
            assert ray.angle == pytest.approx(angle_rad, abs=1e-9)


class TestSystemRayTracerParallelRays:
    """Tests for SystemRayTracer.trace_parallel_rays"""

    @pytest.fixture
    def single_lens_system(self):
        """Create optical system with single symmetric lens."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        system.add_lens(lens)
        return system

    def test_empty_system(self):
        """Test that empty optical system returns empty list."""
        empty_system = OpticalSystem()
        tracer = SystemRayTracer(empty_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert rays == []

    def test_ray_count_and_types(self, single_lens_system):
        """Test ray count and types for single-element system."""
        tracer = SystemRayTracer(single_lens_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert len(rays) == 5
        assert all(isinstance(ray, Ray) for ray in rays)
        # Rays must be distinct objects
        assert len(set(id(ray) for ray in rays)) == 5

    def test_zero_rays_and_symmetry(self, single_lens_system):
        """Test zero rays and symmetry at angle_deg=0."""
        tracer = SystemRayTracer(single_lens_system)
        
        # Zero rays
        assert tracer.trace_parallel_rays(num_rays=0) == []
        
        # Symmetry
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        for i in range(len(rays) // 2):
            assert rays[i].y == pytest.approx(-rays[-(i+1)].y, abs=1e-9)
            assert rays[i].angle == pytest.approx(-rays[-(i+1)].angle, abs=1e-9)
        
        # Middle ray on-axis
        middle = rays[2]
        assert middle.y == pytest.approx(0.0, abs=1e-9)
        assert middle.angle == pytest.approx(0.0, abs=1e-9)

    def test_single_ray_and_wavelength(self, single_lens_system):
        """Test single ray and wavelength independence."""
        tracer = SystemRayTracer(single_lens_system)
        
        # Single ray on-axis
        rays = tracer.trace_parallel_rays(num_rays=1, angle_deg=0.0)
        assert len(rays) == 1
        assert rays[0].y == pytest.approx(0.0, abs=1e-9)
        
        # Wavelength independence
        rays_wl1 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0005)
        rays_wl2 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0007)
        for r1, r2 in zip(rays_wl1, rays_wl2):
            assert r1.angle == pytest.approx(r2.angle, abs=1e-9)

    def test_fill_parameter(self, single_lens_system):
        """Test fill parameter effect on rays."""
        tracer = SystemRayTracer(single_lens_system)
        
        rays_full = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        
        # Both should return 5 rays
        assert len(rays_full) == 5
        assert len(rays_half) == 5

    def test_cross_check_with_lens_tracer(self, single_lens_system):
        """Test single-element system matches equivalent LensRayTracer."""
        lens = single_lens_system.elements[0].lens
        pos = single_lens_system.elements[0].position
        
        sys_tracer = SystemRayTracer(single_lens_system)
        lens_tracer = LensRayTracer(lens, x_offset=pos)
        
        # Trace with same parameters
        sys_rays = sys_tracer.trace_parallel_rays(
            num_rays=5, angle_deg=3.0, fill=1.0
        )
        lens_rays = lens_tracer.trace_parallel_rays(
            num_rays=5, angle_deg=3.0, fill=1.0
        )
        
        # Final angles should match (same refraction)
        for sys_ray, lens_ray in zip(sys_rays, lens_rays):
            assert sys_ray.angle == pytest.approx(lens_ray.angle, abs=1e-6)
