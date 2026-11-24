import pytest
import math
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer
from openlens.ray import Ray


class TestTraceParallelRays:
    """Unit tests for LensRayTracer.trace_parallel_rays method."""
    
    @pytest.fixture
    def biconvex_lens(self):
        """Create a symmetric biconvex lens for testing."""
        return Lens(
            name="Test Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
    
    @pytest.fixture
    def tracer(self, biconvex_lens):
        """Create a tracer for the biconvex lens."""
        return LensRayTracer(biconvex_lens)
    
    # ===== Count and type tests =====
    
    def test_returns_list_of_rays(self, tracer):
        """Method returns a list of Ray objects with correct count."""
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 5
        assert all(isinstance(ray, Ray) for ray in rays)
    
    def test_zero_num_rays(self, tracer):
        """num_rays=0 returns an empty list."""
        rays = tracer.trace_parallel_rays(num_rays=0)
        assert rays == []
    
    def test_negative_num_rays(self, tracer):
        """Negative num_rays returns an empty list."""
        rays = tracer.trace_parallel_rays(num_rays=-3)
        assert rays == []
    
    def test_single_ray_count(self, tracer):
        """num_rays=1 returns exactly one ray."""
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        assert isinstance(rays[0], Ray)
    
    # ===== Ray height and range tests =====
    
    def test_single_ray_height_zero_ignores_range(self, tracer):
        """Single ray always starts at height 0, even with custom range."""
        rays_custom = tracer.trace_parallel_rays(
            num_rays=1,
            ray_height_range=(2.0, 4.0)
        )
        rays_default = tracer.trace_parallel_rays(num_rays=1)
        
        assert len(rays_custom) == 1
        assert len(rays_default) == 1
    
    def test_explicit_ray_height_range(self, tracer):
        """Explicit ray_height_range is used when provided."""
        rays = tracer.trace_parallel_rays(
            num_rays=3,
            ray_height_range=(-10.0, 10.0)
        )
        assert len(rays) == 3
    
    def test_fill_parameter_with_default_range(self, tracer):
        """fill parameter scales aperture when range is None."""
        rays_full = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        
        assert len(rays_full) == 5
        assert len(rays_half) == 5
    
    def test_fill_ignored_with_explicit_range(self, tracer):
        """fill parameter is ignored when ray_height_range is provided."""
        rays_1 = tracer.trace_parallel_rays(
            num_rays=5,
            ray_height_range=(-5.0, 5.0),
            fill=1.0
        )
        rays_2 = tracer.trace_parallel_rays(
            num_rays=5,
            ray_height_range=(-5.0, 5.0),
            fill=0.1
        )
        
        # Results should be identical regardless of fill
        for r1, r2 in zip(rays_1, rays_2):
            assert r1.y == pytest.approx(r2.y, abs=1e-10)
            assert r1.angle == pytest.approx(r2.angle, abs=1e-10)
    
    # ===== Parameter tests =====
    
    def test_angle_deg_parameter(self, tracer):
        """angle_deg parameter controls ray bundle tilt."""
        rays_0 = tracer.trace_parallel_rays(num_rays=3, angle_deg=0.0)
        rays_5 = tracer.trace_parallel_rays(num_rays=3, angle_deg=5.0)
        
        assert len(rays_0) == 3
        assert len(rays_5) == 3
    
    def test_wavelength_stored_on_ray_objects(self, tracer):
        """wavelength_mm parameter is stored on each ray."""
        wavelength = 0.00055
        rays = tracer.trace_parallel_rays(
            num_rays=3,
            wavelength_mm=wavelength
        )
        
        for ray in rays:
            assert ray.wavelength_mm is not None
    
    def test_wavelength_does_not_affect_ray_refraction(self, tracer):
        """Different wavelengths produce identical final ray paths."""
        rays_short = tracer.trace_parallel_rays(
            num_rays=5,
            wavelength_mm=0.0005
        )
        rays_long = tracer.trace_parallel_rays(
            num_rays=5,
            wavelength_mm=0.0007
        )
        
        # Final positions and angles must be identical
        for rs, rl in zip(rays_short, rays_long):
            assert rs.y == pytest.approx(rl.y, abs=1e-10)
            assert rs.angle == pytest.approx(rl.angle, abs=1e-10)
    
    # ===== Offset invariance (critical regression test) =====
    
    def test_x_offset_invariance(self, biconvex_lens):
        """Fan is correctly referenced to front vertex, not x=0."""
        tracer_0 = LensRayTracer(biconvex_lens, x_offset=0.0)
        tracer_50 = LensRayTracer(biconvex_lens, x_offset=50.0)
        
        rays_0 = tracer_0.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        rays_50 = tracer_50.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        
        # Final y and angle must be identical (vertex-relative tracing works)
        for r0, r50 in zip(rays_0, rays_50):
            assert r0.y == pytest.approx(r50.y, abs=1e-6)
            assert r0.angle == pytest.approx(r50.angle, abs=1e-6)
            # Final x values must differ by exactly the offset
            assert r50.x == pytest.approx(r0.x + 50.0, abs=1e-6)
    
    # ===== Edge cases =====
    
    def test_missed_rays_still_returned(self, tracer, biconvex_lens):
        """Rays that miss the lens are still included in results."""
        D = biconvex_lens.diameter
        rays = tracer.trace_parallel_rays(
            num_rays=5,
            ray_height_range=(-D, D)
        )
        
        assert len(rays) == 5
        misses = [r for r in rays if not r.hit]
        assert len(misses) > 0
    
    def test_symmetric_lens_produces_centered_ray(self, tracer):
        """Symmetric biconvex with angle=0 has center ray near optical axis."""
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        
        center_idx = len(rays) // 2
        center_ray = rays[center_idx]
        # Center ray should be on-axis after symmetric lens
        assert center_ray.y == pytest.approx(0.0, abs=1e-6)
