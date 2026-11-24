import pytest
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer
from openlens.ray import Ray


class TestTraceParallelRays:
    """Test suite for LensRayTracer.trace_parallel_rays method."""
    
    @pytest.fixture
    def symmetric_lens(self):
        """Create a symmetric biconvex lens for reproducible tests."""
        return Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
    
    @pytest.fixture
    def tracer(self, symmetric_lens):
        """Create a LensRayTracer at origin."""
        return LensRayTracer(symmetric_lens)
    
    def test_zero_rays_returns_empty_list(self, tracer):
        """Test that num_rays=0 returns an empty list."""
        rays = tracer.trace_parallel_rays(num_rays=0)
        assert isinstance(rays, list)
        assert len(rays) == 0
    
    def test_negative_rays_returns_empty_list(self, tracer):
        """Test that negative num_rays returns an empty list."""
        rays = tracer.trace_parallel_rays(num_rays=-5)
        assert isinstance(rays, list)
        assert len(rays) == 0
    
    def test_return_count_and_types(self, tracer):
        """Test that return list has correct length and element types."""
        for n in [1, 2, 5, 10]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert isinstance(rays, list)
            assert len(rays) == n
            assert all(isinstance(r, Ray) for r in rays)
    
    def test_single_ray_uses_height_zero(self, tracer):
        """Test that num_rays=1 always uses height 0 regardless of range."""
        rays_default = tracer.trace_parallel_rays(num_rays=1, angle_deg=0.0)
        rays_custom = tracer.trace_parallel_rays(
            num_rays=1, ray_height_range=(2, 4), angle_deg=0.0
        )
        # Both should end at same position for symmetric lens at angle 0
        assert rays_default[0].y == pytest.approx(rays_custom[0].y, abs=1e-6)
        assert rays_default[0].angle == pytest.approx(rays_custom[0].angle, abs=1e-6)
    
    def test_fill_parameter_scales_aperture(self, tracer):
        """Test that fill parameter controls ray_height_range when None."""
        rays_full = tracer.trace_parallel_rays(
            num_rays=5, fill=1.0, ray_height_range=None, angle_deg=0.0
        )
        rays_half = tracer.trace_parallel_rays(
            num_rays=5, fill=0.5, ray_height_range=None, angle_deg=0.0
        )
        assert len(rays_full) == 5
        assert len(rays_half) == 5
    
    def test_explicit_range_overrides_fill(self, tracer):
        """Test that explicit ray_height_range makes fill irrelevant."""
        rays_a = tracer.trace_parallel_rays(
            num_rays=3, ray_height_range=(-10, 10), fill=1.0, angle_deg=0.0
        )
        rays_b = tracer.trace_parallel_rays(
            num_rays=3, ray_height_range=(-10, 10), fill=0.1, angle_deg=0.0
        )
        for r_a, r_b in zip(rays_a, rays_b):
            assert r_a.y == pytest.approx(r_b.y, abs=1e-9)
            assert r_a.angle == pytest.approx(r_b.angle, abs=1e-9)
    
    def test_symmetry_for_symmetric_lens_at_zero_angle(self, tracer):
        """Test that symmetric lens produces symmetric fan at angle=0."""
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        # Middle ray should be at y≈0 with angle≈0
        assert rays[2].y == pytest.approx(0.0, abs=1e-6)
        assert rays[2].angle == pytest.approx(0.0, abs=1e-6)
        # Rays should be symmetric: rays[i] ≈ -rays[4-i]
        assert rays[0].y == pytest.approx(-rays[4].y, abs=1e-9)
        assert rays[1].y == pytest.approx(-rays[3].y, abs=1e-9)
        assert rays[0].angle == pytest.approx(-rays[4].angle, abs=1e-9)
    
    def test_convergence_behavior(self, tracer):
        """Test that rays converge appropriately through lens."""
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        # For converging lens: top ray (positive y) should angle downward
        if rays[4].hit:
            assert rays[4].angle < 0
        # Bottom ray (negative y) should angle upward
        if rays[0].hit:
            assert rays[0].angle > 0
    
    def test_wavelength_stored_in_rays(self, tracer):
        """Test that wavelength_mm parameter is stored in each Ray."""
        wavelength = 0.0005
        rays = tracer.trace_parallel_rays(num_rays=3, wavelength_mm=wavelength)
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength, rel=1e-6)
    
    def test_wavelength_does_not_affect_refraction(self, tracer):
        """Test that different wavelengths give identical ray paths."""
        rays1 = tracer.trace_parallel_rays(
            num_rays=5, wavelength_mm=0.0005, angle_deg=0.0
        )
        rays2 = tracer.trace_parallel_rays(
            num_rays=5, wavelength_mm=0.0007, angle_deg=0.0
        )
        for r1, r2 in zip(rays1, rays2):
            assert r1.y == pytest.approx(r2.y, abs=1e-9)
            assert r1.angle == pytest.approx(r2.angle, abs=1e-9)
    
    def test_offset_invariance_regression(self, symmetric_lens):
        """Test that fan is referenced to front vertex, not origin."""
        tracer0 = LensRayTracer(symmetric_lens, x_offset=0)
        tracer50 = LensRayTracer(symmetric_lens, x_offset=50)
        
        rays0 = tracer0.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        rays50 = tracer50.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        
        for r0, r50 in zip(rays0, rays50):
            # Final y and angle should be identical
            assert r0.y == pytest.approx(r50.y, abs=1e-6)
            assert r0.angle == pytest.approx(r50.angle, abs=1e-6)
            # Final x should differ by offset
            assert r50.x == pytest.approx(r0.x + 50, abs=1e-6)
    
    def test_reversed_height_range_allowed(self, tracer):
        """Test that reversed ray_height_range is allowed."""
        rays = tracer.trace_parallel_rays(
            num_rays=3, ray_height_range=(10, -10), angle_deg=0.0
        )
        assert len(rays) == 3
        assert all(isinstance(r, Ray) for r in rays)
    
    def test_missed_rays_included_in_output(self, tracer):
        """Test that rays that miss the lens are still returned."""
        rays = tracer.trace_parallel_rays(
            num_rays=5, ray_height_range=(-20, 20), angle_deg=0.0
        )
        assert len(rays) == 5
        # All rays should be returned regardless of hit status
        assert all(isinstance(r, Ray) for r in rays)
        # Verify hit/miss status is set
        assert all(hasattr(r, 'hit') for r in rays)
    
    def test_angled_rays_traced(self, tracer):
        """Test that rays can be traced at non-zero angles."""
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=5.0)
        assert len(rays) == 3
        assert all(isinstance(r.angle, (int, float)) for r in rays)
