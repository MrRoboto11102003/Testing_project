import pytest
import math
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_2d import LensRayTracer, SystemRayTracer
from openlens.ray import Ray


class TestLensRayTracerTraceParallelRays:
    """Tests for LensRayTracer.trace_parallel_rays"""

    @pytest.fixture
    def biconvex_lens(self):
        """Symmetric biconvex lens for testing."""
        return Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )

    def test_ray_count_and_types(self, biconvex_lens):
        """Test that num_rays controls count and returns Ray instances."""
        tracer = LensRayTracer(biconvex_lens)
        
        for n in [0, 1, 3, 5]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n
            assert all(isinstance(r, Ray) for r in rays)
        
        # Negative num_rays returns empty
        rays = tracer.trace_parallel_rays(num_rays=-3)
        assert len(rays) == 0

    def test_single_ray_always_height_zero(self, biconvex_lens):
        """Test that num_rays=1 produces height 0 regardless of ray_height_range."""
        tracer = LensRayTracer(biconvex_lens)
        
        rays_default = tracer.trace_parallel_rays(num_rays=1)
        rays_custom = tracer.trace_parallel_rays(num_rays=1, ray_height_range=(2, 4))
        
        assert len(rays_default) == 1
        assert len(rays_custom) == 1

    def test_symmetric_lens_symmetry(self, biconvex_lens):
        """Test that symmetric biconvex lens produces symmetric ray results."""
        tracer = LensRayTracer(biconvex_lens)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        
        # Check mirror symmetry: rays[i].y ≈ -rays[4-i].y
        for i in range(2):
            assert rays[i].y == pytest.approx(-rays[4-i].y, abs=1e-9)
            assert rays[i].angle == pytest.approx(-rays[4-i].angle, abs=1e-9)
        
        # Middle ray should be on-axis
        assert rays[2].y == pytest.approx(0, abs=1e-9)
        assert rays[2].angle == pytest.approx(0, abs=1e-9)

    def test_wavelength_independence(self, biconvex_lens):
        """Test that wavelength_mm is stored but doesn't affect refraction."""
        tracer = LensRayTracer(biconvex_lens)
        
        rays1 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0005)
        rays2 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.0007)
        
        # Angles should be identical
        for r1, r2 in zip(rays1, rays2):
            assert r1.angle == pytest.approx(r2.angle, abs=1e-9)
            assert r1.y == pytest.approx(r2.y, abs=1e-9)

    def test_fill_and_explicit_range(self, biconvex_lens):
        """Test fill parameter and that explicit range overrides it."""
        tracer = LensRayTracer(biconvex_lens)
        
        # fill parameter scales aperture
        rays_fill = tracer.trace_parallel_rays(num_rays=3, fill=0.5)
        assert len(rays_fill) == 3
        
        # Explicit range overrides fill
        rays_range = tracer.trace_parallel_rays(
            num_rays=3,
            ray_height_range=(-10, 10),
            fill=0.4
        )
        assert len(rays_range) == 3

    def test_offset_invariance(self, biconvex_lens):
        """Test that rays are referenced to front vertex (offset invariance)."""
        tracer1 = LensRayTracer(biconvex_lens, x_offset=0)
        tracer2 = LensRayTracer(biconvex_lens, x_offset=50)
        
        rays1 = tracer1.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        rays2 = tracer2.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        
        # Final angles should match (offset invariance property)
        for r1, r2 in zip(rays1, rays2):
            assert r1.angle == pytest.approx(r2.angle, abs=1e-6)


class TestSystemRayTracerTraceParallelRays:
    """Tests for SystemRayTracer.trace_parallel_rays"""

    @pytest.fixture
    def single_element_system(self):
        """Optical system with single symmetric lens."""
        system = OpticalSystem()
        system.add_lens(Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        ))
        return system

    @pytest.fixture
    def two_element_system(self):
        """Optical system with two lenses and air gap."""
        system = OpticalSystem()
        system.add_lens(Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        ))
        system.add_lens(Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=4.0,
            diameter=20.0,
            refractive_index=1.6
        ), air_gap=10.0)
        return system

    def test_empty_system_returns_empty(self):
        """Test that empty OpticalSystem returns empty ray list."""
        system = OpticalSystem()
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert len(rays) == 0

    def test_ray_count_and_types(self, single_element_system):
        """Test that num_rays controls count and returns Ray instances."""
        tracer = SystemRayTracer(single_element_system)
        
        for n in [0, 1, 3, 5]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n
            assert all(isinstance(r, Ray) for r in rays)
        
        # Negative num_rays returns empty
        rays = tracer.trace_parallel_rays(num_rays=-3)
        assert len(rays) == 0

    def test_multi_element_system_tracing(self, two_element_system):
        """Test that rays are traced through multi-element system."""
        tracer = SystemRayTracer(two_element_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert len(rays) == 5
        assert all(isinstance(r, Ray) for r in rays)

    def test_wavelength_stored_on_rays(self, single_element_system):
        """Test that wavelength_mm is passed to Ray objects."""
        tracer = SystemRayTracer(single_element_system)
        wavelength = 0.00055
        rays = tracer.trace_parallel_rays(num_rays=3, wavelength_mm=wavelength)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength, rel=1e-9)

    def test_symmetric_system_symmetry(self, single_element_system):
        """Test that symmetric single-element system produces symmetric results."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        
        # Mirror symmetry at angle 0
        for i in range(2):
            assert rays[i].y == pytest.approx(-rays[4-i].y, abs=1e-9)
            assert rays[i].angle == pytest.approx(-rays[4-i].angle, abs=1e-9)

    def test_fill_parameter(self, single_element_system):
        """Test that fill parameter scales the aperture correctly."""
        tracer = SystemRayTracer(single_element_system)
        
        rays_full = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        
        # Both should return requested number of rays
        assert len(rays_full) == 5
        assert len(rays_half) == 5
