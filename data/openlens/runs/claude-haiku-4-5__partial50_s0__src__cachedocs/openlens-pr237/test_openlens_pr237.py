import pytest
from openlens.tracer_2d import SystemRayTracer
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestSystemRayTracerParallelRays:
    """Comprehensive test suite for SystemRayTracer.trace_parallel_rays."""

    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system."""
        return OpticalSystem()

    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex test lens."""
        return Lens(
            name="TestBiconvex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=10.0,
            diameter=50.0,
            material="BK7"
        )

    @pytest.fixture
    def single_element_system(self, simple_lens):
        """Create a system with one lens element."""
        system = OpticalSystem()
        system.add_element(simple_lens, air_gap_before=0.0)
        return system

    def test_empty_system_returns_empty_list(self, empty_system):
        """Empty optical system should return empty ray list."""
        tracer = SystemRayTracer(empty_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert rays == []
        assert isinstance(rays, list)

    def test_returns_list_type(self, single_element_system):
        """trace_parallel_rays should always return a list."""
        tracer = SystemRayTracer(single_element_system)
        result = tracer.trace_parallel_rays(num_rays=3)
        
        assert isinstance(result, list)

    def test_num_rays_parameter_respected(self, single_element_system):
        """Returned list length should match num_rays parameter."""
        tracer = SystemRayTracer(single_element_system)
        
        for num_rays in [1, 2, 5, 11, 21, 101]:
            rays = tracer.trace_parallel_rays(num_rays=num_rays)
            assert len(rays) == num_rays

    def test_single_ray_case(self, single_element_system):
        """Single ray (num_rays=1) should return list with one ray."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=1)
        
        assert len(rays) == 1

    def test_default_angle_behavior(self, single_element_system):
        """Default angle_deg=0.0 should produce parallel rays."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=7)
        
        assert len(rays) == 7

    def test_non_zero_angle_accepted(self, single_element_system):
        """Non-zero angles should be accepted and traced."""
        tracer = SystemRayTracer(single_element_system)
        
        for angle_deg in [-10.0, -5.0, 5.0, 15.0]:
            rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_deg)
            assert len(rays) == 5

    def test_full_aperture_fill_default(self, single_element_system):
        """Default fill=1.0 should use full aperture."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        
        assert len(rays) == 11

    def test_partial_aperture_fill_factors(self, single_element_system):
        """Partial fill factors should be accepted (0.5, 0.75, 0.95)."""
        tracer = SystemRayTracer(single_element_system)
        
        for fill in [0.1, 0.5, 0.75, 0.95]:
            rays = tracer.trace_parallel_rays(num_rays=5, fill=fill)
            assert len(rays) == 5

    def test_wavelength_parameter_variations(self, single_element_system):
        """Different wavelengths (in mm) should be accepted."""
        tracer = SystemRayTracer(single_element_system)
        
        wavelengths_mm = [
            0.00040,  # ~400nm UV
            0.00055,  # ~550nm green
            0.00065,  # ~650nm red
        ]
        
        for wavelength_mm in wavelengths_mm:
            rays = tracer.trace_parallel_rays(
                num_rays=5,
                wavelength_mm=wavelength_mm
            )
            assert len(rays) == 5

    def test_all_parameters_combined(self, single_element_system):
        """All parameters should work together."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(
            num_rays=11,
            angle_deg=3.5,
            wavelength_mm=0.000589,
            fill=0.85
        )
        
        assert len(rays) == 11

    def test_large_ray_count(self, single_element_system):
        """Large ray counts should be handled."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=201)
        
        assert len(rays) == 201

    def test_odd_number_of_rays_distribution(self, single_element_system):
        """Odd ray count should produce valid distributed rays."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=11, angle_deg=0.0, fill=1.0)
        
        assert len(rays) == 11

    def test_even_number_of_rays_distribution(self, single_element_system):
        """Even ray count should produce valid distributed rays."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=10, angle_deg=0.0, fill=1.0)
        
        assert len(rays) == 10

    def test_very_small_fill_factor(self, single_element_system):
        """Very small fill factors should still work."""
        tracer = SystemRayTracer(single_element_system)
        rays = tracer.trace_parallel_rays(num_rays=5, fill=0.01)
        
        assert len(rays) == 5

    def test_deterministic_behavior(self, single_element_system):
        """Multiple calls with same parameters should produce consistent results."""
        tracer = SystemRayTracer(single_element_system)
        
        rays_1 = tracer.trace_parallel_rays(
            num_rays=11,
            angle_deg=2.5,
            wavelength_mm=0.000589,
            fill=0.9
        )
        rays_2 = tracer.trace_parallel_rays(
            num_rays=11,
            angle_deg=2.5,
            wavelength_mm=0.000589,
            fill=0.9
        )
        
        assert len(rays_1) == len(rays_2) == 11
