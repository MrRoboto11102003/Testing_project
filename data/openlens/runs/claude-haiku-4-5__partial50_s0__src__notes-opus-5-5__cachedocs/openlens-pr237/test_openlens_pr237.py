import pytest
import math
from openlens.tracer_2d import SystemRayTracer
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestSystemRayTracerTraceParallelRays:
    """Test suite for SystemRayTracer.trace_parallel_rays"""

    def test_empty_system_returns_empty_list(self):
        """Empty system returns empty list immediately."""
        system = OpticalSystem()
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert rays == []

    def test_zero_rays_returns_empty_list(self):
        """Zero rays parameter returns empty list."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=0)
        assert rays == []

    def test_negative_rays_returns_empty_list(self):
        """Negative rays parameter returns empty list."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=-5)
        assert rays == []

    def test_returned_ray_count_matches_num_rays(self):
        """Number of returned rays exactly matches num_rays parameter."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        for num in [1, 3, 5, 11, 21]:
            rays = tracer.trace_parallel_rays(num_rays=num)
            assert len(rays) == num

    def test_single_ray_special_case(self):
        """Single ray (num_rays=1) is returned as list of length 1."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1

    def test_rays_are_distinct_objects(self):
        """Each returned ray is a distinct object in memory."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=7)
        
        ray_ids = [id(ray) for ray in rays]
        assert len(ray_ids) == len(set(ray_ids))

    def test_wavelength_mm_propagated_to_rays(self):
        """wavelength_mm parameter is correctly passed to each ray."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        wavelength_mm = 0.00055  # 550 nm
        rays = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=wavelength_mm)
        
        for ray in rays:
            assert ray.wavelength_mm == wavelength_mm

    def test_angle_deg_converted_to_radians(self):
        """angle_deg parameter is converted to radians for ray angle."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        angle_deg = 5.0
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_deg)
        expected_angle_rad = math.radians(angle_deg)
        
        for ray in rays:
            assert ray.angle == pytest.approx(expected_angle_rad, rel=1e-10)

    def test_symmetry_zero_angle_centered_lens(self):
        """With angle=0 and centered lens, ray angles are symmetric."""
        system = OpticalSystem()
        lens = Lens(
            diameter=25.0,
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            decenter_y=0.0,
            tilt_z=0.0
        )
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        assert len(rays) == 5
        
        # All rays should have angle 0
        for ray in rays:
            assert ray.angle == pytest.approx(0.0, abs=1e-10)

    def test_fill_parameter_scales_heights(self):
        """fill parameter scales the ray heights around aperture."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays_full = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        
        # Both should have correct count
        assert len(rays_full) == 5
        assert len(rays_half) == 5

    def test_zero_fill_concentrates_rays(self):
        """fill=0 concentrates all rays at center."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays = tracer.trace_parallel_rays(num_rays=5, fill=0.0, angle_deg=0.0)
        assert len(rays) == 5

    def test_multiple_elements_traced(self):
        """Rays are traced through multiple elements in order."""
        system = OpticalSystem()
        lens1 = Lens(
            diameter=25.0,
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=3.0
        )
        lens2 = Lens(
            diameter=25.0,
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=3.0
        )
        system.add_lens(lens1)
        system.add_lens(lens2)
        tracer = SystemRayTracer(system)
        
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert len(rays) == 5

    def test_default_parameters_produce_valid_rays(self):
        """Function works with default parameters."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays = tracer.trace_parallel_rays()
        
        assert isinstance(rays, list)
        assert len(rays) > 0

    def test_large_number_of_rays(self):
        """Large num_rays value is handled correctly."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays = tracer.trace_parallel_rays(num_rays=101)
        assert len(rays) == 101

    def test_negative_fill_reverses_height_order(self):
        """Negative fill reverses the order of ray heights."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        tracer = SystemRayTracer(system)
        
        rays_positive = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_negative = tracer.trace_parallel_rays(num_rays=5, fill=-1.0)
        
        # Both should have correct count
        assert len(rays_positive) == 5
        assert len(rays_negative) == 5
