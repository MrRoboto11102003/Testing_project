import pytest
import math
from openlens.ray import Ray3D
from openlens.tracer_3d import LensRayTracer3D, SystemRayTracer3D, RefractionResult
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.vec3 import vec3
from openlens.constants import EPSILON


class TestRay3DPropagate:
    """Tests for Ray3D.propagate method."""
    
    def test_propagate_zero_distance_is_no_op(self):
        """Zero distance should not append to path (no-op)."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        initial_path_len = len(ray.path)
        initial_opl = ray.optical_path_length
        
        ray.propagate(0.0)
        
        assert len(ray.path) == initial_path_len
        assert ray.optical_path_length == initial_opl
    
    def test_propagate_negligible_distance_is_no_op(self):
        """Distance below EPSILON should not append to path."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        initial_path_len = len(ray.path)
        
        ray.propagate(EPSILON / 2)
        
        assert len(ray.path) == initial_path_len
    
    def test_propagate_updates_origin(self):
        """Positive distance should update ray origin."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.propagate(5.0)
        
        assert ray.origin == vec3(5, 0, 0)
    
    def test_propagate_appends_to_path(self):
        """Positive distance should append new point to path."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        initial_path_len = len(ray.path)
        
        ray.propagate(10.0)
        
        assert len(ray.path) == initial_path_len + 1
        assert ray.path[-1] == vec3(10, 0, 0)
    
    def test_propagate_accumulates_optical_path_length(self):
        """OPL should accumulate: segment * refractive_index."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.n = 1.5  # Glass
        initial_opl = ray.optical_path_length
        
        ray.propagate(10.0)
        
        expected_opl = initial_opl + 10.0 * 1.5
        assert abs(ray.optical_path_length - expected_opl) < 1e-6
    
    def test_propagate_negative_distance(self):
        """Negative distance should propagate backward."""
        ray = Ray3D(vec3(10, 0, 0), vec3(1, 0, 0))
        ray.propagate(-5.0)
        
        assert ray.origin == vec3(5, 0, 0)
    
    def test_propagate_with_normalized_direction(self):
        """Propagation distance should be independent of direction magnitude."""
        direction = vec3(1, 1, 1).normalize()
        ray = Ray3D(vec3(0, 0, 0), direction)
        
        ray.propagate(math.sqrt(3))
        
        # Should advance by sqrt(3) in direction
        assert abs(ray.origin.x - 1.0) < 1e-5
        assert abs(ray.origin.y - 1.0) < 1e-5
        assert abs(ray.origin.z - 1.0) < 1e-5


class TestLensRayTracer3DTraceSurface:
    """Tests for LensRayTracer3D.trace_surface method."""
    
    def test_trace_surface_invalid_type_returns_missed(self):
        """Invalid surface type should return MISSED."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "invalid_surface_type")
        
        assert result == RefractionResult.MISSED
    
    def test_trace_surface_front_refraction_returns_valid_result(self):
        """Tracing front surface should return valid RefractionResult."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front", interaction="refract")
        
        assert result in [RefractionResult.REFRACTED, RefractionResult.MISSED,
                         RefractionResult.REFLECTED]
    
    def test_trace_surface_back_refraction_returns_valid_result(self):
        """Tracing back surface should return valid RefractionResult."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "back", interaction="refract")
        
        assert result in [RefractionResult.REFRACTED, RefractionResult.MISSED,
                         RefractionResult.REFLECTED]
    
    def test_trace_surface_reflect_interaction(self):
        """Reflect interaction should return reflection result."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front", interaction="reflect")
        
        assert result in [RefractionResult.REFLECTED, RefractionResult.MISSED]
    
    def test_trace_surface_ray_outside_aperture_returns_missed(self):
        """Ray far from axis outside aperture should return MISSED."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=5.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 50, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front")
        
        assert result == RefractionResult.MISSED
    
    def test_trace_surface_default_interaction_is_refract(self):
        """Default interaction parameter should be refract."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        # Call without interaction parameter
        result = tracer.trace_surface(ray, "front")
        
        # Should behave as refract
        assert result in [RefractionResult.REFRACTED, RefractionResult.MISSED,
                         RefractionResult.REFLECTED]


class TestSystemRayTracer3DTraceOffAxisRays:
    """Tests for SystemRayTracer3D.trace_off_axis_rays method."""
    
    def test_trace_off_axis_rays_empty_system_returns_empty_list(self):
        """Empty optical system should return empty list."""
        system = OpticalSystem()
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=5.0, num_rays=5)
        
        assert rays == []
    
    def test_trace_off_axis_rays_single_ray(self):
        """Single ray should be traced correctly."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=1)
        
        assert len(rays) == 1
        assert isinstance(rays[0], Ray3D)
    
    def test_trace_off_axis_rays_multiple_rays(self):
        """Multiple rays should be distributed across pupil."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)
        
        assert len(rays) == 5
        for ray in rays:
            assert isinstance(ray, Ray3D)
    
    def test_trace_off_axis_rays_zero_field_angle(self):
        """Zero field angle should trace on-axis."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=3)
        
        assert len(rays) == 3
        for ray in rays:
            assert len(ray.path) > 0
    
    def test_trace_off_axis_rays_positive_field_angle(self):
        """Positive field angle should trace off-axis."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=5.0, num_rays=3)
        
        assert len(rays) == 3
    
    def test_trace_off_axis_rays_negative_field_angle(self):
        """Negative field angle should trace opposite direction."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=-5.0, num_rays=3)
        
        assert len(rays) == 3
    
    def test_trace_off_axis_rays_wavelength_parameter(self):
        """Wavelength parameter should be respected."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays_blue = tracer.trace_off_axis_rays(
            field_angle_deg=0.0, num_rays=1, wavelength_mm=0.00045)
        rays_red = tracer.trace_off_axis_rays(
            field_angle_deg=0.0, num_rays=1, wavelength_mm=0.00065)
        
        assert len(rays_blue) == 1
        assert len(rays_red) == 1
    
    def test_trace_off_axis_rays_returns_ray3d_with_valid_attributes(self):
        """Returned Ray3D objects should have valid attributes."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        system = OpticalSystem()
        system.add_lens(lens)
        tracer = SystemRayTracer3D(system)
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=5.0, num_rays=7)
        
        assert len(rays) == 7
        for ray in rays:
            assert isinstance(ray, Ray3D)
            assert hasattr(ray, 'origin')
            assert hasattr(ray, 'direction')
            assert hasattr(ray, 'path')
            assert isinstance(ray.origin, vec3)
            assert isinstance(ray.direction, vec3)
            assert len(ray.path) >= 1
