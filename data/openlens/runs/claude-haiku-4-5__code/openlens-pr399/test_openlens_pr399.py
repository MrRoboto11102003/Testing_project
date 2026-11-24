import pytest
import numpy as np
import math
from openlens.ray import Ray3D, RefractionResult
from openlens.vector3 import vec3
from openlens.tracer_3d import LensRayTracer3D, SystemRayTracer3D
from openlens.optics_3d import Lens3D
from openlens.system_3d import OpticalSystem3D
from openlens.constants import EPSILON, REFRACTIVE_INDEX_AIR


class TestRay3DPropagate:
    """Tests for Ray3D.propagate method."""

    def test_propagate_positive_distance(self):
        """Test propagation with positive distance moves origin correctly."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.propagate(2.5)
        
        assert ray.origin.x == pytest.approx(2.5)
        assert len(ray.path) == 1
        assert ray.path[0].x == pytest.approx(2.5)

    def test_propagate_negative_distance(self):
        """Test propagation with negative distance."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.propagate(-3.0)
        
        assert ray.origin.x == pytest.approx(-3.0)
        assert len(ray.path) == 1

    def test_propagate_zero_distance_is_noop(self):
        """Test that zero distance does not modify path (no-op)."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        initial_origin = ray.origin
        initial_path_len = len(ray.path)
        
        ray.propagate(0.0)
        
        assert ray.origin == initial_origin
        assert len(ray.path) == initial_path_len

    def test_propagate_negligible_distance_is_noop(self):
        """Test that distance below EPSILON is a no-op."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.propagate(EPSILON / 10.0)
        
        assert len(ray.path) == 0

    def test_propagate_optical_path_length_in_air(self):
        """Test optical path length update in air (n=1)."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.n = 1.0
        
        ray.propagate(5.0)
        
        assert ray.optical_path_length == pytest.approx(5.0)

    def test_propagate_optical_path_length_in_glass(self):
        """Test optical path length update in glass (n > 1)."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.n = 1.5
        
        ray.propagate(2.0)
        
        assert ray.optical_path_length == pytest.approx(3.0)

    def test_propagate_no_duplicate_path_points(self):
        """Test that consecutive identical points are not added to path."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0))
        ray.propagate(1.0)
        ray.propagate(1.0)
        
        assert len(ray.path) == 2
        assert ray.path[0].x == pytest.approx(1.0)
        assert ray.path[1].x == pytest.approx(2.0)

    def test_propagate_3d_normalized_direction(self):
        """Test propagation along normalized direction in 3D."""
        direction = vec3(1, 1, 1).normalize()
        ray = Ray3D(vec3(0, 0, 0), direction)
        
        ray.propagate(math.sqrt(3))
        
        assert ray.origin.x == pytest.approx(1.0)
        assert ray.origin.y == pytest.approx(1.0)
        assert ray.origin.z == pytest.approx(1.0)


class TestLensRayTracer3DTraceSurface:
    """Tests for LensRayTracer3D.trace_surface method."""

    @pytest.fixture
    def plano_convex_lens(self):
        """Create a plano-convex lens tracer."""
        lens = Lens3D(
            diameter=20.0,
            thickness=5.0,
            R1=50.0,
            R2=float('inf'),
            material="BK7"
        )
        return LensRayTracer3D(lens), lens

    def test_trace_surface_invalid_surface_type_returns_missed(self, plano_convex_lens):
        """Test that invalid surface type returns MISSED."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "side")
        
        assert result == RefractionResult.MISSED

    def test_trace_surface_front_surface_refraction(self, plano_convex_lens):
        """Test ray refraction at front surface."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-20, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front", interaction="refract")
        
        assert result in [RefractionResult.REFRACTED, RefractionResult.TOTAL_INTERNAL_REFLECTION]
        assert ray.origin != vec3(-20, 0, 0)

    def test_trace_surface_back_surface_refraction(self, plano_convex_lens):
        """Test ray refraction at back surface."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-2, 0, 0), vec3(1, 0, 0))
        ray.n = 1.5
        
        result = tracer.trace_surface(ray, "back", interaction="refract")
        
        assert result in [RefractionResult.REFRACTED, RefractionResult.TOTAL_INTERNAL_REFLECTION]

    def test_trace_surface_reflection(self, plano_convex_lens):
        """Test reflection interaction at surface."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-20, 0, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front", interaction="reflect")
        
        assert result == RefractionResult.REFLECTED

    def test_trace_surface_ray_outside_aperture_missed(self, plano_convex_lens):
        """Test that ray outside aperture is missed."""
        tracer, lens = plano_convex_lens
        max_height = lens.diameter / 2.0 + 1.0
        ray = Ray3D(vec3(-20, max_height, 0), vec3(1, 0, 0))
        
        result = tracer.trace_surface(ray, "front", interaction="refract")
        
        assert result == RefractionResult.MISSED

    def test_trace_surface_accumulates_optical_path_length(self, plano_convex_lens):
        """Test that optical path length is accumulated for segment."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-20, 0, 0), vec3(1, 0, 0))
        ray.n = 1.0
        initial_opl = ray.optical_path_length
        
        tracer.trace_surface(ray, "front", interaction="refract")
        
        assert ray.optical_path_length > initial_opl

    def test_trace_surface_updates_ray_origin(self, plano_convex_lens):
        """Test that ray origin is updated to intersection point."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-20, 0, 0), vec3(1, 0, 0))
        original_origin = ray.origin
        
        tracer.trace_surface(ray, "front", interaction="refract")
        
        assert ray.origin != original_origin

    def test_trace_surface_path_avoids_duplicates_on_zero_length_hit(self, plano_convex_lens):
        """Test that zero-length hits don't create duplicate path points."""
        tracer, _ = plano_convex_lens
        ray = Ray3D(vec3(-20, 0, 0), vec3(1, 0, 0))
        
        tracer.trace_surface(ray, "front", interaction="refract")
        initial_path_len = len(ray.path)
        
        tracer.trace_surface(ray, "back", interaction="refract")
        
        # Should append a new point, not create a duplicate
        assert len(ray.path) > initial_path_len


class TestSystemRayTracer3DTraceOffAxisRays:
    """Tests for SystemRayTracer3D.trace_off_axis_rays method."""

    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system."""
        system = OpticalSystem3D()
        return SystemRayTracer3D(system)

    @pytest.fixture
    def single_lens_system(self):
        """Create system with one lens."""
        lens = Lens3D(
            diameter=15.0,
            thickness=5.0,
            R1=50.0,
            R2=-50.0,
            material="BK7"
        )
        system = OpticalSystem3D()
        system.add_element(lens, position=0.0)
        return SystemRayTracer3D(system), system, lens

    def test_trace_off_axis_rays_empty_system_returns_empty(self, empty_system):
        """Test that empty system returns empty list."""
        rays = empty_system.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)
        
        assert rays == []

    def test_trace_off_axis_rays_single_ray(self, single_lens_system):
        """Test tracing with num_rays=1."""
        tracer, _, _ = single_lens_system
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=1)
        
        assert len(rays) == 1
        assert isinstance(rays[0], Ray3D)

    def test_trace_off_axis_rays_multiple_rays(self, single_lens_system):
        """Test tracing with multiple rays."""
        tracer, _, _ = single_lens_system
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=7)
        
        assert len(rays) == 7
        for ray in rays:
            assert isinstance(ray, Ray3D)

    def test_trace_off_axis_rays_pupil_distribution(self, single_lens_system):
        """Test that rays are distributed across entrance pupil."""
        tracer, _, lens = single_lens_system
        
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)
        
        y_coords = [ray.origin.y for ray in rays]
        y_min, y_max = min(y_coords), max(y_coords)
        expected_max = lens.diameter / 2.0
        
        assert len(rays) == 5
        assert y_min <= pytest.approx(-expected_max, abs=0.1)
        assert y_max >= pytest.approx(expected_max * 0.5)

    def test_trace_off_axis_rays_field_angle_affects_direction(self, single_lens_system):
        """Test that field angle affects ray direction."""
        tracer, _, _ = single_lens_system
        
        rays_on_axis = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=1)
        rays_off_axis = tracer.trace_off_axis_rays(field_angle_deg=10.0, num_rays=1)
        
        assert rays_on_axis[0].direction.x > rays_off_axis[0].direction.x
        assert rays_off_axis[0].direction.y > 0

    def test_trace_off_axis_rays_custom_wavelength(self, single_lens_system):
        """Test tracing with custom wavelength."""
        tracer, _, _ = single_lens_system
        custom_wavelength = 0.000486
        
        rays = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=2,
            wavelength_mm=custom_wavelength
        )
        
        assert len(rays) == 2
        for ray in rays:
            assert ray.wavelength == pytest.approx(custom_wavelength)

    def test_trace_off_axis_rays_returns_list(self, single_lens_system):
        """Test that return type is a list."""
        tracer, _, _ = single_lens_system
        
        result = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=3)
        
        assert isinstance(result, list)
