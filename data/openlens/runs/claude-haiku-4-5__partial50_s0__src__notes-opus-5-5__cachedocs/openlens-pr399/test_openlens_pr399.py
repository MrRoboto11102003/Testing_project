import pytest
import math
from openlens.ray import Ray3D, RefractionResult
from openlens.vector3 import vec3
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_3d import LensRayTracer3D, SystemRayTracer3D
from openlens import constants as C


class TestRay3DPropagate:
    """Tests for Ray3D.propagate(distance)."""

    def test_propagate_zero_distance_is_noop(self):
        """Zero distance should not change origin, path, or OPL."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)
        path_len_before = len(ray.path)
        opl_before = ray.optical_path_length

        ray.propagate(0.0)

        assert ray.origin.x == pytest.approx(0.0)
        assert len(ray.path) == path_len_before
        assert ray.optical_path_length == pytest.approx(opl_before)

    def test_propagate_positive_distance_moves_origin(self):
        """Positive distance should move origin along unit direction."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)

        ray.propagate(10.0)

        assert ray.origin.x == pytest.approx(10.0, abs=1e-6)
        assert ray.origin.y == pytest.approx(0.0, abs=1e-6)

    def test_propagate_accumulates_optical_path_length(self):
        """OPL should increase by distance * n."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.5)

        ray.propagate(2.0)

        assert ray.optical_path_length == pytest.approx(3.0, abs=1e-6)

    def test_propagate_negative_distance(self):
        """Negative distance should move backward."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)

        ray.propagate(-4.0)

        assert ray.origin.x == pytest.approx(-4.0, abs=1e-6)
        assert ray.optical_path_length == pytest.approx(-4.0, abs=1e-6)

    def test_propagate_direction_normalized(self):
        """Direction should be unit length after initialization."""
        # vec3(3, 4, 0) normalizes to (0.6, 0.8, 0)
        ray = Ray3D(vec3(0, 0, 0), vec3(3, 4, 0), n=1.0)

        ray.propagate(5.0)

        # Moving 5 units along unit direction (0.6, 0.8, 0) gives (3, 4, 0)
        assert ray.origin.x == pytest.approx(3.0, abs=1e-6)
        assert ray.origin.y == pytest.approx(4.0, abs=1e-6)

    def test_propagate_multiple_calls_accumulate(self):
        """Sequential propagations should accumulate distance and OPL."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)

        ray.propagate(1.0)
        ray.propagate(2.0)

        assert ray.origin.x == pytest.approx(3.0, abs=1e-6)
        assert ray.optical_path_length == pytest.approx(3.0, abs=1e-6)


class TestLensRayTracerSurface:
    """Tests for LensRayTracer3D.trace_surface()."""

    @pytest.fixture
    def biconvex_lens(self):
        """Standard biconvex lens for testing."""
        return Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )

    @pytest.fixture
    def tracer(self, biconvex_lens):
        """Ray tracer for test lens."""
        return LensRayTracer3D(biconvex_lens)

    def test_trace_surface_on_axis_refraction(self, tracer):
        """On-axis ray should refract at front surface."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))

        result = tracer.trace_surface(ray, "front")

        assert result == RefractionResult.REFRACTED
        assert ray.origin.x == pytest.approx(0.0, abs=1e-6)
        assert ray.origin.y == pytest.approx(0.0, abs=1e-6)

    def test_trace_surface_accumulates_opl(self, tracer):
        """OPL should increase by segment length times current n."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0), n=1.0)

        tracer.trace_surface(ray, "front")

        # Ray traveled 10 mm through air (n=1)
        assert ray.optical_path_length == pytest.approx(10.0, abs=1e-6)

    def test_trace_surface_aperture_miss_state_unchanged(self, tracer):
        """Aperture miss should return MISSED with state unchanged."""
        ray = Ray3D(vec3(-10, 13, 0), vec3(1, 0, 0))
        origin_x = ray.origin.x
        opl_before = ray.optical_path_length

        result = tracer.trace_surface(ray, "front")

        assert result == RefractionResult.MISSED
        assert ray.origin.x == pytest.approx(origin_x, abs=1e-9)
        assert ray.optical_path_length == pytest.approx(opl_before, abs=1e-9)

    def test_trace_surface_invalid_surface_type(self, tracer):
        """Invalid surface_type should return MISSED."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        opl_before = ray.optical_path_length

        result = tracer.trace_surface(ray, "invalid_type")

        assert result == RefractionResult.MISSED
        assert ray.optical_path_length == pytest.approx(opl_before, abs=1e-9)

    def test_trace_surface_reflect_interaction(self, tracer):
        """Reflect interaction should return REFLECTED."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))

        result = tracer.trace_surface(ray, "front", interaction="reflect")

        assert result == RefractionResult.REFLECTED
        assert ray.origin.x == pytest.approx(0.0, abs=1e-6)


class TestSystemRayTracerOffAxis:
    """Tests for SystemRayTracer3D.trace_off_axis_rays()."""

    @pytest.fixture
    def one_lens_system(self):
        """Optical system with single biconvex lens."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5168
        )
        system = OpticalSystem()
        system.add_lens(lens)
        return system

    @pytest.fixture
    def tracer(self, one_lens_system):
        """Ray tracer for system."""
        return SystemRayTracer3D(one_lens_system)

    def test_trace_off_axis_ray_count(self, tracer):
        """Should return exactly num_rays rays."""
        for num in [1, 3, 7]:
            rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=num)
            assert len(rays) == num

    def test_trace_off_axis_zero_rays(self, tracer):
        """num_rays=0 should return empty list."""
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=0)
        assert len(rays) == 0

    def test_trace_off_axis_empty_system(self):
        """Empty system should return empty list."""
        system = OpticalSystem()
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)

        assert len(rays) == 0

    def test_trace_off_axis_field_zero_z_component_zero(self, tracer):
        """At field 0°, z-component should be zero for all rays."""
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)

        for ray in rays:
            assert abs(ray.origin.z) < 1e-6
            assert abs(ray.direction.z) < 1e-6

    def test_trace_off_axis_field_zero_x_direction_positive(self, tracer):
        """At field 0°, all rays should travel in +x direction."""
        rays = tracer.trace_off_axis_rays(field_angle_deg=0.0, num_rays=5)

        for ray in rays:
            assert ray.direction.x > 0.99

    def test_trace_off_axis_nonzero_field_has_y_component(self, tracer):
        """Nonzero field angle should produce y-component in direction."""
        rays = tracer.trace_off_axis_rays(field_angle_deg=10.0, num_rays=1)

        assert len(rays) == 1
        # At 10°, sin(10°) ≈ 0.174
        assert abs(rays[0].direction.y) > 0.1
