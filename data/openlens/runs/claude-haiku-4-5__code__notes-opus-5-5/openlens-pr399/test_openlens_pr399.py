import pytest
import math
from openlens.ray import Ray3D, RefractionResult
from openlens.vector3 import vec3
from openlens.lens import Lens
from openlens.tracer_3d import LensRayTracer3D, SystemRayTracer3D
from openlens.optical_system import OpticalSystem
from openlens import constants as C


class TestRay3DPropagate:
    """Tests for Ray3D.propagate method."""

    def test_propagate_zero_distance_is_noop(self):
        """Zero distance should be a no-op."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)
        path_len_before = len(ray.path)
        opl_before = ray.optical_path_length

        ray.propagate(0.0)

        assert len(ray.path) == path_len_before
        assert ray.optical_path_length == opl_before

    def test_propagate_positive_distance_moves_ray(self):
        """Positive distance should move ray along direction."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)
        ray.propagate(10.0)

        assert ray.origin.x == pytest.approx(10.0)
        assert ray.optical_path_length == pytest.approx(10.0)
        assert len(ray.path) == 2
        assert ray.path[-1].x == pytest.approx(ray.origin.x)

    def test_propagate_direction_normalized(self):
        """Direction should be treated as unit vector."""
        # Direction (3,4,0) normalized is (0.6, 0.8, 0); distance 5 gives (3, 4, 0)
        ray = Ray3D(vec3(0, 0, 0), vec3(3, 4, 0), n=1.0)
        ray.propagate(5.0)

        assert ray.origin.x == pytest.approx(3.0)
        assert ray.origin.y == pytest.approx(4.0)

    def test_propagate_optical_path_with_refractive_index(self):
        """Optical path length should be distance * n."""
        ray = Ray3D(vec3(0, 0, 0), vec3(0, 0, 1), n=1.5)
        ray.propagate(2.0)

        assert ray.optical_path_length == pytest.approx(3.0)

    def test_propagate_negative_distance(self):
        """Negative distance should move ray backward."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)
        ray.propagate(-4.0)

        assert ray.origin.x == pytest.approx(-4.0)
        assert ray.optical_path_length == pytest.approx(-4.0)

    def test_propagate_accumulation(self):
        """Multiple propagate calls should accumulate correctly."""
        ray = Ray3D(vec3(0, 0, 0), vec3(1, 0, 0), n=1.0)
        ray.propagate(1.0)
        ray.propagate(2.0)

        assert ray.origin.x == pytest.approx(3.0)
        assert ray.optical_path_length == pytest.approx(3.0)
        assert len(ray.path) == 3
        assert ray.path[1].x == pytest.approx(1.0)
        assert ray.path[2].x == pytest.approx(3.0)


class TestLensRayTracerTrace:
    """Tests for LensRayTracer3D.trace_surface method."""

    @pytest.fixture
    def lens_tracer(self):
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        return LensRayTracer3D(lens)

    def test_trace_surface_on_axis_front_hit(self, lens_tracer):
        """Ray should refract at front surface."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0), n=1.0)
        result = lens_tracer.trace_surface(ray, "front")

        assert result == RefractionResult.REFRACTED
        assert ray.origin.x == pytest.approx(0.0, abs=1e-6)
        assert ray.optical_path_length == pytest.approx(10.0, abs=1e-6)

    def test_trace_surface_back_hit_after_front(self, lens_tracer):
        """Back surface should be traced after front refraction."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0), n=1.0)
        lens_tracer.trace_surface(ray, "front")
        opl_at_front = ray.optical_path_length
        result = lens_tracer.trace_surface(ray, "back")

        assert result == RefractionResult.REFRACTED
        assert ray.origin.x == pytest.approx(5.0, abs=1e-6)
        assert ray.optical_path_length > opl_at_front

    def test_trace_surface_invalid_surface_unchanged(self, lens_tracer):
        """Invalid surface type returns MISSED without changing ray."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0), n=1.0)
        initial_x = ray.origin.x
        initial_opl = ray.optical_path_length

        result = lens_tracer.trace_surface(ray, "invalid")

        assert result == RefractionResult.MISSED
        assert ray.origin.x == initial_x
        assert ray.optical_path_length == initial_opl

    def test_trace_surface_aperture_miss(self, lens_tracer):
        """Ray outside aperture should be missed."""
        ray = Ray3D(vec3(-10, 13, 0), vec3(1, 0, 0), n=1.0)
        result = lens_tracer.trace_surface(ray, "front")

        assert result == RefractionResult.MISSED

    def test_trace_surface_reflect(self, lens_tracer):
        """Reflection interaction should return REFLECTED."""
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0), n=1.0)
        result = lens_tracer.trace_surface(ray, "front", interaction="reflect")

        assert result == RefractionResult.REFLECTED
        assert ray.origin.x == pytest.approx(0.0, abs=1e-6)


class TestSystemRayTracerOffAxis:
    """Tests for SystemRayTracer3D.trace_off_axis_rays method."""

    @pytest.fixture
    def system(self):
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5168,
        )
        sys = OpticalSystem()
        sys.add_lens(lens)
        return sys

    @pytest.fixture
    def tracer(self, system):
        return SystemRayTracer3D(system)

    def test_trace_off_axis_rays_returns_list_with_correct_count(self, tracer):
        """Should return list of Ray3D with correct count."""
        rays = tracer.trace_off_axis_rays(0.0, num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 5
        assert all(isinstance(r, Ray3D) for r in rays)

    def test_trace_off_axis_rays_empty_system_returns_empty(self):
        """Empty system should return empty list."""
        sys = OpticalSystem()
        tracer = SystemRayTracer3D(sys)
        rays = tracer.trace_off_axis_rays(0.0, num_rays=5)
        assert rays == []

    def test_trace_off_axis_rays_on_axis_field_z_zero(self, tracer):
        """On-axis field (0°) should keep all z coordinates zero."""
        rays = tracer.trace_off_axis_rays(0.0, num_rays=5)
        for ray in rays:
            assert abs(ray.origin.z) < 1e-9
            assert abs(ray.direction.z) < 1e-9

    def test_trace_off_axis_rays_distributed_across_diameter(self, tracer, system):
        """Rays should be distributed across the first lens diameter."""
        lens = system.elements[0].lens
        rays = tracer.trace_off_axis_rays(0.0, num_rays=5)

        y_coords = [r.origin.y for r in rays]
        # Rays should span from negative to positive
        assert min(y_coords) < 0
        assert max(y_coords) > 0

    def test_trace_off_axis_rays_wavelength_parameter(self, tracer):
        """Wavelength parameter should be passed to rays."""
        wavelength = C.WAVELENGTH_GREEN * C.NM_TO_MM
        rays = tracer.trace_off_axis_rays(0.0, num_rays=1, wavelength_mm=wavelength)
        assert len(rays) == 1
        assert rays[0].wavelength == pytest.approx(wavelength)

    def test_trace_off_axis_rays_default_num_rays(self, tracer):
        """Default num_rays should use DEFAULT_NUM_RAYS constant."""
        rays = tracer.trace_off_axis_rays(0.0)
        assert len(rays) == C.DEFAULT_NUM_RAYS

    def test_trace_off_axis_rays_zero_and_one_rays(self, tracer):
        """num_rays=0 returns empty; num_rays=1 returns single ray."""
        rays_zero = tracer.trace_off_axis_rays(0.0, num_rays=0)
        assert rays_zero == []

        rays_one = tracer.trace_off_axis_rays(0.0, num_rays=1)
        assert len(rays_one) == 1
        assert abs(rays_one[0].origin.z) < 1e-9
