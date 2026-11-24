"""
Pytest unit tests for openlens ray tracing functions:
- Ray3D.propagate
- LensRayTracer3D.trace_surface
- SystemRayTracer3D.trace_off_axis_rays
"""

import pytest
import math
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem, Element
from openlens.ray import Ray3D, RefractionResult
from openlens.tracer_3d import LensRayTracer3D, SystemRayTracer3D


class TestRay3DPropagate:
    """Tests for Ray3D.propagate method"""

    def test_propagate_zero_distance_is_noop(self):
        """Zero distance should not modify ray state."""
        ray = Ray3D(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), n=1.0)
        initial_origin = tuple(ray.origin)
        initial_path_len = len(ray.path)
        initial_opl = ray.optical_path_length

        ray.propagate(0.0)

        assert tuple(ray.origin) == initial_origin
        assert len(ray.path) == initial_path_len
        assert ray.optical_path_length == initial_opl

    def test_propagate_tiny_distance_is_noop(self):
        """Distance below EPSILON should be a no-op."""
        ray = Ray3D(origin=(5.0, 5.0, 5.0), direction=(1.0, 0.0, 0.0), n=1.0)
        initial_origin = tuple(ray.origin)
        initial_opl = ray.optical_path_length

        ray.propagate(1e-10)  # Negligible distance

        assert tuple(ray.origin) == initial_origin
        assert ray.optical_path_length == initial_opl

    def test_propagate_positive_distance_updates_origin(self):
        """Positive distance should update origin along direction."""
        ray = Ray3D(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), n=1.0)

        ray.propagate(5.0)

        assert ray.origin[0] == pytest.approx(5.0, abs=0.001)
        assert ray.origin[1] == pytest.approx(0.0, abs=0.001)
        assert ray.origin[2] == pytest.approx(0.0, abs=0.001)

    def test_propagate_updates_optical_path_length(self):
        """Propagation should increment OPL by distance * n."""
        n_glass = 1.5
        ray = Ray3D(
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            n=n_glass
        )
        initial_opl = ray.optical_path_length

        distance = 10.0
        ray.propagate(distance)

        expected_opl = initial_opl + distance * n_glass
        assert ray.optical_path_length == pytest.approx(expected_opl, abs=0.001)

    def test_propagate_appends_to_path(self):
        """Propagation should append new position to path."""
        ray = Ray3D(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), n=1.0)
        initial_len = len(ray.path)

        ray.propagate(5.0)

        assert len(ray.path) > initial_len
        # Last path point should be new origin
        assert tuple(ray.path[-1]) == pytest.approx(tuple(ray.origin), abs=0.001)

    def test_propagate_direction_unchanged(self):
        """Direction should remain constant during propagation."""
        direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        ray = Ray3D(origin=(0.0, 0.0, 0.0), direction=direction, n=1.0)

        ray.propagate(10.0)

        assert np.allclose(ray.direction, direction, atol=1e-6)

    def test_propagate_negative_distance_moves_backward(self):
        """Negative distance should propagate ray backward."""
        ray = Ray3D(origin=(10.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), n=1.0)

        ray.propagate(-5.0)

        assert ray.origin[0] == pytest.approx(5.0, abs=0.001)


class TestLensRayTracer3DTraceSurface:
    """Tests for LensRayTracer3D.trace_surface method"""

    @pytest.fixture
    def biconvex_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="test_biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )

    def test_trace_surface_invalid_surface_type_returns_missed(self, biconvex_lens):
        """Invalid surface type should return MISSED."""
        tracer = LensRayTracer3D(biconvex_lens)
        ray = Ray3D(origin=(-10.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

        result = tracer.trace_surface(ray, "invalid_surface", "refract")

        assert result == RefractionResult.MISSED

    def test_trace_surface_front_surface_valid(self, biconvex_lens):
        """Front surface should be traceable."""
        tracer = LensRayTracer3D(biconvex_lens)
        ray = Ray3D(origin=(-10.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

        result = tracer.trace_surface(ray, "front", "refract")

        assert isinstance(result, RefractionResult)

    def test_trace_surface_back_surface_valid(self, biconvex_lens):
        """Back surface should be traceable."""
        tracer = LensRayTracer3D(biconvex_lens)
        ray = Ray3D(origin=(-10.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

        result = tracer.trace_surface(ray, "back", "refract")

        assert isinstance(result, RefractionResult)

    def test_trace_surface_refract_interaction(self, biconvex_lens):
        """Refract interaction should be handled."""
        tracer = LensRayTracer3D(biconvex_lens)
        ray = Ray3D(origin=(-15.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

        result = tracer.trace_surface(ray, "front", "refract")

        # Result must be a valid RefractionResult enum value
        assert result in [
            RefractionResult.MISSED,
            RefractionResult.REFRACTED,
            RefractionResult.REFLECTED,
        ]

    def test_trace_surface_reflect_interaction(self, biconvex_lens):
        """Reflect interaction should be handled."""
        tracer = LensRayTracer3D(biconvex_lens)
        ray = Ray3D(origin=(-15.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

        result = tracer.trace_surface(ray, "front", "reflect")

        assert result in [
            RefractionResult.MISSED,
            RefractionResult.REFLECTED,
        ]


class TestSystemRayTracer3DTraceOffAxisRays:
    """Tests for SystemRayTracer3D.trace_off_axis_rays method"""

    def test_empty_system_returns_empty_list(self):
        """Empty optical system should return empty ray list."""
        system = OpticalSystem()
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(
            field_angle_deg=5.0,
            num_rays=5
        )

        assert rays == []
        assert len(rays) == 0

    def test_returns_correct_number_of_rays(self):
        """Should return exactly the requested number of rays."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        for num_rays in [1, 5, 11, 21]:
            rays = tracer.trace_off_axis_rays(
                field_angle_deg=0.0,
                num_rays=num_rays
            )
            assert len(rays) == num_rays

    def test_all_returned_objects_are_ray3d(self):
        """All returned objects should be Ray3D instances."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=5
        )

        assert all(isinstance(r, Ray3D) for r in rays)

    def test_field_angle_zero_produces_on_axis_rays(self):
        """Zero field angle should produce on-axis rays."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=5
        )

        # At zero field angle, all rays should have similar direction
        # (approximately along optical axis)
        dirs = [r.direction for r in rays]
        assert all(d[0] > 0 for d in dirs)  # All propagating forward

    def test_nonzero_field_angle_tilts_rays(self):
        """Non-zero field angle should tilt ray bundle."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        rays_zero = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=3
        )
        rays_tilted = tracer.trace_off_axis_rays(
            field_angle_deg=10.0,
            num_rays=3
        )

        # At least one ray direction should differ
        dirs_differ = any(
            not np.allclose(r1.direction, r2.direction, atol=1e-3)
            for r1, r2 in zip(rays_zero, rays_tilted)
        )
        assert dirs_differ

    def test_wavelength_parameter_accepted(self):
        """Should accept custom wavelength parameter."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        # Should not raise with different wavelengths
        rays_red = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=3,
            wavelength_mm=0.00070
        )
        rays_blue = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=3,
            wavelength_mm=0.00045
        )

        assert len(rays_red) == 3
        assert len(rays_blue) == 3

    def test_single_ray_case(self):
        """Single ray (num_rays=1) should work correctly."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(
            field_angle_deg=5.0,
            num_rays=1
        )

        assert len(rays) == 1
        assert isinstance(rays[0], Ray3D)

    def test_rays_distributed_across_aperture(self):
        """Multiple rays should be distributed across lens aperture."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(Element(lens, position=0.0))
        tracer = SystemRayTracer3D(system)

        rays = tracer.trace_off_axis_rays(
            field_angle_deg=0.0,
            num_rays=11
        )

        # Ray origins should span the pupil (different y-coordinates)
        y_coords = [r.origin[1] for r in rays]
        y_range = max(y_coords) - min(y_coords)
        
        assert y_range > 0  # Rays should not all be on-axis
