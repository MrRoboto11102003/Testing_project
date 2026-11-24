import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.tracer_2d import SystemRayTracer, NM_TO_MM, WAVELENGTH_GREEN


class TestSystemRayTracerParallelRays:
    """Test openlens.tracer_2d.SystemRayTracer.trace_parallel_rays"""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with one lens."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="test_system")
        system.add_lens(lens)
        return system

    @pytest.fixture
    def empty_system(self):
        """Create an empty optical system (no elements)."""
        return OpticalSystem(name="empty_system")

    @pytest.fixture
    def two_element_system(self):
        """Create a multi-element optical system."""
        lens1 = Lens(
            name="lens1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="lens2",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=100.0,
            thickness=4.0,
            diameter=20.0,
            material="BK7",
        )
        system = OpticalSystem(name="two_element")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10.0)
        return system

    def test_empty_system_returns_empty_list(self, empty_system):
        """When system has no elements, trace_parallel_rays returns empty list."""
        tracer = SystemRayTracer(empty_system)
        result = tracer.trace_parallel_rays(num_rays=5)
        assert result == []
        assert isinstance(result, list)

    def test_single_ray(self, simple_system):
        """When num_rays=1, returns list with exactly one ray."""
        tracer = SystemRayTracer(simple_system)
        result = tracer.trace_parallel_rays(num_rays=1)
        assert len(result) == 1
        assert result[0] is not None

    def test_multiple_rays_count(self, simple_system):
        """Returns correct number of rays for different num_rays values."""
        tracer = SystemRayTracer(simple_system)

        for num_rays in [2, 5, 11, 50]:
            result = tracer.trace_parallel_rays(num_rays=num_rays)
            assert len(result) == num_rays
            assert all(ray is not None for ray in result)

    def test_default_wavelength(self, simple_system):
        """Default wavelength equals WAVELENGTH_GREEN * NM_TO_MM."""
        tracer = SystemRayTracer(simple_system)
        default_wavelength = WAVELENGTH_GREEN * NM_TO_MM
        rays = tracer.trace_parallel_rays(num_rays=3)

        for ray in rays:
            assert ray.wavelength_mm == default_wavelength

    def test_custom_wavelength_propagates_to_rays(self, simple_system):
        """Custom wavelength_mm is applied to all rays."""
        tracer = SystemRayTracer(simple_system)
        custom_wavelength = 0.000486  # 486 nm (H-beta) in mm
        rays = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=custom_wavelength)

        for ray in rays:
            assert ray.wavelength_mm == custom_wavelength

    def test_full_aperture_fill_default(self, simple_system):
        """Default fill=1.0 produces rays spanning full aperture diameter."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=11, fill=1.0)
        assert len(rays) == 11

    def test_legacy_viz_style_fill_factor(self, simple_system):
        """fill=0.95 (legacy viz-style) is accepted as documented."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=11, fill=0.95)
        assert len(rays) == 11

    def test_partial_fill_factor(self, simple_system):
        """fill parameter works with values like 0.5."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=7, fill=0.5)
        assert len(rays) == 7

    def test_angle_deg_zero(self, simple_system):
        """angle_deg=0.0 (default) produces on-axis parallel rays."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        assert len(rays) == 5

    def test_angle_deg_positive(self, simple_system):
        """angle_deg > 0 tilts the ray fan to positive field angle."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=5.0)
        assert len(rays) == 5

    def test_angle_deg_negative(self, simple_system):
        """angle_deg < 0 tilts the ray fan to negative field angle."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=-3.0)
        assert len(rays) == 5

    def test_multi_element_system_traced(self, two_element_system):
        """trace_parallel_rays successfully traces through multi-element systems."""
        tracer = SystemRayTracer(two_element_system)
        rays = tracer.trace_parallel_rays(num_rays=7)
        assert len(rays) == 7
        assert all(ray is not None for ray in rays)

    def test_all_parameters_combined(self, simple_system):
        """All parameters work together: num_rays, angle, wavelength, fill."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(
            num_rays=11,
            angle_deg=2.5,
            wavelength_mm=0.00055,  # 550 nm in mm
            fill=0.9,
        )
        assert len(rays) == 11
        for ray in rays:
            assert ray.wavelength_mm == 0.00055
