import pytest
from openlens.tracer_2d import LensRayTracer
from openlens.ray import Ray


@pytest.fixture
def tracer():
    """Create a basic lens tracer for testing."""
    tracer = LensRayTracer(D=10.0)
    return tracer


class TestTraceParallelRays:
    """Focused tests for LensRayTracer.trace_parallel_rays method."""

    def test_returns_list_of_rays(self, tracer):
        """Test that method returns a list of Ray objects."""
        result = tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(r, Ray) for r in result)

    def test_num_rays_parameter(self, tracer):
        """Test that num_rays parameter controls the number of rays."""
        for num in [1, 2, 3, 10]:
            rays = tracer.trace_parallel_rays(num_rays=num)
            assert len(rays) == num

    def test_single_ray_centered_case(self, tracer):
        """Test special handling when num_rays=1 produces centered ray."""
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1

    def test_default_full_aperture_fill(self, tracer):
        """Test that default fill=1.0 uses full semi-aperture."""
        rays = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        assert len(rays) == 5

    def test_fill_parameter_reduces_aperture(self, tracer):
        """Test that fill<1.0 reduces aperture span."""
        rays_full = tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        assert len(rays_full) == 5
        assert len(rays_half) == 5

    def test_legacy_visualization_fill_factor(self, tracer):
        """Test legacy fill factor value 0.95 for display fans."""
        rays = tracer.trace_parallel_rays(num_rays=5, fill=0.95)
        assert len(rays) == 5

    def test_explicit_ray_height_range(self, tracer):
        """Test explicit ray_height_range parameter."""
        rays = tracer.trace_parallel_rays(
            num_rays=5,
            ray_height_range=(-2.0, 2.0)
        )
        assert len(rays) == 5

    def test_ray_height_range_overrides_fill(self, tracer):
        """Test that ray_height_range takes precedence over fill."""
        rays = tracer.trace_parallel_rays(
            num_rays=3,
            ray_height_range=(-1.5, 1.5),
            fill=0.5
        )
        assert len(rays) == 3

    def test_wavelength_parameter(self, tracer):
        """Test wavelength_mm parameter is accepted."""
        wavelength_mm = 0.00055  # 550 nm in mm
        rays = tracer.trace_parallel_rays(
            num_rays=3,
            wavelength_mm=wavelength_mm
        )
        assert len(rays) == 3

    def test_angle_zero_degrees_on_axis(self, tracer):
        """Test on-axis collimated beam with angle_deg=0.0."""
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=0.0)
        assert len(rays) == 3

    def test_angle_nonzero_degrees_tilted_beam(self, tracer):
        """Test tilted collimated beam with non-zero angle_deg."""
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=5.0)
        assert len(rays) == 3

    def test_asymmetric_ray_height_range(self, tracer):
        """Test asymmetric ray_height_range."""
        rays = tracer.trace_parallel_rays(
            num_rays=4,
            ray_height_range=(-1.0, 3.0)
        )
        assert len(rays) == 4

    def test_large_number_of_rays(self, tracer):
        """Test with large number of rays for dense sampling."""
        rays = tracer.trace_parallel_rays(num_rays=50)
        assert len(rays) == 50

    def test_combined_parameters(self, tracer):
        """Test with multiple parameters simultaneously."""
        rays = tracer.trace_parallel_rays(
            num_rays=7,
            ray_height_range=(-3.0, 3.0),
            wavelength_mm=0.00055,
            angle_deg=3.0,
            fill=0.95
        )
        assert len(rays) == 7

    def test_default_parameters(self, tracer):
        """Test method with all default parameters."""
        rays = tracer.trace_parallel_rays()
        assert isinstance(rays, list)
        assert len(rays) > 0
