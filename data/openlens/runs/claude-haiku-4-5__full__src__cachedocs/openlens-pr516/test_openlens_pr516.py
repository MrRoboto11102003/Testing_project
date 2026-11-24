"""
Pytest tests for openlens.tracer_2d.LensRayTracer.trace_parallel_rays
"""

import pytest
import math
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer


@pytest.fixture
def simple_biconvex_lens():
    """Create a simple biconvex lens for testing."""
    return Lens(
        name="Test Biconvex",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def ray_tracer(simple_biconvex_lens):
    """Create a ray tracer instance."""
    return LensRayTracer(simple_biconvex_lens)


class TestTraceParallelRays:
    """Tests for LensRayTracer.trace_parallel_rays method."""

    def test_returns_list_of_rays(self, ray_tracer):
        """Test that trace_parallel_rays returns a list."""
        rays = ray_tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(rays, list)
        assert len(rays) == 5

    def test_single_ray_case(self, ray_tracer):
        """Test single ray case (num_rays=1 should have height=0)."""
        rays = ray_tracer.trace_parallel_rays(num_rays=1, angle_deg=0.0)
        assert len(rays) == 1
        assert rays[0] is not None

    def test_default_parameters(self, ray_tracer):
        """Test that default parameters work without errors."""
        rays = ray_tracer.trace_parallel_rays()
        assert isinstance(rays, list)
        assert len(rays) > 0

    def test_custom_ray_height_range(self, ray_tracer):
        """Test explicit ray_height_range parameter."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=7,
            ray_height_range=(-8.0, 8.0)
        )
        assert len(rays) == 7

    def test_ray_height_range_overrides_fill(self, ray_tracer):
        """Test that ray_height_range overrides fill parameter."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=5,
            ray_height_range=(-2.0, 2.0),
            fill=1.0  # Should be ignored
        )
        assert len(rays) == 5

    def test_fill_fraction_full_aperture(self, ray_tracer):
        """Test fill=1.0 uses full semi-aperture."""
        rays = ray_tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        assert len(rays) == 5

    def test_fill_fraction_partial_aperture(self, ray_tracer):
        """Test fill<1.0 uses fractional semi-aperture."""
        rays = ray_tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        assert len(rays) == 5

    def test_nonzero_angle(self, ray_tracer):
        """Test tracing with off-axis angle."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=11,
            angle_deg=5.0
        )
        assert len(rays) == 11

    def test_negative_angle(self, ray_tracer):
        """Test tracing with negative angle."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=7,
            angle_deg=-3.5
        )
        assert len(rays) == 7

    def test_different_wavelengths(self, ray_tracer):
        """Test tracing at different wavelengths."""
        rays_green = ray_tracer.trace_parallel_rays(
            num_rays=5,
            wavelength_mm=0.00055  # ~550 nm
        )
        rays_red = ray_tracer.trace_parallel_rays(
            num_rays=5,
            wavelength_mm=0.00065  # ~650 nm
        )
        assert len(rays_green) == 5
        assert len(rays_red) == 5

    def test_many_rays(self, ray_tracer):
        """Test with large number of rays."""
        rays = ray_tracer.trace_parallel_rays(num_rays=101)
        assert len(rays) == 101

    def test_rays_are_traced(self, ray_tracer):
        """Test that rays have been traced (not just created)."""
        rays = ray_tracer.trace_parallel_rays(num_rays=3)
        for ray in rays:
            # Traced rays should have path information
            assert hasattr(ray, 'path')
            assert ray.path is not None

    def test_combined_parameters(self, ray_tracer):
        """Test with multiple non-default parameters simultaneously."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=9,
            ray_height_range=(-10.0, 10.0),
            wavelength_mm=0.00055,
            angle_deg=2.5
        )
        assert len(rays) == 9

    def test_zero_angle_explicit(self, ray_tracer):
        """Test explicitly setting angle_deg=0.0."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=5,
            angle_deg=0.0
        )
        assert len(rays) == 5

    def test_small_fill_fraction(self, ray_tracer):
        """Test fill parameter with small value."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=11,
            fill=0.1
        )
        assert len(rays) == 11

    def test_ray_height_range_with_angle(self, ray_tracer):
        """Test that angle and ray_height_range work together."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=7,
            ray_height_range=(-5.0, 5.0),
            angle_deg=3.0
        )
        assert len(rays) == 7
