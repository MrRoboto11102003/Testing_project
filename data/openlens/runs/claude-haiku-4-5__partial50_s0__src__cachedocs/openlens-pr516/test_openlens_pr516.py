import pytest
from openlens.lens import Lens
from openlens.tracer_2d import LensRayTracer


class TestTraceParallelRays:
    """Test suite for LensRayTracer.trace_parallel_rays method."""
    
    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Test Biconvex Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
    
    @pytest.fixture
    def ray_tracer(self, simple_lens):
        """Create a ray tracer for the test lens."""
        return LensRayTracer(simple_lens)
    
    def test_returns_list_of_rays(self, ray_tracer):
        """Test that trace_parallel_rays returns a list."""
        result = ray_tracer.trace_parallel_rays(num_rays=5)
        assert isinstance(result, list)
        assert len(result) == 5
    
    def test_num_rays_single_ray(self, ray_tracer):
        """Test edge case with single ray (num_rays=1)."""
        rays = ray_tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        assert rays[0] is not None
    
    def test_num_rays_parameter_respected(self, ray_tracer):
        """Test that num_rays parameter controls the number of output rays."""
        test_values = [1, 2, 5, 11, 20, 100]
        for num_rays in test_values:
            rays = ray_tracer.trace_parallel_rays(num_rays=num_rays)
            assert len(rays) == num_rays
    
    def test_default_fill_factor_full_aperture(self, ray_tracer):
        """Test that default fill=1.0 uses full semi-aperture."""
        rays = ray_tracer.trace_parallel_rays(num_rays=10, fill=1.0)
        assert len(rays) == 10
        assert all(ray is not None for ray in rays)
    
    def test_reduced_fill_factor_visualization(self, ray_tracer):
        """Test reduced fill factor (0.95) for visualization purposes."""
        rays = ray_tracer.trace_parallel_rays(num_rays=10, fill=0.95)
        assert len(rays) == 10
        assert all(ray is not None for ray in rays)
    
    def test_very_small_fill_factor(self, ray_tracer):
        """Test tracing with very small fill factor."""
        rays = ray_tracer.trace_parallel_rays(num_rays=5, fill=0.1)
        assert len(rays) == 5
    
    def test_custom_ray_height_range(self, ray_tracer):
        """Test explicit ray_height_range parameter."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=7,
            ray_height_range=(-5.0, 5.0)
        )
        assert len(rays) == 7
    
    def test_ray_height_range_overrides_fill(self, ray_tracer):
        """Test that ray_height_range takes precedence over fill factor."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=6,
            ray_height_range=(-3.0, 3.0),
            fill=1.0
        )
        assert len(rays) == 6
    
    def test_zero_degree_angle_on_axis(self, ray_tracer):
        """Test on-axis tracing with angle_deg=0."""
        rays = ray_tracer.trace_parallel_rays(num_rays=8, angle_deg=0.0)
        assert len(rays) == 8
    
    def test_positive_angle_off_axis(self, ray_tracer):
        """Test off-axis tracing with positive angle."""
        rays = ray_tracer.trace_parallel_rays(num_rays=8, angle_deg=5.0)
        assert len(rays) == 8
    
    def test_negative_angle_off_axis(self, ray_tracer):
        """Test off-axis tracing with negative angle."""
        rays = ray_tracer.trace_parallel_rays(num_rays=8, angle_deg=-5.0)
        assert len(rays) == 8
    
    def test_different_wavelengths(self, ray_tracer):
        """Test that different wavelengths can be specified."""
        wavelength_blue = 0.00045  # ~450 nm
        wavelength_green = 0.00055  # ~550 nm
        wavelength_red = 0.00070   # ~700 nm
        
        rays_blue = ray_tracer.trace_parallel_rays(
            num_rays=4,
            wavelength_mm=wavelength_blue
        )
        rays_green = ray_tracer.trace_parallel_rays(
            num_rays=4,
            wavelength_mm=wavelength_green
        )
        rays_red = ray_tracer.trace_parallel_rays(
            num_rays=4,
            wavelength_mm=wavelength_red
        )
        
        assert len(rays_blue) == 4
        assert len(rays_green) == 4
        assert len(rays_red) == 4
    
    def test_combined_parameters(self, ray_tracer):
        """Test combination of multiple parameters together."""
        rays = ray_tracer.trace_parallel_rays(
            num_rays=12,
            ray_height_range=(-8.0, 8.0),
            wavelength_mm=0.000587,
            angle_deg=2.5,
            fill=0.8
        )
        assert len(rays) == 12
        assert all(ray is not None for ray in rays)
    
    def test_large_number_of_rays(self, ray_tracer):
        """Test tracing a large number of rays."""
        rays = ray_tracer.trace_parallel_rays(num_rays=500)
        assert len(rays) == 500
