import pytest
import math
from unittest.mock import Mock, patch
from openlens.tracer_2d import SystemRayTracer, Ray


class TestTraceParallelRays:
    """Tests for SystemRayTracer.trace_parallel_rays method."""

    @pytest.fixture
    def mock_lens(self):
        """Create a mock lens with diameter."""
        lens = Mock()
        lens.diameter = 20.0  # 20mm diameter, so semi-aperture = 10mm
        return lens

    @pytest.fixture
    def mock_element(self, mock_lens):
        """Create a mock optical element."""
        element = Mock()
        element.lens = mock_lens
        element.position = 10.0  # First element at 10mm
        return element

    @pytest.fixture
    def empty_system(self):
        """Create a system with no elements."""
        system = Mock()
        system.elements = []
        return system

    @pytest.fixture
    def system_with_elements(self, mock_element):
        """Create a system with one element."""
        system = Mock()
        system.elements = [mock_element]
        return system

    @pytest.fixture
    def tracer_empty(self, empty_system):
        """Create a tracer with empty system."""
        return SystemRayTracer(empty_system)

    @pytest.fixture
    def tracer(self, system_with_elements):
        """Create a tracer with elements, mocking internal methods."""
        tracer = SystemRayTracer(system_with_elements)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        return tracer

    def test_empty_system_returns_empty_list(self, tracer_empty):
        """Test that system with no elements returns empty list."""
        result = tracer_empty.trace_parallel_rays()
        assert result == []
        assert isinstance(result, list)

    def test_single_ray_produces_one_ray(self, tracer):
        """Test that num_rays=1 produces exactly one ray."""
        result = tracer.trace_parallel_rays(num_rays=1)
        assert len(result) == 1
        assert isinstance(result, Ray)

    def test_multiple_rays_count(self, tracer):
        """Test that num_rays parameter controls number of rays returned."""
        for num_rays_val in [1, 2, 5, 10, 20]:
            tracer._trace_ray_through_system.reset_mock()
            result = tracer.trace_parallel_rays(num_rays=num_rays_val)
            assert len(result) == num_rays_val

    def test_sync_tracers_called_before_tracing(self, tracer):
        """Test that _sync_tracers is called."""
        tracer.trace_parallel_rays(num_rays=1)
        tracer._sync_tracers.assert_called_once()

    def test_all_rays_traced_through_system(self, tracer):
        """Test that each ray is traced through the system."""
        num_rays = 4
        tracer.trace_parallel_rays(num_rays=num_rays)
        assert tracer._trace_ray_through_system.call_count == num_rays

    def test_single_ray_height_zero(self, tracer):
        """Test that single ray (num_rays=1) has height 0."""
        tracer.trace_parallel_rays(num_rays=1, angle_deg=0.0)
        ray_arg = tracer._trace_ray_through_system.call_args[0][0]
        # For single ray with angle=0, y should equal height (which is 0)
        assert ray_arg.y == 0.0

    def test_angle_converted_to_radians(self, tracer):
        """Test that angle_deg is converted to radians in rays."""
        angle_deg = 15.0
        tracer.trace_parallel_rays(num_rays=1, angle_deg=angle_deg)
        
        ray_arg = tracer._trace_ray_through_system.call_args[0][0]
        expected_angle_rad = math.radians(angle_deg)
        assert abs(ray_arg.angle - expected_angle_rad) < 1e-10

    def test_parallel_rays_maintain_same_angle(self, tracer):
        """Test that all parallel rays have identical angle."""
        angle_deg = 25.0
        tracer.trace_parallel_rays(num_rays=5, angle_deg=angle_deg)
        
        rays = [call[0][0] for call in tracer._trace_ray_through_system.call_args_list]
        expected_angle = math.radians(angle_deg)
        
        for ray in rays:
            assert abs(ray.angle - expected_angle) < 1e-10

    def test_wavelength_parameter_passed_to_rays(self, tracer):
        """Test that wavelength_mm parameter is set on rays."""
        wavelength = 0.000550  # 550nm in mm
        tracer.trace_parallel_rays(num_rays=1, wavelength_mm=wavelength)
        
        ray_arg = tracer._trace_ray_through_system.call_args[0][0]
        assert ray_arg.wavelength_mm == wavelength

    def test_fill_parameter_affects_ray_distribution(self, tracer):
        """Test that fill parameter scales the range of ray heights."""
        tracer.trace_parallel_rays(num_rays=5, fill=1.0)
        heights_full = [call[0][0].y for call in tracer._trace_ray_through_system.call_args_list]
        
        tracer._trace_ray_through_system.reset_mock()
        tracer.trace_parallel_rays(num_rays=5, fill=0.5)
        heights_half = [call[0][0].y for call in tracer._trace_ray_through_system.call_args_list]
        
        range_full = max(heights_full) - min(heights_full)
        range_half = max(heights_half) - min(heights_half)
        
        # Half fill should produce approximately half the range
        assert range_half < range_full
        assert abs(range_half - range_full / 2) < 1e-6

    def test_legacy_aperture_fill_factor_095(self, tracer):
        """Test support for legacy visualization fill factor of 0.95."""
        result = tracer.trace_parallel_rays(num_rays=3, fill=0.95)
        assert len(result) == 3
        tracer._trace_ray_through_system.assert_called()

    def test_multiple_rays_distributed_linearly(self, tracer):
        """Test that multiple rays are distributed linearly across aperture."""
        num_rays = 5
        tracer.trace_parallel_rays(num_rays=num_rays, angle_deg=0.0, fill=1.0)
        
        rays = [call[0][0] for call in tracer._trace_ray_through_system.call_args_list]
        y_positions = [ray.y for ray in rays]
        
        # With angle=0, y positions should be monotonically increasing
        assert y_positions == sorted(y_positions)
        # First should be negative, last positive (symmetric around 0)
        assert y_positions[0] < 0
        assert y_positions[-1] > 0

    def test_ray_x_coordinate_consistent(self, tracer):
        """Test that all rays start at the same x coordinate."""
        tracer.trace_parallel_rays(num_rays=3)
        
        rays = [call[0][0] for call in tracer._trace_ray_through_system.call_args_list]
        x_coords = [ray.x for ray in rays]
        
        # All rays should have same starting x coordinate
        assert len(set(x_coords)) == 1

    def test_default_parameters(self, tracer):
        """Test default parameter values."""
        tracer.trace_parallel_rays()  # Use all defaults
        
        ray_arg = tracer._trace_ray_through_system.call_args[0][0]
        # Default angle should be 0
        assert abs(ray_arg.angle - 0.0) < 1e-10
        # Default wavelength should be WAVELENGTH_GREEN * NM_TO_MM
        from openlens import WAVELENGTH_GREEN, NM_TO_MM
        assert ray_arg.wavelength_mm == WAVELENGTH_GREEN * NM_TO_MM
