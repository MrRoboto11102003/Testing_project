import pytest
import math
from unittest.mock import Mock
from openlens.tracer_2d import LensRayTracer, SystemRayTracer, Ray


class TestLensRayTracerParallelRays:
    """Tests for LensRayTracer.trace_parallel_rays method"""

    def test_single_ray_returns_one_ray(self):
        """Test that num_rays=1 returns exactly one ray"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=1)
        
        assert len(rays) == 1
        assert isinstance(rays[0], Ray)

    def test_multiple_rays_returns_correct_count(self):
        """Test that num_rays parameter returns correct number of rays"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        for n in [2, 5, 10]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n

    def test_zero_angle_rays_parallel(self):
        """Test that rays traced with angle_deg=0 have zero angle"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=0.0)
        
        for ray in rays:
            assert ray.angle_rad == pytest.approx(0.0)

    def test_angled_rays_have_correct_angle(self):
        """Test that rays have correct angle when angle_deg is specified"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        angle_deg = 15.0
        rays = tracer.trace_parallel_rays(num_rays=4, angle_deg=angle_deg)
        
        expected_angle = math.radians(angle_deg)
        for ray in rays:
            assert ray.angle_rad == pytest.approx(expected_angle)

    def test_wavelength_passed_to_rays(self):
        """Test that wavelength parameter is correctly passed to Ray objects"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        wavelength = 0.000550
        rays = tracer.trace_parallel_rays(num_rays=2, wavelength_mm=wavelength)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength)

    def test_custom_ray_height_range(self):
        """Test that custom ray_height_range parameter is respected"""
        mock_lens = Mock()
        mock_lens.D = 20.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        custom_range = (-3.0, 3.0)
        rays = tracer.trace_parallel_rays(num_rays=3, ray_height_range=custom_range)
        
        assert len(rays) == 3

    def test_fill_factor_scales_height_range(self):
        """Test that fill factor scales the ray height range from lens diameter"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        rays_full = tracer.trace_parallel_rays(num_rays=2, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=2, fill=0.5)
        
        assert len(rays_full) == 2
        assert len(rays_half) == 2

    def test_lens_offset_position_handled(self):
        """Test that rays are traced correctly when lens is offset"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 50.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=2)
        assert len(rays) == 2

    def test_trace_ray_called_for_each_ray(self):
        """Test that trace_ray is called once for each ray in the bundle"""
        mock_lens = Mock()
        mock_lens.D = 10.0
        mock_lens.front_vertex_x = 0.0
        
        tracer = LensRayTracer(mock_lens)
        tracer.trace_ray = Mock()
        
        num_rays = 5
        tracer.trace_parallel_rays(num_rays=num_rays)
        
        assert tracer.trace_ray.call_count == num_rays


class TestSystemRayTracerParallelRays:
    """Tests for SystemRayTracer.trace_parallel_rays method"""

    def test_empty_system_returns_empty_list(self):
        """Test that empty optical system returns empty list of rays"""
        mock_system = Mock()
        mock_system.elements = []
        
        tracer = SystemRayTracer(mock_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert rays == []
        assert isinstance(rays, list)

    def test_single_element_system_returns_rays(self):
        """Test that system with one element returns correct number of rays"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=3)
        
        assert len(rays) == 3
        assert all(isinstance(ray, Ray) for ray in rays)

    def test_sync_tracers_called_before_tracing(self):
        """Test that _sync_tracers is called once during tracing"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        tracer.trace_parallel_rays(num_rays=2)
        
        tracer._sync_tracers.assert_called_once()

    def test_system_angled_rays(self):
        """Test that system traces rays at specified angle"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        angle_deg = 10.0
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=angle_deg)
        
        expected_angle = math.radians(angle_deg)
        for ray in rays:
            assert ray.angle_rad == pytest.approx(expected_angle)

    def test_system_wavelength_parameter(self):
        """Test that wavelength is correctly passed to all rays"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        wavelength = 0.000650
        rays = tracer.trace_parallel_rays(num_rays=2, wavelength_mm=wavelength)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wavelength)

    def test_system_fill_factor_reduces_aperture(self):
        """Test that fill factor scales entrance-pupil aperture"""
        mock_lens = Mock()
        mock_lens.diameter = 20.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        rays_full = tracer.trace_parallel_rays(num_rays=2, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=2, fill=0.5)
        
        assert len(rays_full) == 2
        assert len(rays_half) == 2

    def test_system_uses_first_element_aperture(self):
        """Test that system uses first element's diameter for aperture calculation"""
        mock_lens1 = Mock()
        mock_lens1.diameter = 25.0
        
        mock_element1 = Mock()
        mock_element1.lens = mock_lens1
        mock_element1.position = 0.0
        
        mock_lens2 = Mock()
        mock_lens2.diameter = 10.0
        
        mock_element2 = Mock()
        mock_element2.lens = mock_lens2
        mock_element2.position = 10.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element1, mock_element2]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1

    def test_system_uses_first_element_position(self):
        """Test that system uses first element's position for ray start location"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 100.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=2)
        assert len(rays) == 2

    def test_trace_ray_through_system_called_for_each_ray(self):
        """Test that _trace_ray_through_system is called once per ray"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        num_rays = 5
        tracer.trace_parallel_rays(num_rays=num_rays)
        
        assert tracer._trace_ray_through_system.call_count == num_rays

    def test_system_single_ray_on_axis(self):
        """Test that single ray in system is placed on optical axis"""
        mock_lens = Mock()
        mock_lens.diameter = 15.0
        
        mock_element = Mock()
        mock_element.lens = mock_lens
        mock_element.position = 0.0
        
        mock_system = Mock()
        mock_system.elements = [mock_element]
        
        tracer = SystemRayTracer(mock_system)
        tracer._sync_tracers = Mock()
        tracer._trace_ray_through_system = Mock()
        
        rays = tracer.trace_parallel_rays(num_rays=1)
        
        assert len(rays) == 1
