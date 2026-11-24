import pytest
import math
from openlens import OpticalSystem, Lens, SystemRayTracer, Ray
from openlens.constants import WAVELENGTH_GREEN, NM_TO_MM


@pytest.fixture
def simple_system():
    """Single lens system for testing."""
    system = OpticalSystem()
    system.add_lens(Lens(diameter=25.0))
    return system


@pytest.fixture
def two_lens_system():
    """Two lens system for testing."""
    system = OpticalSystem()
    system.add_lens(Lens(diameter=25.0))
    system.add_lens(Lens(diameter=20.0))
    return system


class TestTraceParallelRaysBasic:
    """Test basic functionality and return values."""
    
    def test_empty_system_returns_empty_list(self):
        """Empty system should return empty list immediately."""
        system = OpticalSystem()
        tracer = SystemRayTracer(system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        assert rays == []
    
    def test_zero_rays_returns_empty_list(self, simple_system):
        """num_rays=0 should return empty list."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=0)
        assert rays == []
    
    def test_single_ray_returned(self, simple_system):
        """Single ray should be returned as list of length 1."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=1)
        assert len(rays) == 1
        assert isinstance(rays[0], Ray)
    
    def test_correct_number_of_rays(self, simple_system):
        """Should return exactly num_rays rays."""
        tracer = SystemRayTracer(simple_system)
        for n in [2, 5, 10, 20]:
            rays = tracer.trace_parallel_rays(num_rays=n)
            assert len(rays) == n
    
    def test_rays_are_distinct_objects(self, simple_system):
        """Each returned ray should be a distinct object."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        ids = [id(ray) for ray in rays]
        assert len(ids) == len(set(ids))


class TestRayHeights:
    """Test ray height calculation and symmetry."""
    
    def test_single_ray_height_on_axis(self, simple_system):
        """Single ray should be at height 0 for no angle."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=1, angle_deg=0.0)
        assert pytest.approx(rays[0].y, abs=1e-6) == 0.0
    
    def test_ray_heights_symmetric_centered(self, simple_system):
        """Ray heights should be symmetric for centered element with angle_deg=0."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0)
        
        n = len(rays)
        for i in range(n // 2):
            assert pytest.approx(rays[i].y, abs=1e-6) == -rays[n - 1 - i].y
    
    def test_ray_heights_monotonic_increasing(self, simple_system):
        """Ray heights should increase monotonically when fill > 0."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=10, angle_deg=0.0, fill=1.0)
        
        y_values = [ray.y for ray in rays]
        for i in range(len(y_values) - 1):
            assert y_values[i] <= y_values[i + 1]
    
    def test_fill_zero_all_rays_at_zero(self, simple_system):
        """With fill=0, all rays should be at height 0."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=0.0)
        
        for ray in rays:
            assert pytest.approx(ray.y, abs=1e-9) == 0.0


class TestWavelengthParameter:
    """Test wavelength_mm parameter."""
    
    def test_wavelength_mm_set_on_all_rays(self, simple_system):
        """wavelength_mm should be set on each ray."""
        tracer = SystemRayTracer(simple_system)
        wl = WAVELENGTH_GREEN * NM_TO_MM * 2.0
        rays = tracer.trace_parallel_rays(num_rays=3, wavelength_mm=wl)
        
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(wl)
    
    def test_wavelength_mm_default_value(self, simple_system):
        """Default wavelength should be WAVELENGTH_GREEN."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=2)
        
        default_wl = WAVELENGTH_GREEN * NM_TO_MM
        for ray in rays:
            assert ray.wavelength_mm == pytest.approx(default_wl)
    
    def test_wavelength_does_not_affect_initial_geometry(self, simple_system):
        """Different wavelengths should not affect initial ray position/angle."""
        tracer = SystemRayTracer(simple_system)
        
        rays1 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=0.5e-3)
        rays2 = tracer.trace_parallel_rays(num_rays=5, wavelength_mm=1.0e-3)
        
        for r1, r2 in zip(rays1, rays2):
            assert pytest.approx(r1.y, abs=1e-9) == r2.y
            assert pytest.approx(r1.angle, abs=1e-9) == r2.angle


class TestAngleParameter:
    """Test angle_deg parameter."""
    
    def test_angle_deg_zero_gives_vertical_rays(self, simple_system):
        """With angle_deg=0, rays should be vertical."""
        tracer = SystemRayTracer(simple_system)
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=0.0)
        
        for ray in rays:
            assert ray.angle == pytest.approx(0.0, abs=1e-9)
    
    def test_angle_deg_converted_to_radians(self, simple_system):
        """angle_deg should be converted to radians in ray.angle."""
        tracer = SystemRayTracer(simple_system)
        angle_deg = 15.0
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=angle_deg)
        
        angle_rad = math.radians(angle_deg)
        for ray in rays:
            assert ray.angle == pytest.approx(angle_rad, abs=1e-9)
    
    def test_negative_angle_deg(self, simple_system):
        """Negative angle_deg should be handled correctly."""
        tracer = SystemRayTracer(simple_system)
        angle_deg = -10.0
        rays = tracer.trace_parallel_rays(num_rays=3, angle_deg=angle_deg)
        
        angle_rad = math.radians(angle_deg)
        for ray in rays:
            assert ray.angle == pytest.approx(angle_rad, abs=1e-9)


class TestFillParameter:
    """Test fill parameter."""
    
    def test_fill_one_spans_full_diameter(self, simple_system):
        """With fill=1.0, rays should span approximately the full diameter."""
        tracer = SystemRayTracer(simple_system)
        first_lens = simple_system.elements[0].lens
        diameter = first_lens.diameter
        
        rays = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=1.0)
        
        y_values = [ray.y for ray in rays]
        span = max(y_values) - min(y_values)
        
        assert pytest.approx(span, rel=0.01) == diameter
    
    def test_fill_half_spans_half_diameter(self, simple_system):
        """With fill=0.5, rays should span approximately half the diameter."""
        tracer = SystemRayTracer(simple_system)
        
        rays_full = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=1.0)
        rays_half = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=0.5)
        
        span_full = max(r.y for r in rays_full) - min(r.y for r in rays_full)
        span_half = max(r.y for r in rays_half) - min(r.y for r in rays_half)
        
        assert pytest.approx(span_half, rel=0.01) == span_full * 0.5
    
    def test_fill_negative_reverses_height_order(self, simple_system):
        """Negative fill should reverse the order of ray heights."""
        tracer = SystemRayTracer(simple_system)
        
        rays_pos = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=1.0)
        rays_neg = tracer.trace_parallel_rays(num_rays=5, angle_deg=0.0, fill=-1.0)
        
        n = len(rays_pos)
        for i in range(n):
            assert pytest.approx(rays_neg[i].y, abs=1e-6) == -rays_pos[n - 1 - i].y


class TestMultiElementSystem:
    """Test with multi-element systems."""
    
    def test_traces_through_multiple_elements(self, two_lens_system):
        """Should trace rays through multiple optical elements."""
        tracer = SystemRayTracer(two_lens_system)
        rays = tracer.trace_parallel_rays(num_rays=5)
        
        assert len(rays) == 5
        for ray in rays:
            assert isinstance(ray, Ray)
