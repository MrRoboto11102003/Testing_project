import pytest
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")

from openlens.analysis.beam_synthesis import GaussianBeam, BeamSynthesisPropagator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.ray_tracer import Ray3D


class TestGaussianBeamWX:
    """Tests for GaussianBeam.w_x property"""
    
    def test_w_x_basic_calculation(self):
        """w_x should calculate correctly from q for valid imaginary part"""
        wavelength = 0.00055  # 550 nm in mm
        zR = 1.0
        q = complex(0, zR)  # Planar wavefront: q = i*zR
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        # w_x = sqrt(-wavelength / (pi * (1/q).imag))
        # For q = i*zR: 1/q = -i/zR, (1/q).imag = -1/zR
        # w_x = sqrt(wavelength * zR / pi)
        expected = math.sqrt(wavelength * zR / math.pi)
        assert abs(beam.w_x - expected) < 1e-10
    
    def test_w_x_zero_when_q_purely_real(self):
        """w_x returns 0.0 when q.imag == 0 (no imaginary part)"""
        wavelength = 0.00055
        q = complex(1.0, 0)  # Purely real
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        assert beam.w_x == 0.0


class TestGaussianBeamWY:
    """Tests for GaussianBeam.w_y property"""
    
    def test_w_y_zero_when_q_purely_real(self):
        """w_y returns 0.0 when q.imag == 0"""
        wavelength = 0.00055
        q = complex(1.0, 0)
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        assert beam.w_y == 0.0
    
    def test_w_y_equals_w_x_for_symmetric_q(self):
        """w_y should equal w_x when q_x == q_y"""
        wavelength = 0.00055
        q = complex(0.5, 1.5)
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        assert abs(beam.w_y - beam.w_x) < 1e-10


class TestGaussianBeamRX:
    """Tests for GaussianBeam.R_x property"""
    
    def test_R_x_reciprocal_of_inv_q_real(self):
        """R_x should equal 1 / (1/q).real"""
        wavelength = 0.00055
        # Create q such that 1/q has real part = 2.0
        # 1/q = 2 + 1j => q = 1/(2+1j) = (2-1j)/5 = 0.4 - 0.2j
        q = complex(0.4, -0.2)
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        inv_q = 1.0 / q
        expected_R_x = 1.0 / inv_q.real
        assert abs(beam.R_x - expected_R_x) < 1e-10
    
    def test_R_x_infinity_when_inv_q_real_zero(self):
        """R_x returns inf when (1/q).real == 0 (purely imaginary 1/q)"""
        wavelength = 0.00055
        # Planar wavefront: q = i*zR => 1/q = -i/zR (purely imaginary)
        q = complex(0, 1.0)
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q, q, ray)
        
        assert beam.R_x == float('inf')


class TestGaussianBeamAdvanceQ:
    """Tests for GaussianBeam.advance_q method"""
    
    def test_advance_q_scales_by_n_in_glass(self):
        """In glass (n=1.5), q advances by distance/n, not distance"""
        wavelength = 0.00055
        q_initial = complex(0, 1.0)
        distance = 0.6  # Arbitrary distance
        n_glass = 1.5
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q_initial, q_initial, ray)
        beam.n = n_glass
        
        q_x_before = beam.q_x
        beam.advance_q(distance)
        
        # Should advance by distance/n, not distance
        expected_step = distance / n_glass
        assert abs(beam.q_x - (q_x_before + expected_step)) < 1e-10
    
    def test_advance_q_uses_exact_distance_in_air(self):
        """In air (n=1), q advances by exactly distance"""
        wavelength = 0.00055
        q_initial = complex(0, 1.0)
        distance = 0.5
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q_initial, q_initial, ray)
        beam.n = 1.0
        
        q_x_before = beam.q_x
        q_y_before = beam.q_y
        beam.advance_q(distance)
        
        assert abs(beam.q_x - (q_x_before + distance)) < 1e-10
        assert abs(beam.q_y - (q_y_before + distance)) < 1e-10
    
    def test_advance_q_with_zero_n_uses_exact_distance(self):
        """With n=0, should use exact distance (avoid division by zero)"""
        wavelength = 0.00055
        q_initial = complex(0, 1.0)
        distance = 0.5
        
        ray = Ray3D(origin=(0, 0, 0), direction=(1, 0, 0), wavelength=wavelength)
        beam = GaussianBeam(wavelength, q_initial, q_initial, ray)
        beam.n = 0
        
        q_x_before = beam.q_x
        beam.advance_q(distance)
        
        # With n=0, should use exact distance (not divide by zero)
        assert abs(beam.q_x - (q_x_before + distance)) < 1e-10


class TestBeamSynthesisPropagatorPropagateToImage:
    """Tests for BeamSynthesisPropagator.propagate_to_image method"""
    
    def test_propagate_empty_system_returns_empty(self):
        """Empty system returns empty arrays"""
        system = OpticalSystem(name="Empty")
        propagator = BeamSynthesisPropagator(system)
        
        dy, dz, intensity = propagator.propagate_to_image()
        
        assert len(dy) == 0
        assert len(dz) == 0
        assert len(intensity) == 0
    
    def test_propagate_returns_numpy_arrays(self):
        """Should return three numpy arrays"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Simple")
        system.add_lens(lens)
        
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image(grid_size=8)
        
        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)
    
    def test_propagate_output_shape_is_grid_product(self):
        """Output arrays should be (detector_pixels, detector_pixels)"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Simple")
        system.add_lens(lens)
        
        propagator = BeamSynthesisPropagator(system)
        detector_pixels = 32
        dy, dz, intensity = propagator.propagate_to_image(
            grid_size=16,
            detector_pixels=detector_pixels
        )
        
        assert dy.shape == (detector_pixels, detector_pixels)
        assert dz.shape == (detector_pixels, detector_pixels)
        assert intensity.shape == (detector_pixels, detector_pixels)
    
    def test_propagate_intensity_all_non_negative(self):
        """Intensity must be non-negative (physical constraint)"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Simple")
        system.add_lens(lens)
        
        propagator = BeamSynthesisPropagator(system)
        _, _, intensity = propagator.propagate_to_image(grid_size=8)
        
        if intensity.size > 0:
            assert np.all(intensity >= 0)
    
    def test_propagate_detector_size_affects_grid_range(self):
        """Detector coordinates should scale with detector_size"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Simple")
        system.add_lens(lens)
        propagator = BeamSynthesisPropagator(system)
        
        dy_small, _, _ = propagator.propagate_to_image(
            grid_size=8, detector_size=0.05, detector_pixels=16
        )
        dy_large, _, _ = propagator.propagate_to_image(
            grid_size=8, detector_size=0.10, detector_pixels=16
        )
        
        if dy_small.size > 0 and dy_large.size > 0:
            # Larger detector should span larger coordinate range
            assert np.max(np.abs(dy_large)) > np.max(np.abs(dy_small))
    
    def test_propagate_different_grid_sizes_produce_different_results(self):
        """Changing grid_size (beamlet count) should affect intensity"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Simple")
        system.add_lens(lens)
        propagator = BeamSynthesisPropagator(system)
        
        _, _, intensity_8 = propagator.propagate_to_image(grid_size=8)
        _, _, intensity_16 = propagator.propagate_to_image(grid_size=16)
        
        # Different beamlet grids should produce different intensity distributions
        if intensity_8.size > 0 and intensity_16.size > 0:
            # Resample to same size for comparison
            assert not np.allclose(intensity_8, intensity_16)
