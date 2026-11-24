import pytest
import math
import numpy as np
from unittest.mock import Mock

from openlens.analysis.beam_synthesis import GaussianBeam, BeamSynthesisPropagator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestGaussianBeamRx:
    """Test GaussianBeam.R_x property (radius of curvature in X plane)."""
    
    def test_rx_infinity_for_planar_wavefront(self):
        """R_x returns inf when wavefront is planar (pure imaginary q_x)."""
        wavelength = 0.00055  # 550 nm in mm
        q_x = complex(0, 100)  # Pure imaginary = planar wavefront
        q_y = complex(0, 100)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x, q_y, ray, n=1.0)
        assert beam.R_x == float("inf")
    
    def test_rx_finite_for_converging_wavefront(self):
        """R_x is finite and positive for converging wavefront."""
        wavelength = 0.00055
        q_x = complex(50, 100)  # Positive real part = converging
        q_y = complex(50, 100)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x, q_y, ray, n=1.0)
        expected = 1.0 / (1.0 / q_x).real
        assert abs(beam.R_x - expected) < 1e-10


class TestGaussianBeamWx:
    """Test GaussianBeam.w_x property (beam radius in X plane)."""
    
    def test_wx_zero_for_real_q(self):
        """w_x is 0 when q_x is purely real (no imaginary component)."""
        wavelength = 0.00055
        q_x = complex(50, 0)
        q_y = complex(50, 0)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x, q_y, ray, n=1.0)
        assert beam.w_x == 0.0
    
    def test_wx_scales_with_wavelength(self):
        """w_x increases with longer wavelength (for same q_x)."""
        q_x = complex(0, 50)
        q_y = complex(0, 50)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam_short = GaussianBeam(0.0004, q_x, q_y, ray, n=1.0)  # 400 nm
        beam_long = GaussianBeam(0.0007, q_x, q_y, ray, n=1.0)   # 700 nm
        
        assert beam_long.w_x > beam_short.w_x


class TestGaussianBeamWy:
    """Test GaussianBeam.w_y property (beam radius in Y plane)."""
    
    def test_wy_zero_for_real_q(self):
        """w_y is 0 when q_y is purely real."""
        wavelength = 0.00055
        q_x = complex(0, 50)
        q_y = complex(50, 0)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x, q_y, ray, n=1.0)
        assert beam.w_y == 0.0
    
    def test_wy_larger_for_larger_imaginary_q(self):
        """w_y increases with larger imaginary part of q_y."""
        wavelength = 0.00055
        q_x = complex(0, 50)
        q_y_small = complex(0, 50)
        q_y_large = complex(0, 150)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam_small = GaussianBeam(wavelength, q_x, q_y_small, ray, n=1.0)
        beam_large = GaussianBeam(wavelength, q_x, q_y_large, ray, n=1.0)
        
        assert beam_large.w_y > beam_small.w_y


class TestGaussianBeamAdvanceQ:
    """Test GaussianBeam.advance_q method."""
    
    def test_advance_q_divides_by_refractive_index(self):
        """advance_q divides distance by n when inside glass medium."""
        wavelength = 0.00055
        q_x_init = complex(10, 100)
        q_y_init = complex(10, 100)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x_init, q_y_init, ray, n=1.5)
        beam.advance_q(10.0)
        
        expected = q_x_init + 10.0 / 1.5
        assert beam.q_x == expected
        assert beam.q_y == expected
    
    def test_advance_q_uses_full_distance_when_n_is_zero(self):
        """advance_q uses full distance when n is 0 (docstring: d/n when n exists)."""
        wavelength = 0.00055
        q_x_init = complex(10, 100)
        q_y_init = complex(10, 100)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x_init, q_y_init, ray, n=0)
        beam.advance_q(10.0)
        
        expected = q_x_init + 10.0
        assert beam.q_x == expected
        assert beam.q_y == expected
    
    def test_advance_q_updates_both_axes_equally(self):
        """advance_q updates both q_x and q_y by the same amount."""
        wavelength = 0.00055
        q_x_init = complex(5, 50)
        q_y_init = complex(8, 80)
        ray = Mock(terminated=False, optical_path_length=0)
        
        beam = GaussianBeam(wavelength, q_x_init, q_y_init, ray, n=2.0)
        beam.advance_q(6.0)
        
        step = 6.0 / 2.0
        assert beam.q_x == q_x_init + step
        assert beam.q_y == q_y_init + step


class TestBeamSynthesisPropagatorPropagateToImage:
    """Test BeamSynthesisPropagator.propagate_to_image method."""
    
    def test_propagate_to_image_empty_system_returns_empty_arrays(self):
        """Should return empty arrays when system has no elements."""
        system = Mock()
        system.elements = []
        
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image()
        
        assert len(dy) == 0
        assert len(dz) == 0
        assert len(intensity) == 0
    
    def test_propagate_to_image_returns_correct_array_shapes(self):
        """Should return arrays with shape matching detector_pixels."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(lens)
        
        propagator = BeamSynthesisPropagator(system)
        detector_pixels = 64
        dy, dz, intensity = propagator.propagate_to_image(detector_pixels=detector_pixels)
        
        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)
        assert dy.shape == (detector_pixels, detector_pixels)
        assert dz.shape == (detector_pixels, detector_pixels)
        assert intensity.shape == (detector_pixels, detector_pixels)
    
    def test_propagate_to_image_intensity_nonnegative(self):
        """Intensity should be non-negative (represents power/irradiance)."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(lens)
        
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image()
        
        assert np.all(intensity >= 0)
    
    def test_propagate_to_image_respects_detector_parameters(self):
        """Should respect wavelength, grid_size, detector_size and detector_pixels."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(lens)
        
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image(
            wavelength_mm=0.00055,
            grid_size=16,
            detector_size=0.2,
            detector_pixels=32
        )
        
        assert intensity.shape == (32, 32)
        assert isinstance(intensity, np.ndarray)
    
    def test_propagate_to_image_with_custom_image_plane(self):
        """Should handle custom image_plane_x parameter."""
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem()
        system.add_element(lens)
        
        propagator = BeamSynthesisPropagator(system)
        dy, dz, intensity = propagator.propagate_to_image(image_plane_x=50.0)
        
        assert isinstance(intensity, np.ndarray)
        assert intensity.size > 0
