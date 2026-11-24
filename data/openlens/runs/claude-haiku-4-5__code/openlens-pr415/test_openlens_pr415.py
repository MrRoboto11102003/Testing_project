import pytest
import math
import numpy as np
from unittest.mock import Mock

try:
    from openlens.analysis.beam_synthesis import GaussianBeam, BeamSynthesisPropagator
    OPENLENS_AVAILABLE = True
except ImportError:
    OPENLENS_AVAILABLE = False


@pytest.mark.skipif(not OPENLENS_AVAILABLE, reason="openlens not installed")
class TestGaussianBeamRx:
    """Test GaussianBeam.R_x property (radius of curvature in X plane)"""

    def test_rx_finite_value(self):
        """Test R_x with normal non-planar wavefront"""
        wavelength = 0.5e-3
        R = 100.0
        zR = 10.0
        q_x = complex(R, zR)

        beam = GaussianBeam(wavelength, q_x, q_x, Mock())

        # For q = R + i*zR: R_x = 1 / Re(1/q) = (R^2 + zR^2) / R
        inv_q_real = R / (R**2 + zR**2)
        expected_rx = 1.0 / inv_q_real

        assert abs(beam.R_x - expected_rx) < 1e-6

    def test_rx_returns_infinity_for_planar_wavefront(self):
        """Test R_x returns infinity when wavefront is planar (Re(1/q)=0)"""
        wavelength = 0.5e-3
        zR = 10.0
        q_x = complex(0, zR)

        beam = GaussianBeam(wavelength, q_x, q_x, Mock())

        assert beam.R_x == float("inf")


@pytest.mark.skipif(not OPENLENS_AVAILABLE, reason="openlens not installed")
class TestGaussianBeamWaist:
    """Test GaussianBeam.w_x and w_y properties (beam radius 1/e^2 intensity)"""

    def test_wx_calculation(self):
        """Test w_x is computed correctly from q_x"""
        wavelength = 0.5e-3
        R = 50.0
        zR = 25.0
        q_x = complex(R, zR)

        beam = GaussianBeam(wavelength, q_x, q_x, Mock())

        inv_q = 1.0 / q_x
        inv_q_imag = -zR / (R**2 + zR**2)
        expected_wx = math.sqrt(-wavelength / (math.pi * inv_q_imag))

        assert abs(beam.w_x - expected_wx) < 1e-10

    def test_wx_returns_zero_for_spherical_wavefront(self):
        """Test w_x returns 0.0 when q_x is purely real (Im(1/q)=0)"""
        wavelength = 0.5e-3
        q_x = complex(100.0, 0.0)

        beam = GaussianBeam(wavelength, q_x, q_x, Mock())

        assert beam.w_x == 0.0

    def test_wy_equals_wx_for_isotropic_beam(self):
        """Test w_y equals w_x when q_x equals q_y"""
        wavelength = 0.5e-3
        q = complex(50.0, 25.0)

        beam = GaussianBeam(wavelength, q, q, Mock())

        assert abs(beam.w_x - beam.w_y) == 0.0

    def test_wy_differs_from_wx_for_astigmatic_beam(self):
        """Test w_y differs from w_x for anisotropic (astigmatic) beams"""
        wavelength = 0.5e-3
        q_x = complex(50.0, 25.0)
        q_y = complex(100.0, 10.0)

        beam = GaussianBeam(wavelength, q_x, q_y, Mock())

        assert abs(beam.w_x - beam.w_y) > 1e-10


@pytest.mark.skipif(not OPENLENS_AVAILABLE, reason="openlens not installed")
class TestGaussianBeamAdvanceQ:
    """Test GaussianBeam.advance_q method (advance q parameter by distance)"""

    def test_advance_q_in_vacuum(self):
        """Test advance_q with n=1.0 (vacuum)"""
        wavelength = 0.5e-3
        q_x_init = complex(50.0, 25.0)
        q_y_init = complex(100.0, 10.0)

        beam = GaussianBeam(wavelength, q_x_init, q_y_init, Mock())
        beam.n = 1.0

        distance = 5.0
        beam.advance_q(distance)

        assert abs(beam.q_x - (q_x_init + distance)) < 1e-10
        assert abs(beam.q_y - (q_y_init + distance)) < 1e-10

    def test_advance_q_in_glass(self):
        """Test advance_q with n>1 (glass medium)"""
        wavelength = 0.5e-3
        q_x_init = complex(50.0, 25.0)
        q_y_init = complex(100.0, 10.0)

        beam = GaussianBeam(wavelength, q_x_init, q_y_init, Mock())
        n = 1.5
        beam.n = n

        distance = 10.0
        beam.advance_q(distance)

        step = distance / n
        assert abs(beam.q_x - (q_x_init + step)) < 1e-10
        assert abs(beam.q_y - (q_y_init + step)) < 1e-10

    def test_advance_q_with_zero_n(self):
        """Test advance_q edge case with n=0 (uses distance as fallback)"""
        wavelength = 0.5e-3
        q_x_init = complex(50.0, 25.0)
        q_y_init = complex(100.0, 10.0)

        beam = GaussianBeam(wavelength, q_x_init, q_y_init, Mock())
        beam.n = 0

        distance = 5.0
        beam.advance_q(distance)

        assert abs(beam.q_x - (q_x_init + distance)) < 1e-10
        assert abs(beam.q_y - (q_y_init + distance)) < 1e-10

    def test_advance_q_accumulates_correctly(self):
        """Test multiple advance_q calls accumulate correctly"""
        wavelength = 0.5e-3
        q_x_init = complex(50.0, 25.0)
        q_y_init = complex(100.0, 10.0)

        beam = GaussianBeam(wavelength, q_x_init, q_y_init, Mock())
        beam.n = 1.5

        distances = [1.0, 2.0, 3.0]
        total_step = sum(d / beam.n for d in distances)

        for d in distances:
            beam.advance_q(d)

        expected_q_x = q_x_init + total_step
        expected_q_y = q_y_init + total_step

        assert abs(beam.q_x - expected_q_x) < 1e-10
        assert abs(beam.q_y - expected_q_y) < 1e-10


@pytest.mark.skipif(not OPENLENS_AVAILABLE, reason="openlens not installed")
class TestBeamSynthesisPropagatorPropagateToImage:
    """Test BeamSynthesisPropagator.propagate_to_image method"""

    def test_propagate_to_image_empty_system(self):
        """Test propagate_to_image returns empty arrays for empty system"""
        empty_system = Mock(elements=[])
        propagator = BeamSynthesisPropagator(empty_system)

        dy, dz, intensity = propagator.propagate_to_image()

        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)
        assert dy.size == 0
        assert dz.size == 0
        assert intensity.size == 0

    def test_propagate_to_image_returns_numpy_arrays(self):
        """Test propagate_to_image returns numpy arrays"""
        element = Mock()
        element.lens.diameter = 10.0
        element.position = 0.0
        system = Mock(elements=[element])

        propagator = BeamSynthesisPropagator(system)
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=100.0)

        dy, dz, intensity = propagator.propagate_to_image()

        assert isinstance(dy, np.ndarray)
        assert isinstance(dz, np.ndarray)
        assert isinstance(intensity, np.ndarray)

    def test_propagate_to_image_output_shape(self):
        """Test output arrays have correct shape (detector_pixels x detector_pixels)"""
        element = Mock()
        element.lens.diameter = 10.0
        element.position = 0.0
        system = Mock(elements=[element])

        propagator = BeamSynthesisPropagator(system)
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=100.0)

        pixels = 64
        dy, dz, intensity = propagator.propagate_to_image(detector_pixels=pixels)

        assert dy.shape == (pixels, pixels)
        assert dz.shape == (pixels, pixels)
        assert intensity.shape == (pixels, pixels)

    def test_propagate_to_image_intensity_non_negative(self):
        """Test intensity map contains only non-negative values"""
        element = Mock()
        element.lens.diameter = 10.0
        element.position = 0.0
        system = Mock(elements=[element])

        propagator = BeamSynthesisPropagator(system)
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=100.0)

        dy, dz, intensity = propagator.propagate_to_image()

        assert np.all(intensity >= 0.0)

    def test_propagate_to_image_detector_grid_centered(self):
        """Test detector grid is centered at origin (symmetric)"""
        element = Mock()
        element.lens.diameter = 10.0
        element.position = 0.0
        system = Mock(elements=[element])

        propagator = BeamSynthesisPropagator(system)
        propagator._element_tracers = Mock(return_value=[])
        propagator._trace_beamlet = Mock()
        propagator._default_image_plane = Mock(return_value=100.0)

        detector_size = 0.4
        dy, dz, intensity = propagator.propagate_to_image(detector_size=detector_size)

        assert abs(np.min(dy) + np.max(dy)) < 1e-10
        assert abs(np.min(dz) + np.max(dz)) < 1e-10
        assert abs(np.min(dy)) <= detector_size / 2
        assert abs(np.max(dy)) <= detector_size / 2
