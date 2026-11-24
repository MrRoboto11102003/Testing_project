import pytest
import math
import numpy as np
from openlens.analysis.beam_synthesis import GaussianBeam, BeamSynthesisPropagator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestGaussianBeamRx:
    """Test GaussianBeam.R_x property."""
    
    def test_rx_infinite_pure_imaginary(self):
        """R_x is infinite for pure imaginary q_x."""
        beam = GaussianBeam(wavelength=1.0, q_x=1j, q_y=1j, ray=None)
        assert math.isinf(beam.R_x) and beam.R_x > 0
    
    def test_rx_finite_positive(self):
        """R_x = 2.0 for q_x = 1+1j."""
        beam = GaussianBeam(wavelength=1.0, q_x=1+1j, q_y=1j, ray=None)
        assert beam.R_x == pytest.approx(2.0)
    
    def test_rx_finite_negative(self):
        """R_x is negative for negative real part of q."""
        beam = GaussianBeam(wavelength=1.0, q_x=-3+4j, q_y=1j, ray=None)
        assert beam.R_x == pytest.approx(-25/3)
    
    def test_rx_zero_q_raises(self):
        """R_x raises ZeroDivisionError when q_x = 0."""
        beam = GaussianBeam(wavelength=1.0, q_x=0j, q_y=1j, ray=None)
        with pytest.raises(ZeroDivisionError):
            _ = beam.R_x


class TestGaussianBeamAdvanceQ:
    """Test GaussianBeam.advance_q method."""
    
    def test_advance_q_unity_n(self):
        """advance_q with n=1 adds distance to q real parts."""
        beam = GaussianBeam(wavelength=1.0, q_x=0+10j, q_y=0+10j, ray=None, n=1.0)
        beam.advance_q(5.0)
        assert beam.q_x == pytest.approx(5+10j)
        assert beam.q_y == pytest.approx(5+10j)
    
    def test_advance_q_divides_by_n(self):
        """advance_q divides distance by n when n != 0."""
        beam = GaussianBeam(wavelength=1.0, q_x=0+10j, q_y=0+10j, ray=None, n=1.5)
        beam.advance_q(3.0)
        assert beam.q_x == pytest.approx(2+10j)
    
    def test_advance_q_zero_n_fallback(self):
        """advance_q with n=0 uses raw distance (no division)."""
        beam = GaussianBeam(wavelength=1.0, q_x=0+10j, q_y=0+10j, ray=None, n=0)
        beam.advance_q(3.0)
        assert beam.q_x == pytest.approx(3+10j)


class TestGaussianBeamWx:
    """Test GaussianBeam.w_x property."""
    
    def test_wx_zero_imaginary(self):
        """w_x returns 0.0 when q_x.imag == 0."""
        beam = GaussianBeam(wavelength=1.0, q_x=5+0j, q_y=1j, ray=None)
        assert beam.w_x == 0.0
    
    def test_wx_at_waist(self):
        """w_x = 1.0 at waist with wavelength = pi."""
        beam = GaussianBeam(wavelength=math.pi, q_x=1j*math.pi, q_y=1j, ray=None)
        assert beam.w_x == pytest.approx(1.0)
    
    def test_wx_negative_imaginary_raises(self):
        """w_x raises ValueError for negative imaginary q_x."""
        beam = GaussianBeam(wavelength=1.0, q_x=-1j*math.pi, q_y=1j, ray=None)
        with pytest.raises(ValueError):
            _ = beam.w_x


class TestGaussianBeamWy:
    """Test GaussianBeam.w_y property."""
    
    def test_wy_zero_imaginary(self):
        """w_y returns 0.0 when q_y.imag == 0."""
        beam = GaussianBeam(wavelength=1.0, q_x=1j, q_y=5+0j, ray=None)
        assert beam.w_y == 0.0
    
    def test_wy_at_waist(self):
        """w_y = 1.0 at waist with wavelength = pi."""
        beam = GaussianBeam(wavelength=math.pi, q_x=1j, q_y=1j*math.pi, ray=None)
        assert beam.w_y == pytest.approx(1.0)
    
    def test_wy_negative_imaginary_raises(self):
        """w_y raises ValueError for negative imaginary q_y."""
        beam = GaussianBeam(wavelength=1.0, q_x=1j, q_y=-1j*math.pi, ray=None)
        with pytest.raises(ValueError):
            _ = beam.w_y


class TestBeamSynthesisPropagator:
    """Test BeamSynthesisPropagator.propagate_to_image method."""
    
    def test_empty_system_empty_output(self):
        """Empty system returns three empty arrays."""
        system = OpticalSystem()
        prop = BeamSynthesisPropagator(system)
        DY, DZ, I = prop.propagate_to_image(grid_size=8, detector_pixels=16)
        assert DY.size == 0 and DZ.size == 0 and I.size == 0
    
    def test_output_shape_and_validity(self):
        """Output has correct shape; intensity is real, non-negative, finite."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        prop = BeamSynthesisPropagator(system)
        DY, DZ, I = prop.propagate_to_image(grid_size=6, detector_pixels=16)
        assert DY.shape == (16, 16) and DZ.shape == (16, 16) and I.shape == (16, 16)
        assert np.isrealobj(I) and (I >= 0).all() and np.isfinite(I).all()
    
    def test_detector_grids_range(self):
        """Detector grids span the specified range."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        prop = BeamSynthesisPropagator(system)
        DY, DZ, _ = prop.propagate_to_image(
            grid_size=6, detector_pixels=20, detector_size=0.2
        )
        expected = np.linspace(-0.1, 0.1, 20)
        assert np.allclose(DY[0, :], expected)
        assert np.allclose(DZ[:, 0], expected)
    
    def test_determinism(self):
        """Identical calls produce identical intensities."""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        prop = BeamSynthesisPropagator(system)
        _, _, I1 = prop.propagate_to_image(grid_size=6, detector_pixels=12)
        _, _, I2 = prop.propagate_to_image(grid_size=6, detector_pixels=12)
        assert np.array_equal(I1, I2)
