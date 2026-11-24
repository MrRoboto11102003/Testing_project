import pytest
import math
import numpy as np
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.beam_synthesis import GaussianBeam, BeamSynthesisPropagator
from openlens.ray import Ray3D
from openlens.vector3 import vec3


class TestGaussianBeamWX:
    """Tests for GaussianBeam.w_x property"""
    
    def test_w_x_zero_imaginary_returns_zero(self):
        """w_x returns 0.0 when q_x has zero imaginary part"""
        beam = GaussianBeam(1.0, complex(5, 0), complex(1, 1), None)
        assert beam.w_x == 0.0
    
    def test_w_x_at_waist(self):
        """w_x at waist: q_x=1j*pi, wavelength=1.0 gives w_x=1.0"""
        beam = GaussianBeam(1.0, complex(0, math.pi), complex(0, 1), None)
        assert beam.w_x == pytest.approx(1.0)


class TestGaussianBeamWY:
    """Tests for GaussianBeam.w_y property"""
    
    def test_w_y_zero_imaginary_returns_zero(self):
        """w_y returns 0.0 when q_y has zero imaginary part"""
        beam = GaussianBeam(1.0, complex(1, 1), complex(5, 0), None)
        assert beam.w_y == 0.0
    
    def test_w_y_at_waist(self):
        """w_y at waist with specific q_y"""
        beam = GaussianBeam(math.pi, complex(0, 1), complex(0, 1), None)
        assert beam.w_y == pytest.approx(1.0)


class TestGaussianBeamRX:
    """Tests for GaussianBeam.R_x property"""
    
    def test_R_x_pure_imaginary_returns_infinity(self):
        """R_x returns +infinity for pure imaginary q_x"""
        beam = GaussianBeam(1.0, complex(0, 10), complex(0, 1), None)
        assert math.isinf(beam.R_x) and beam.R_x > 0
    
    def test_R_x_specific_value(self):
        """R_x for q_x=1+1j equals 2.0"""
        beam = GaussianBeam(1.0, complex(1, 1), complex(0, 1), None)
        assert beam.R_x == pytest.approx(2.0)
    
    def test_R_x_negative_real_part(self):
        """R_x for negative real part of q"""
        beam = GaussianBeam(1.0, complex(-3, 4), complex(0, 1), None)
        assert beam.R_x == pytest.approx(-25/3)
    
    def test_R_x_zero_q_raises(self):
        """R_x raises ZeroDivisionError for q_x=0"""
        beam = GaussianBeam(1.0, complex(0, 0), complex(0, 1), None)
        with pytest.raises(ZeroDivisionError):
            _ = beam.R_x


class TestGaussianBeamAdvanceQ:
    """Tests for GaussianBeam.advance_q method"""
    
    def test_advance_q_with_unit_index(self):
        """advance_q with n=1 adds distance directly"""
        beam = GaussianBeam(1.0, complex(0, 10), complex(0, 20), None, n=1.0)
        beam.advance_q(5.0)
        assert beam.q_x == pytest.approx(complex(5, 10))
        assert beam.q_y == pytest.approx(complex(5, 20))
    
    def test_advance_q_divides_by_index_and_preserves_ray(self):
        """advance_q with n>1 divides distance; ray unchanged"""
        ray = Ray3D(vec3(1, 2, 3), vec3(1, 0, 0))
        beam = GaussianBeam(1.0, complex(0, 10), complex(0, 20), ray, n=1.5)
        beam.advance_q(3.0)
        assert beam.q_x == pytest.approx(complex(2.0, 10))
        assert beam.ray.origin == vec3(1, 2, 3)


class TestBeamSynthesisPropagatorPropagateToImage:
    """Tests for BeamSynthesisPropagator.propagate_to_image"""
    
    def test_propagate_empty_system(self):
        """propagate_to_image returns empty arrays for empty system"""
        system = OpticalSystem()
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        assert DY.size == 0 and DZ.size == 0 and I.size == 0
    
    def test_propagate_correct_shapes(self):
        """propagate_to_image returns correct array shapes"""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0, radius_of_curvature_1=50.0))
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        assert DY.shape == (16, 16) and I.shape == (16, 16)
    
    def test_propagate_intensity_non_negative_finite(self):
        """propagate_to_image intensity is non-negative and finite"""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0, radius_of_curvature_1=50.0))
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=6, detector_pixels=12)
        assert (I >= 0.0).all()
        assert np.isfinite(I).all()
    
    def test_propagate_grid_size_one_zero_intensity(self):
        """propagate_to_image with grid_size=1 returns zero intensity"""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0))
        propagator = BeamSynthesisPropagator(system)
        DY, DZ, I = propagator.propagate_to_image(grid_size=1, detector_pixels=8)
        assert I.sum() == 0.0 and I.shape == (8, 8)
    
    def test_propagate_deterministic_results(self):
        """propagate_to_image is deterministic"""
        system = OpticalSystem()
        system.add_lens(Lens(diameter=25.0, radius_of_curvature_1=50.0))
        propagator = BeamSynthesisPropagator(system)
        
        I1 = propagator.propagate_to_image(grid_size=6, detector_pixels=12)[2]
        I2 = propagator.propagate_to_image(grid_size=6, detector_pixels=12)[2]
        
        assert np.array_equal(I1, I2)
