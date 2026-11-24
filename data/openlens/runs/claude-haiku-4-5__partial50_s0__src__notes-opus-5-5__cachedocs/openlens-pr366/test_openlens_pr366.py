import pytest
import numpy as np
from unittest.mock import patch
import logging

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
import openlens.analysis.beam_synthesis as bs
from openlens.constants import WAVELENGTH_GREEN, NM_TO_MM
from openlens.analysis.paraxial import find_paraxial_focus


class TestBeamSynthesisPropagatorPropagateToImage:
    """Test BeamSynthesisPropagator.propagate_to_image method."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple optical system with one lens."""
        system = OpticalSystem()
        lens = Lens(
            name="test_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system.add_lens(lens)
        return system

    @pytest.fixture
    def propagator(self, simple_system):
        """Create a BeamSynthesisPropagator."""
        return BeamSynthesisPropagator(simple_system)

    def test_basic_return_shape(self, propagator):
        """Test return tuple has correct shapes for non-empty system."""
        grid_size = 8
        detector_pixels = 16
        DY, DZ, Intensity = propagator.propagate_to_image(
            grid_size=grid_size, detector_pixels=detector_pixels
        )
        
        assert isinstance(DY, np.ndarray)
        assert isinstance(DZ, np.ndarray)
        assert isinstance(Intensity, np.ndarray)
        assert DY.shape == (detector_pixels, detector_pixels)
        assert DZ.shape == (detector_pixels, detector_pixels)
        assert Intensity.shape == (detector_pixels, detector_pixels)

    def test_intensity_properties(self, propagator):
        """Test intensity is real, non-negative, and finite."""
        _, _, Intensity = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert np.isrealobj(Intensity)
        assert (Intensity >= 0).all()
        assert np.isfinite(Intensity).all()

    def test_detector_grid_correctness(self, propagator):
        """Test detector grid values match expected linspace ranges."""
        detector_size = 0.1
        detector_pixels = 16
        DY, DZ, _ = propagator.propagate_to_image(
            grid_size=8, 
            detector_size=detector_size,
            detector_pixels=detector_pixels
        )
        
        expected_range = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels)
        
        # DY should vary along columns (axis 1), constant along rows (axis 0)
        for i in range(detector_pixels):
            assert np.allclose(DY[i, :], expected_range)
        
        # DZ should vary along rows (axis 0), constant along columns (axis 1)
        for j in range(detector_pixels):
            assert np.allclose(DZ[:, j], expected_range)

    def test_empty_system(self):
        """Test empty optical system returns empty arrays."""
        system = OpticalSystem()
        propagator = BeamSynthesisPropagator(system)
        
        DY, DZ, Intensity = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert DY.size == 0
        assert DZ.size == 0
        assert Intensity.size == 0

    def test_grid_size_one_no_beamlets(self, propagator):
        """Test grid_size=1 produces zero intensity (single point outside circle)."""
        _, _, Intensity = propagator.propagate_to_image(grid_size=1, detector_pixels=16)
        assert Intensity.sum() == 0.0

    def test_grid_size_zero(self, propagator):
        """Test grid_size=0 produces zero intensity."""
        _, _, Intensity = propagator.propagate_to_image(grid_size=0, detector_pixels=16)
        assert Intensity.sum() == 0.0

    def test_detector_pixels_zero(self, propagator):
        """Test detector_pixels=0 produces empty 2D arrays."""
        DY, DZ, Intensity = propagator.propagate_to_image(grid_size=8, detector_pixels=0)
        
        assert DY.shape == (0, 0)
        assert DZ.shape == (0, 0)
        assert Intensity.shape == (0, 0)

    def test_detector_pixels_one(self, propagator):
        """Test detector_pixels=1 produces 1x1 arrays with center value."""
        detector_size = 0.1
        DY, DZ, Intensity = propagator.propagate_to_image(
            grid_size=8, 
            detector_size=detector_size,
            detector_pixels=1
        )
        
        assert DY.shape == (1, 1)
        assert DZ.shape == (1, 1)
        assert Intensity.shape == (1, 1)
        assert np.isclose(DY[0, 0], -detector_size / 2)
        assert np.isclose(DZ[0, 0], -detector_size / 2)

    def test_determinism(self, propagator):
        """Test identical calls produce identical results."""
        result1 = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        result2 = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        DY1, DZ1, I1 = result1
        DY2, DZ2, I2 = result2
        
        assert np.array_equal(DY1, DY2)
        assert np.array_equal(DZ1, DZ2)
        assert np.array_equal(I1, I2)

    def test_wavelength_parameter_affects_intensity(self, propagator):
        """Test that different wavelengths produce different intensities."""
        wl1 = WAVELENGTH_GREEN * NM_TO_MM
        wl2 = wl1 * 1.1
        
        _, _, I1 = propagator.propagate_to_image(wavelength_mm=wl1, grid_size=8, detector_pixels=16)
        _, _, I2 = propagator.propagate_to_image(wavelength_mm=wl2, grid_size=8, detector_pixels=16)
        
        # Different wavelengths should produce different intensities
        assert not np.allclose(I1, I2)

    def test_default_image_plane_matches_paraxial_focus(self, propagator, simple_system):
        """Test default image plane uses paraxial focus."""
        _, _, I_default = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        focus = find_paraxial_focus(simple_system, WAVELENGTH_GREEN, tracer=propagator.tracer)
        
        if focus is not None and np.isfinite(focus):
            _, _, I_explicit = propagator.propagate_to_image(
                grid_size=8, detector_pixels=16, image_plane_x=focus
            )
            assert np.allclose(I_default, I_explicit)

    def test_explicit_image_plane_parameter(self, propagator, simple_system):
        """Test that image_plane_x parameter affects intensity."""
        focus = find_paraxial_focus(simple_system, WAVELENGTH_GREEN, tracer=propagator.tracer)
        
        if focus is not None:
            shifted_plane = focus + 10.0
            _, _, I_focus = propagator.propagate_to_image(
                grid_size=8, detector_pixels=16, image_plane_x=focus
            )
            _, _, I_shifted = propagator.propagate_to_image(
                grid_size=8, detector_pixels=16, image_plane_x=shifted_plane
            )
            
            assert not np.allclose(I_focus, I_shifted, atol=1e-6)

    def test_fallback_on_invalid_configured_plane(self, caplog):
        """Test fallback to default when system has non-finite configured plane."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        system.image_plane_x_mm = float("inf")
        
        propagator = BeamSynthesisPropagator(system)
        
        with caplog.at_level(logging.WARNING):
            DY, DZ, Intensity = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert "ignoring non-finite image plane" in caplog.text
        assert Intensity.shape == (16, 16)

    def test_fallback_on_afocal_system(self, caplog):
        """Test fallback plane when paraxial focus search returns None."""
        system = OpticalSystem()
        lens = Lens(diameter=25.0, radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system.add_lens(lens)
        
        propagator = BeamSynthesisPropagator(system)
        
        with patch.object(bs, "find_paraxial_focus", return_value=None):
            with caplog.at_level(logging.WARNING):
                DY, DZ, Intensity = propagator.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert "no paraxial focus" in caplog.text
        assert Intensity.shape == (16, 16)

    def test_larger_grid_size_increases_total_intensity(self, propagator):
        """Test that larger grid_size produces more beamlets and higher total intensity."""
        _, _, I_small = propagator.propagate_to_image(grid_size=4, detector_pixels=16)
        _, _, I_large = propagator.propagate_to_image(grid_size=10, detector_pixels=16)
        
        assert I_large.sum() > I_small.sum()
