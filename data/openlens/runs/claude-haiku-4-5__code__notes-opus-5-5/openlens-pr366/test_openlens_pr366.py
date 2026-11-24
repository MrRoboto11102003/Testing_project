import pytest
import numpy as np

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.beam_synthesis import BeamSynthesisPropagator
from openlens.analysis.paraxial import find_paraxial_focus
import openlens.analysis.beam_synthesis as bs
import openlens.constants as const


class TestBeamSynthesisPropagatorPropagateToImage:
    """Test suite for BeamSynthesisPropagator.propagate_to_image"""

    @pytest.fixture
    def simple_system(self):
        """Create a simple single-lens optical system."""
        s = OpticalSystem()
        s.add_lens(Lens(diameter=25.0))
        return s

    def test_empty_system_returns_empty_arrays(self):
        """Empty system should return three empty arrays with size 0."""
        s = OpticalSystem()
        p = BeamSynthesisPropagator(s)
        DY, DZ, I = p.propagate_to_image()
        
        assert DY.size == 0
        assert DZ.size == 0
        assert I.size == 0

    def test_output_shapes_match_detector_pixels(self, simple_system):
        """Output shapes should be (detector_pixels, detector_pixels)."""
        p = BeamSynthesisPropagator(simple_system)
        detector_pixels = 16
        DY, DZ, I = p.propagate_to_image(grid_size=8, detector_pixels=detector_pixels)
        
        assert DY.shape == (detector_pixels, detector_pixels)
        assert DZ.shape == (detector_pixels, detector_pixels)
        assert I.shape == (detector_pixels, detector_pixels)

    def test_intensity_is_real_nonnegative_finite(self, simple_system):
        """Intensity must be real, non-negative, and finite."""
        p = BeamSynthesisPropagator(simple_system)
        DY, DZ, I = p.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert np.isrealobj(I)
        assert (I >= 0).all()
        assert np.isfinite(I).all()

    def test_detector_grid_linspace_bounds(self, simple_system):
        """Detector grids should match expected linspace values."""
        p = BeamSynthesisPropagator(simple_system)
        detector_size = 0.1
        detector_pixels = 16
        DY, DZ, I = p.propagate_to_image(
            grid_size=8, 
            detector_size=detector_size,
            detector_pixels=detector_pixels
        )
        
        expected = np.linspace(-detector_size/2, detector_size/2, detector_pixels)
        
        # DY varies along columns (axis 1)
        assert np.allclose(DY[0], expected)
        # DZ varies along rows (axis 0)
        assert np.allclose(DZ[:, 0], expected)

    def test_grid_size_one_produces_zero_intensity(self, simple_system):
        """grid_size=1 should produce zero intensity (single point outside pupil)."""
        p = BeamSynthesisPropagator(simple_system)
        DY, DZ, I = p.propagate_to_image(grid_size=1, detector_pixels=8)
        
        assert I.sum() == 0.0

    def test_grid_size_zero_produces_zero_intensity(self, simple_system):
        """grid_size=0 should produce zero intensity."""
        p = BeamSynthesisPropagator(simple_system)
        DY, DZ, I = p.propagate_to_image(grid_size=0, detector_pixels=8)
        
        assert (I == 0.0).all()

    def test_detector_pixels_zero_produces_empty_arrays(self, simple_system):
        """detector_pixels=0 should produce empty (0,0) arrays."""
        p = BeamSynthesisPropagator(simple_system)
        DY, DZ, I = p.propagate_to_image(grid_size=8, detector_pixels=0)
        
        assert DY.shape == (0, 0)
        assert DZ.shape == (0, 0)
        assert I.shape == (0, 0)

    def test_detector_pixels_one_produces_single_point(self, simple_system):
        """detector_pixels=1 should produce (1,1) arrays with single point."""
        p = BeamSynthesisPropagator(simple_system)
        detector_size = 0.1
        DY, DZ, I = p.propagate_to_image(grid_size=8, detector_pixels=1, detector_size=detector_size)
        
        assert DY.shape == (1, 1)
        assert DZ.shape == (1, 1)
        assert I.shape == (1, 1)
        assert np.isclose(DY[0, 0], -detector_size/2)

    def test_determinism_identical_calls(self, simple_system):
        """Two identical calls must produce identical results."""
        p = BeamSynthesisPropagator(simple_system)
        
        DY1, DZ1, I1 = p.propagate_to_image(grid_size=8, detector_pixels=16)
        DY2, DZ2, I2 = p.propagate_to_image(grid_size=8, detector_pixels=16)
        
        assert np.array_equal(DY1, DY2)
        assert np.array_equal(DZ1, DZ2)
        assert np.array_equal(I1, I2)

    def test_default_image_plane_equals_paraxial_focus(self, simple_system):
        """Default image_plane_x should match paraxial focus."""
        p = BeamSynthesisPropagator(simple_system)
        
        DY_default, DZ_default, I_default = p.propagate_to_image(grid_size=6, detector_pixels=12)
        
        focus = find_paraxial_focus(
            simple_system, 
            const.WAVELENGTH_GREEN,
            tracer=p.tracer
        )
        
        if focus is not None and np.isfinite(focus):
            DY_explicit, DZ_explicit, I_explicit = p.propagate_to_image(
                grid_size=6, 
                detector_pixels=12,
                image_plane_x=focus
            )
            
            assert np.allclose(I_default, I_explicit)

    def test_fallback_plane_when_paraxial_focus_none(self, simple_system, monkeypatch, caplog):
        """When paraxial focus is None, use total_length + 50.0 as fallback."""
        p = BeamSynthesisPropagator(simple_system)
        monkeypatch.setattr(bs, "find_paraxial_focus", lambda *a, **k: None)
        
        with caplog.at_level("WARNING"):
            DY_default, DZ_default, I_default = p.propagate_to_image(grid_size=6, detector_pixels=12)
        
        fallback_x = simple_system.get_total_length() + 50.0
        DY_fallback, DZ_fallback, I_fallback = p.propagate_to_image(
            grid_size=6, 
            detector_pixels=12,
            image_plane_x=fallback_x
        )
        
        assert np.allclose(I_default, I_fallback)
        assert "no paraxial focus" in caplog.text

    def test_fallback_plane_when_paraxial_focus_nan(self, simple_system, monkeypatch, caplog):
        """When paraxial focus is NaN, use total_length + 50.0 as fallback."""
        p = BeamSynthesisPropagator(simple_system)
        monkeypatch.setattr(bs, "find_paraxial_focus", lambda *a, **k: float('nan'))
        
        with caplog.at_level("WARNING"):
            DY_default, DZ_default, I_default = p.propagate_to_image(grid_size=6, detector_pixels=12)
        
        fallback_x = simple_system.get_total_length() + 50.0
        DY_fallback, DZ_fallback, I_fallback = p.propagate_to_image(
            grid_size=6, 
            detector_pixels=12,
            image_plane_x=fallback_x
        )
        
        assert np.allclose(I_default, I_fallback)
        assert "no paraxial focus" in caplog.text

    def test_explicit_image_plane_x_changes_intensity(self, simple_system):
        """Changing image_plane_x far from focus should change intensity."""
        p = BeamSynthesisPropagator(simple_system)
        
        focus = find_paraxial_focus(
            simple_system,
            const.WAVELENGTH_GREEN,
            tracer=p.tracer
        )
        
        if focus is not None and np.isfinite(focus):
            DY1, DZ1, I1 = p.propagate_to_image(grid_size=6, detector_pixels=12, image_plane_x=focus)
            DY2, DZ2, I2 = p.propagate_to_image(
                grid_size=6, 
                detector_pixels=12,
                image_plane_x=focus + 20.0
            )
            
            assert not np.allclose(I1, I2)

    def test_different_wavelengths_produce_different_intensities(self, simple_system):
        """Different wavelengths should produce different intensities."""
        p = BeamSynthesisPropagator(simple_system)
        
        wavelength1 = const.WAVELENGTH_GREEN * const.NM_TO_MM
        wavelength2 = const.WAVELENGTH_RED * const.NM_TO_MM
        
        DY1, DZ1, I1 = p.propagate_to_image(wavelength_mm=wavelength1, grid_size=6, detector_pixels=12)
        DY2, DZ2, I2 = p.propagate_to_image(wavelength_mm=wavelength2, grid_size=6, detector_pixels=12)
        
        assert np.array_equal(DY1, DY2)
        assert np.array_equal(DZ1, DZ2)
        assert not np.allclose(I1, I2)

    def test_larger_detector_size_scales_grid(self, simple_system):
        """Larger detector_size should proportionally scale grid values."""
        p = BeamSynthesisPropagator(simple_system)
        
        DY1, DZ1, I1 = p.propagate_to_image(grid_size=6, detector_pixels=12, detector_size=0.1)
        DY2, DZ2, I2 = p.propagate_to_image(grid_size=6, detector_pixels=12, detector_size=0.2)
        
        assert DY1.shape == DY2.shape
        assert np.allclose(DY2, DY1 * 2)
        assert np.allclose(DZ2, DZ1 * 2)
