import pytest
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestGetSystemFocalLength:
    """Tests for OpticalSystem.get_system_focal_length()"""
    
    def test_empty_system_returns_none(self):
        """An empty system has no focal length."""
        s = OpticalSystem()
        assert s.get_system_focal_length() is None
    
    def test_single_lens_returns_focal_length(self):
        """A single lens returns its own focal length."""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(lens)
        
        expected = lens.calculate_focal_length()
        assert s.get_system_focal_length() == expected
    
    def test_two_lens_system_uses_matrix(self):
        """Two-lens system uses exact paraxial ABCD matrix formula."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1)
        s.add_lens(L2, air_gap_before=10.0)
        
        # Expected: C = -0.018519375 -> EFL = 53.9975 mm
        efl = s.get_system_focal_length()
        assert efl is not None
        assert efl == pytest.approx(53.9975, rel=1e-4)
    
    def test_single_lens_does_not_use_matrix(self):
        """Single lens ignores the matrix calculation."""
        lens = Lens(
            name="L1",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(lens)
        
        expected = lens.calculate_focal_length()
        
        # Monkeypatch matrix to return wrong values
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 0.0, 1.0)
        
        # Should still return lens's focal length, not compute from matrix
        assert s.get_system_focal_length() == expected
    
    def test_afocal_zero_c_returns_none(self):
        """When C = 0 exactly, return None (infinite focal length)."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 0.0, 1.0)
        assert s.get_system_focal_length() is None
    
    def test_afocal_small_c_below_threshold(self):
        """When abs(C) < 1e-10, treat as afocal and return None."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 5e-11, 1.0)
        assert s.get_system_focal_length() is None
    
    def test_focal_length_from_non_zero_c(self):
        """When abs(C) >= 1e-10, compute EFL = -1/C."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, -0.01, 1.0)
        assert s.get_system_focal_length() == pytest.approx(100.0)
    
    def test_negative_c_gives_positive_efl(self):
        """C < 0 gives positive EFL (converging system)."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, -0.02, 1.0)
        efl = s.get_system_focal_length()
        assert efl == pytest.approx(50.0)
        assert efl > 0
    
    def test_positive_c_gives_negative_efl(self):
        """C > 0 gives negative EFL (diverging system)."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: (1.0, 0.0, 0.01, 1.0)
        efl = s.get_system_focal_length()
        assert efl == pytest.approx(-100.0)
        assert efl < 0
    
    def test_matrix_none_returns_none(self):
        """When matrix calculation returns None, return None."""
        s = OpticalSystem()
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s.add_lens(L1)
        s.add_lens(L2)
        
        s._calculate_system_matrix = lambda n_overrides=None: None
        assert s.get_system_focal_length() is None
    
    def test_air_gap_sensitivity(self):
        """Changing air gap changes the EFL."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1)
        s.add_lens(L2, air_gap_before=10.0)
        
        efl_10mm = s.get_system_focal_length()
        
        # Change gap to 20 mm
        s.air_gaps[0].thickness = 20.0
        efl_20mm = s.get_system_focal_length()
        
        # Different gaps should give different EFLs
        assert efl_10mm is not None
        assert efl_20mm is not None
        assert efl_10mm != efl_20mm
    
    def test_cemented_pair_different_from_separated(self):
        """Cemented pair (gap = 0) gives different EFL than separated."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1)
        s.add_lens(L2, air_gap_before=0.0)
        
        efl_cemented = s.get_system_focal_length()
        
        # Compare with 10mm separated
        s.air_gaps[0].thickness = 10.0
        efl_separated = s.get_system_focal_length()
        
        assert efl_cemented is not None
        assert efl_separated is not None
        assert efl_cemented != efl_separated
    
    def test_tilt_does_not_affect_efl(self):
        """Tilt on elements does not change the EFL."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1, tilt_x=5.0)
        s.add_lens(L2, air_gap_before=10.0, tilt_y=3.0)
        
        efl = s.get_system_focal_length()
        assert efl is not None
        assert efl == pytest.approx(53.9975, rel=1e-4)
    
    def test_decenter_does_not_affect_efl(self):
        """Decenter on elements does not change the EFL."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1, decenter_x=1.0)
        s.add_lens(L2, air_gap_before=10.0, decenter_y=2.0)
        
        efl = s.get_system_focal_length()
        assert efl is not None
        assert efl == pytest.approx(53.9975, rel=1e-4)
    
    def test_thick_lens_differs_from_thin_combination(self):
        """Thick-lens matrix result differs significantly from thin-lens formula."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        s = OpticalSystem()
        s.add_lens(L1)
        s.add_lens(L2, air_gap_before=10.0)
        
        efl_matrix = s.get_system_focal_length()
        
        # Old thin-lens combination: 1/f = 1/f1 + 1/f2 - d/(f1*f2)
        f1 = L1.calculate_focal_length()
        f2 = L2.calculate_focal_length()
        d = 10.0
        
        if f1 and f2 and abs(f1) > 1e-10 and abs(f2) > 1e-10:
            efl_thin = 1.0 / (1.0/f1 + 1.0/f2 - d/(f1*f2))
            rel_diff = abs(efl_matrix - efl_thin) / abs(efl_thin)
            # Should differ by more than 0.01% per expert notes
            assert rel_diff > 1e-4
