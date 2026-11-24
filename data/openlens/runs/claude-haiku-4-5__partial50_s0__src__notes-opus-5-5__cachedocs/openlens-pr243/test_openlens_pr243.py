import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemCalculateBackFocalLength:
    """Test OpticalSystem.calculate_back_focal_length method."""

    def test_empty_system_returns_none(self):
        """Empty system (no elements) should return None."""
        system = OpticalSystem()
        assert system.calculate_back_focal_length() is None

    def test_single_biconvex_lens(self):
        """Single biconvex lens with analytical result: BFL = 11800/119."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        expected = 11800.0 / 119.0  # ≈ 99.159664
        assert system.calculate_back_focal_length() == pytest.approx(expected, rel=1e-9)

    def test_single_plano_convex_lens(self):
        """Single plano-convex lens: R1=100, R2=inf; BFL = 590/3."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        expected = 590.0 / 3.0  # ≈ 196.66667
        assert system.calculate_back_focal_length() == pytest.approx(expected, rel=1e-9)

    def test_single_biconcave_diverging_lens(self):
        """Single biconcave (diverging) lens: BFL = -12200/121."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        expected = -12200.0 / 121.0  # ≈ -100.82645
        bfl = system.calculate_back_focal_length()
        assert bfl == pytest.approx(expected, rel=1e-9)
        assert bfl < 0

    def test_all_flat_surfaces_returns_none(self):
        """All flat surfaces (R1=inf, R2=inf) result in C=0, returns None."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        assert system.calculate_back_focal_length() is None

    def test_two_lenses_with_air_gap(self):
        """Two identical biconvex lenses with 10mm air gap."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        system.add_lens(lens, air_gap_before=10.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        # Air gap changes the result compared to single lens
        single_bfl = 11800.0 / 119.0
        assert bfl != pytest.approx(single_bfl, rel=0.01)

    def test_cemented_lenses_zero_gap(self):
        """Two identical biconvex lenses cemented (zero air gap)."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        system.add_lens(lens, air_gap_before=0.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)

    def test_n_override_single_element(self):
        """n_overrides[0] correctly overrides first element's refractive index."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.6,
        )
        system.add_lens(lens)
        
        # Override with n=1.5; should match biconvex with n=1.5
        bfl_overridden = system.calculate_back_focal_length(n_overrides={0: 1.5})
        expected = 11800.0 / 119.0
        assert bfl_overridden == pytest.approx(expected, rel=1e-9)

    def test_n_override_produces_different_result(self):
        """Verify n_override produces different result than no override."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.6,
        )
        system.add_lens(lens)
        
        bfl_original = system.calculate_back_focal_length()
        bfl_overridden = system.calculate_back_focal_length(n_overrides={0: 1.5})
        
        # Different indices must produce measurably different BFLs
        assert bfl_original != pytest.approx(bfl_overridden, rel=0.01)

    def test_n_override_nonexistent_index_ignored(self):
        """n_overrides with non-existent element indices are silently ignored."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        bfl_no_override = system.calculate_back_focal_length()
        # Override indices that don't exist
        bfl_with_invalid = system.calculate_back_focal_length(n_overrides={5: 2.0})
        
        assert bfl_no_override == pytest.approx(bfl_with_invalid)

    def test_empty_dict_override_treated_as_falsy(self):
        """Empty dict {} is falsy; behaves identical to None."""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system.add_lens(lens)
        
        bfl_none = system.calculate_back_focal_length(n_overrides=None)
        bfl_empty_dict = system.calculate_back_focal_length(n_overrides={})
        
        assert bfl_none == pytest.approx(bfl_empty_dict)

    def test_n_override_partial_multiple_elements(self):
        """Partial override: only specified element indices are overridden."""
        system = OpticalSystem()
        lens1 = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        lens2 = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.6,
        )
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=5.0)
        
        bfl = system.calculate_back_focal_length(n_overrides={1: 1.5})
        assert bfl is not None
        assert isinstance(bfl, float)

    def test_lens_thickness_affects_bfl(self):
        """Thicker lenses produce different BFL (thick lens effect)."""
        lens_thin = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=1.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        lens_thick = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=10.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        
        sys_thin = OpticalSystem()
        sys_thin.add_lens(lens_thin)
        
        sys_thick = OpticalSystem()
        sys_thick.add_lens(lens_thick)
        
        bfl_thin = sys_thin.calculate_back_focal_length()
        bfl_thick = sys_thick.calculate_back_focal_length()
        
        # Thickness affects the matrix calculation
        assert bfl_thin != pytest.approx(bfl_thick)

    def test_return_type_float_or_none(self):
        """Return type must always be float or None."""
        # None case
        empty_sys = OpticalSystem()
        result_empty = empty_sys.calculate_back_focal_length()
        assert result_empty is None
        
        # Float case
        sys_with_lens = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        sys_with_lens.add_lens(lens)
        result_float = sys_with_lens.calculate_back_focal_length()
        assert isinstance(result_float, float)
