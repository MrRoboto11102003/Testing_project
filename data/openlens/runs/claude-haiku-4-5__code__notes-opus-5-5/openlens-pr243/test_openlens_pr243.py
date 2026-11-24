import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestCalculateBackFocalLength:
    """Test suite for OpticalSystem.calculate_back_focal_length"""

    def test_empty_system_returns_none(self):
        """Empty system (no elements) should return None"""
        system = OpticalSystem()
        result = system.calculate_back_focal_length()
        assert result is None

    def test_single_biconvex_lens_exact_value(self):
        """Single biconvex lens: R1=100, R2=-100, n=1.5, t=5"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        expected = 11800 / 119  # ≈ 99.159664
        assert result == pytest.approx(expected, rel=1e-9)

    def test_single_plano_convex_lens_exact_value(self):
        """Plano-convex lens: R1=100, R2=inf, n=1.5, t=5"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=float('inf'),
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        expected = 590 / 3  # ≈ 196.66667
        assert result == pytest.approx(expected, rel=1e-9)

    def test_single_biconcave_lens_negative_bfl(self):
        """Biconcave (diverging) lens: R1=-100, R2=100, n=1.5, t=5"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=-100,
            radius_of_curvature_2=100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        expected = -12200 / 121  # ≈ -100.82645
        assert result == pytest.approx(expected, rel=1e-9)
        assert result < 0

    def test_both_flat_surfaces_afocal_returns_none(self):
        """Lens with both flat surfaces (R1=R2=inf) should return None"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=float('inf'),
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        assert result is None

    def test_two_biconvex_lenses_with_gap(self):
        """Two biconvex lenses separated by 10 mm air gap"""
        system = OpticalSystem()
        lens1 = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        lens2 = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10)
        result = system.calculate_back_focal_length()
        expected = 10036.5 / 222.2325  # ≈ 45.16216
        assert result == pytest.approx(expected, rel=1e-5)

    def test_override_refractive_index_single_element(self):
        """Override refractive index for single element"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.6
        )
        system.add_lens(lens)
        result_override = system.calculate_back_focal_length(n_overrides={0: 1.5})
        expected = 11800 / 119
        assert result_override == pytest.approx(expected, rel=1e-9)

    def test_override_changes_result_from_base_index(self):
        """Override with different index produces different result"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.6
        )
        system.add_lens(lens)
        result_no_override = system.calculate_back_focal_length()
        result_override = system.calculate_back_focal_length(n_overrides={0: 1.5})
        assert result_no_override != pytest.approx(result_override, rel=1e-6)

    def test_missing_key_in_override_dict_ignored(self):
        """Override dict with non-existent element index is ignored"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result_no_override = system.calculate_back_focal_length()
        result_missing_key = system.calculate_back_focal_length(n_overrides={5: 2.0})
        assert result_no_override == pytest.approx(result_missing_key, rel=1e-9)

    def test_empty_dict_override_treated_as_falsy(self):
        """Empty dict should be treated as falsy (ignored)"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result_no_override = system.calculate_back_focal_length()
        result_empty_dict = system.calculate_back_focal_length(n_overrides={})
        assert result_no_override == pytest.approx(result_empty_dict, rel=1e-9)

    def test_air_gap_thickness_affects_result(self):
        """Different air gaps produce different BFL values"""
        system1 = OpticalSystem()
        system1.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.5))
        system1.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.5), air_gap_before=5)
        
        system2 = OpticalSystem()
        system2.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.5))
        system2.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.5), air_gap_before=10)
        
        result1 = system1.calculate_back_focal_length()
        result2 = system2.calculate_back_focal_length()
        assert result1 != pytest.approx(result2, rel=1e-5)

    def test_plano_concave_diverging_negative_bfl(self):
        """Plano-concave lens is diverging, BFL < 0"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        assert result is not None
        assert result < 0

    def test_biconvex_converging_positive_bfl(self):
        """Biconvex lens is converging, BFL > 0"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            refractive_index=1.5
        )
        system.add_lens(lens)
        result = system.calculate_back_focal_length()
        assert result is not None
        assert result > 0

    def test_return_type_is_float_or_none(self):
        """Return value is always float or None"""
        system = OpticalSystem()
        assert system.calculate_back_focal_length() is None
        
        system.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.5))
        result = system.calculate_back_focal_length()
        assert isinstance(result, float)

    def test_multiple_elements_with_multiple_overrides(self):
        """Override refractive indices for multiple elements"""
        system = OpticalSystem()
        system.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.6))
        system.add_lens(Lens(
            radius_of_curvature_1=100, radius_of_curvature_2=-100,
            thickness=5, diameter=25, refractive_index=1.6), air_gap_before=10)
        
        result = system.calculate_back_focal_length(n_overrides={0: 1.5, 1: 1.5})
        assert result is not None
        assert isinstance(result, float)
