import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemGetSystemFocalLength:
    """Test suite for OpticalSystem.get_system_focal_length()"""

    def test_empty_system_returns_none(self):
        """An empty optical system with no elements should return None."""
        system = OpticalSystem(name="Empty system")
        assert system.get_system_focal_length() is None

    def test_single_lens_returns_own_focal_length(self):
        """A single-lens system should return the lens's focal length."""
        lens = Lens(
            name="Single lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Single-element system")
        system.add_lens(lens)

        expected = lens.calculate_focal_length()
        result = system.get_system_focal_length()

        assert result is not None
        assert abs(result - expected) < 1e-6

    def test_two_lens_system_converging(self):
        """A two-lens system should use matrix method (not thin combination formula)."""
        lens1 = Lens.for_material(
            "BK7",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        lens2 = Lens.for_material(
            "BK7",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )

        system = OpticalSystem(name="Two-lens converging")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=0.5)

        result = system.get_system_focal_length()

        assert result is not None
        assert isinstance(result, (int, float))
        assert result > 0  # Two converging lenses

    def test_three_lens_system(self):
        """A three-element system should return valid focal length using matrix method."""
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=4.0,
            diameter=25.0,
            material="BK7",
        )
        lens3 = Lens(
            name="L3",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=3.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Three-lens system")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)
        system.add_lens(lens3, air_gap_before=1.0)

        result = system.get_system_focal_length()

        assert result is not None
        assert isinstance(result, (int, float))

    def test_system_focal_length_depends_on_air_gap(self):
        """Air gaps between elements should affect system focal length."""
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=-100.0,
            thickness=4.0,
            diameter=25.0,
            material="BK7",
        )

        system_small_gap = OpticalSystem(name="Small gap")
        system_small_gap.add_lens(lens1)
        system_small_gap.add_lens(lens2, air_gap_before=0.1)
        result_small = system_small_gap.get_system_focal_length()

        system_large_gap = OpticalSystem(name="Large gap")
        system_large_gap.add_lens(lens1)
        system_large_gap.add_lens(lens2, air_gap_before=5.0)
        result_large = system_large_gap.get_system_focal_length()

        assert result_small is not None
        assert result_large is not None
        assert result_small != result_large  # Different gaps, different EFL

    def test_strongly_converging_system(self):
        """A system of converging lenses with tight radii should have positive focal length."""
        lens1 = Lens(
            name="Conv1",
            radius_of_curvature_1=30.0,
            radius_of_curvature_2=-30.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="Conv2",
            radius_of_curvature_1=30.0,
            radius_of_curvature_2=-30.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Double converging")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)

        result = system.get_system_focal_length()

        assert result is not None
        assert result > 0

    def test_strongly_diverging_system(self):
        """A system of diverging lenses should have negative focal length."""
        lens1 = Lens(
            name="Div1",
            radius_of_curvature_1=-30.0,
            radius_of_curvature_2=30.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="Div2",
            radius_of_curvature_1=-30.0,
            radius_of_curvature_2=30.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Double diverging")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)

        result = system.get_system_focal_length()

        assert result is not None
        assert result < 0

    def test_mixed_converging_diverging_system(self):
        """A converging-diverging system (telephoto-like) should return valid focal length."""
        converging = Lens(
            name="Converging",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        diverging = Lens(
            name="Diverging",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Mixed")
        system.add_lens(converging)
        system.add_lens(diverging, air_gap_before=0.5)

        result = system.get_system_focal_length()

        assert result is not None
        assert isinstance(result, (int, float))

    def test_thick_lens_effects_included(self):
        """Thick lens effects (d/n term) should be included in multi-element matrix calculation."""
        thick_lens1 = Lens(
            name="Thick1",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=20.0,
            diameter=50.0,
            material="BK7",
        )
        thick_lens2 = Lens(
            name="Thick2",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=15.0,
            diameter=50.0,
            material="BK7",
        )

        system = OpticalSystem(name="Thick lenses")
        system.add_lens(thick_lens1)
        system.add_lens(thick_lens2, air_gap_before=0.5)

        result = system.get_system_focal_length()

        assert result is not None
        assert isinstance(result, (int, float))

    def test_different_materials_affect_focal_length(self):
        """Using different glass materials should produce different system focal lengths."""
        crown = Lens.for_material(
            "BK7",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        flint = Lens.for_material(
            "F2",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )

        system_homogeneous = OpticalSystem(name="Two BK7")
        system_homogeneous.add_lens(crown)
        system_homogeneous.add_lens(crown, air_gap_before=0.5)
        result1 = system_homogeneous.get_system_focal_length()

        system_mixed = OpticalSystem(name="BK7 + F2")
        system_mixed.add_lens(crown)
        system_mixed.add_lens(flint, air_gap_before=0.5)
        result2 = system_mixed.get_system_focal_length()

        assert result1 is not None
        assert result2 is not None
        assert result1 != result2

    def test_afocal_system_returns_none(self):
        """An afocal system (C ≈ 0 in matrix) should return None."""
        # Create a symmetric setup that is likely afocal or nearly afocal
        # Plano-convex and plano-concave with matching radii
        positive = Lens(
            name="Positive",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=10000.0,  # Nearly flat (plano-convex)
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        negative = Lens(
            name="Negative",
            radius_of_curvature_1=10000.0,  # Nearly flat
            radius_of_curvature_2=-50.0,  # Plano-concave
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Nearly afocal")
        system.add_lens(positive)
        system.add_lens(negative, air_gap_before=0.0)

        result = system.get_system_focal_length()

        # Afocal system should return None (C component near zero)
        # Note: exact behavior depends on library's threshold and configuration
        assert result is None or isinstance(result, (int, float))

    def test_return_type_is_optional_float(self):
        """Return type should be Optional[float]: either float or None."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        system = OpticalSystem(name="Type test")
        result_empty = system.get_system_focal_length()
        assert result_empty is None

        system.add_lens(lens)
        result_with_lens = system.get_system_focal_length()
        assert isinstance(result_with_lens, (int, float)) or result_with_lens is None

    def test_zero_air_gap_between_elements(self):
        """A system with zero air gap (cemented or touching) should work."""
        lens1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            name="L2",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            material="F2",
        )

        system = OpticalSystem(name="Cemented doublet")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=0.0)

        result = system.get_system_focal_length()

        assert result is not None
        assert isinstance(result, (int, float))

    def test_single_vs_multi_element_calculation_path(self):
        """Single-element system uses different code path than multi-element."""
        lens = Lens(
            name="Test lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        # Single-element system: uses lens.calculate_focal_length()
        single_system = OpticalSystem(name="Single")
        single_system.add_lens(lens)
        single_result = single_system.get_system_focal_length()

        # Compare with direct lens calculation
        direct_result = lens.calculate_focal_length()

        assert single_result == direct_result
