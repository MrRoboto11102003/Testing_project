import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemCalculateBackFocalLength:
    """Test suite for OpticalSystem.calculate_back_focal_length()"""

    def test_single_lens_system_positive_power(self):
        """Test BFL calculation for a converging single-element system."""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Single Lens")
        system.add_lens(lens)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        assert bfl > 0, "Converging lens should have positive BFL"
        
        # Verify consistency with single lens BFL
        lens_bfl = lens.calculate_back_focal_length()
        assert lens_bfl is not None
        assert abs(bfl - lens_bfl) < 1e-6, "System BFL should match single lens BFL"

    def test_single_lens_system_negative_power(self):
        """Test BFL calculation for a diverging single-element system."""
        lens = Lens(
            name="Biconcave",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=3.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Single Diverging")
        system.add_lens(lens)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        assert bfl < 0, "Diverging lens should have negative BFL"

    def test_two_element_system_with_air_gap(self):
        """Test BFL for a two-element system with air separation."""
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
            radius_of_curvature_2=-200.0,
            thickness=3.0,
            diameter=25.0,
            material="F2",
        )
        
        system = OpticalSystem(name="Two-element")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)

    def test_empty_system_returns_none(self):
        """Test that an empty system (no lenses) returns None."""
        system = OpticalSystem(name="Empty")
        bfl = system.calculate_back_focal_length()
        assert bfl is None

    def test_afocal_system_returns_none(self):
        """Test that afocal systems (zero optical power) return None."""
        # Create a symmetric system: +50mm and -50mm power lenses
        positive = Lens(
            name="Positive",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=1.0,
            diameter=25.0,
            material="BK7",
        )
        negative = Lens(
            name="Negative",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=1.0,
            diameter=25.0,
            material="BK7",
        )
        
        system = OpticalSystem(name="Afocal")
        system.add_lens(positive)
        system.add_lens(negative, air_gap_before=10.0)
        
        # For an afocal system, the C matrix element should be ~0
        bfl = system.calculate_back_focal_length()
        assert bfl is None, "Afocal systems should return None"

    def test_bfl_with_n_overrides_parameter(self):
        """Test BFL calculation using refractive index overrides."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="With override")
        system.add_lens(lens)
        
        # Normal BFL
        bfl_normal = system.calculate_back_focal_length()
        
        # With index override (using different refractive index)
        bfl_override = system.calculate_back_focal_length(n_overrides={0: 1.6})
        
        assert bfl_normal is not None
        assert bfl_override is not None
        assert isinstance(bfl_normal, float)
        assert isinstance(bfl_override, float)
        # Different indices should produce different BFL values
        assert bfl_normal != bfl_override

    def test_thick_lens_effects_included(self):
        """Test that thick lens effects are properly included in calculation."""
        thick_lens = Lens(
            name="Thick",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=20.0,
            diameter=50.0,
            material="BK7",
        )
        thin_lens = Lens(
            name="Thin",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=1.0,
            diameter=50.0,
            material="BK7",
        )
        
        system_thick = OpticalSystem(name="Thick system")
        system_thick.add_lens(thick_lens)
        bfl_thick = system_thick.calculate_back_focal_length()
        
        system_thin = OpticalSystem(name="Thin system")
        system_thin.add_lens(thin_lens)
        bfl_thin = system_thin.calculate_back_focal_length()
        
        assert bfl_thick is not None
        assert bfl_thin is not None
        # Thickness effects should cause measurable difference
        assert abs(bfl_thick - bfl_thin) > 1e-4

    def test_multi_element_three_lens_system(self):
        """Test BFL for a three-element optical system."""
        l1 = Lens(
            name="L1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        l2 = Lens(
            name="L2",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=4.0,
            diameter=25.0,
            material="F2",
        )
        l3 = Lens(
            name="L3",
            radius_of_curvature_1=80.0,
            radius_of_curvature_2=-80.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        
        system = OpticalSystem(name="Three-element")
        system.add_lens(l1)
        system.add_lens(l2, air_gap_before=0.5)
        system.add_lens(l3, air_gap_before=1.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)

    def test_air_gap_variation_affects_bfl(self):
        """Test that varying air gap between elements changes BFL."""
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
            radius_of_curvature_2=-200.0,
            thickness=3.0,
            diameter=25.0,
            material="F2",
        )
        
        system_small = OpticalSystem(name="Small gap")
        system_small.add_lens(lens1)
        system_small.add_lens(lens2, air_gap_before=0.1)
        bfl_small = system_small.calculate_back_focal_length()
        
        system_large = OpticalSystem(name="Large gap")
        system_large.add_lens(lens1)
        system_large.add_lens(lens2, air_gap_before=10.0)
        bfl_large = system_large.calculate_back_focal_length()
        
        assert bfl_small is not None
        assert bfl_large is not None
        assert bfl_small != bfl_large, "Different air gaps should affect BFL"

    def test_return_type_is_optional_float(self):
        """Test that return type is None or float."""
        lens = Lens(
            name="Type check",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Type check system")
        system.add_lens(lens)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is None or isinstance(bfl, float), "BFL must return None or float"

    def test_plano_convex_system(self):
        """Test BFL for a plano-convex lens system."""
        plano_convex = Lens(
            name="Plano-convex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Plano-convex")
        system.add_lens(plano_convex)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        assert bfl > 0

    def test_consistency_across_multiple_calls(self):
        """Test that BFL calculation is deterministic (no side effects)."""
        lens = Lens(
            name="Deterministic test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Consistency")
        system.add_lens(lens)
        
        bfl1 = system.calculate_back_focal_length()
        bfl2 = system.calculate_back_focal_length()
        bfl3 = system.calculate_back_focal_length()
        
        assert bfl1 == bfl2 == bfl3, "Multiple calls should produce identical results"

    def test_zero_thickness_lens(self):
        """Test BFL for thin lens approximation (near-zero thickness)."""
        thin = Lens(
            name="Ultra-thin",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=0.1,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Thin lens")
        system.add_lens(thin)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)

    def test_meniscus_lens_system(self):
        """Test BFL for meniscus lens configuration."""
        meniscus = Lens(
            name="Meniscus",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        system = OpticalSystem(name="Meniscus")
        system.add_lens(meniscus)
        
        bfl = system.calculate_back_focal_length()
        # Meniscus can be positive or negative power
        assert bfl is None or isinstance(bfl, float)

    def test_cemented_doublet_zero_air_gap(self):
        """Test BFL for cemented doublet (zero air gap)."""
        crown = Lens(
            name="Crown",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        flint = Lens(
            name="Flint",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-50.0,
            thickness=4.0,
            diameter=25.0,
            material="F2",
        )
        
        system = OpticalSystem(name="Cemented doublet")
        system.add_lens(crown)
        system.add_lens(flint, air_gap_before=0.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is None or isinstance(bfl, float)
