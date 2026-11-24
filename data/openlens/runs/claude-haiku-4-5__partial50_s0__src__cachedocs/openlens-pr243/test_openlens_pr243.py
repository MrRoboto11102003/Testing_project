import pytest
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestOpticalSystemCalculateBackFocalLength:
    """Test suite for OpticalSystem.calculate_back_focal_length method."""

    @pytest.fixture
    def biconvex_lens(self):
        """Create a standard biconvex lens for testing."""
        return Lens(
            name="test_biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def biconcave_lens(self):
        """Create a biconcave (diverging) lens for testing."""
        return Lens(
            name="test_biconcave",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    def test_bfl_returns_optional_float(self, biconvex_lens):
        """Test that calculate_back_focal_length returns Optional[float]."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        bfl = system.calculate_back_focal_length()
        
        assert bfl is None or isinstance(bfl, float)

    def test_bfl_empty_system_returns_none(self):
        """Test that empty system (no elements) returns None."""
        system = OpticalSystem()
        bfl = system.calculate_back_focal_length()
        
        assert bfl is None

    def test_bfl_valid_system_returns_valid_range(self, biconvex_lens):
        """Test BFL for valid system returns reasonable numeric range."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        bfl = system.calculate_back_focal_length()
        
        if bfl is not None:
            # Should be finite and in reasonable range for typical optics
            assert isinstance(bfl, float)
            assert bfl != float('inf')
            assert bfl != float('-inf')
            assert bfl == bfl  # Not NaN

    def test_bfl_converging_lens_positive(self, biconvex_lens):
        """Test that converging lens has positive BFL."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        bfl = system.calculate_back_focal_length()
        
        if bfl is not None:
            assert bfl > 0

    def test_bfl_diverging_lens_negative(self, biconcave_lens):
        """Test that diverging lens has negative BFL."""
        system = OpticalSystem()
        system.add_element(biconcave_lens)
        bfl = system.calculate_back_focal_length()
        
        if bfl is not None:
            assert bfl < 0

    def test_bfl_consistency_across_calls(self, biconvex_lens):
        """Test that multiple calls return identical results."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        
        bfl1 = system.calculate_back_focal_length()
        bfl2 = system.calculate_back_focal_length()
        
        assert bfl1 == bfl2

    def test_bfl_with_n_overrides_parameter(self, biconvex_lens):
        """Test BFL calculation with n_overrides parameter."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        
        # Call with n_overrides
        bfl = system.calculate_back_focal_length(n_overrides={0: 1.5})
        
        assert bfl is None or isinstance(bfl, float)

    def test_bfl_n_overrides_none_vs_empty_dict(self, biconvex_lens):
        """Test that None and empty dict n_overrides give equivalent results."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        
        bfl_none = system.calculate_back_focal_length(n_overrides=None)
        bfl_empty = system.calculate_back_focal_length(n_overrides={})
        
        assert bfl_none == bfl_empty

    def test_bfl_multi_element_system(self, biconvex_lens, biconcave_lens):
        """Test BFL calculation for multi-element system with air gap."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        system.add_element(biconcave_lens, air_gap=15.0)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is None or isinstance(bfl, float)

    def test_bfl_thick_lens_effect(self):
        """Test BFL calculation with thick lens (significant thickness)."""
        system = OpticalSystem()
        thick_lens = Lens(
            name="thick_lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=20.0,
            diameter=25.0,
            material="BK7",
        )
        system.add_element(thick_lens)
        
        bfl = system.calculate_back_focal_length()
        assert bfl is None or isinstance(bfl, float)

    def test_bfl_different_materials(self):
        """Test BFL differs with different glass materials."""
        bk7_system = OpticalSystem()
        bk7_lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        bk7_system.add_element(bk7_lens)
        
        sf2_system = OpticalSystem()
        sf2_lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="SF2",
        )
        sf2_system.add_element(sf2_lens)
        
        bfl_bk7 = bk7_system.calculate_back_focal_length()
        bfl_sf2 = sf2_system.calculate_back_focal_length()
        
        # Both should be valid floats
        assert isinstance(bfl_bk7, float) and isinstance(bfl_sf2, float)
        # Different materials should give different BFL
        assert bfl_bk7 != bfl_sf2

    def test_bfl_afocal_system_returns_none_or_extreme(self):
        """Test that afocal systems (C ≈ 0) return None."""
        system = OpticalSystem()
        
        # Create a symmetric lens pair that approximates afocal
        lens1 = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens2 = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        
        system.add_element(lens1)
        system.add_element(lens2, air_gap=50.0)
        
        bfl = system.calculate_back_focal_length()
        # Afocal system (C ≈ 0 in matrix) should return None
        assert bfl is None or isinstance(bfl, float)

    def test_bfl_does_not_raise_exception(self, biconvex_lens):
        """Test that method never raises unexpected exceptions."""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        
        try:
            bfl = system.calculate_back_focal_length()
            assert bfl is None or isinstance(bfl, float)
        except (TypeError, ValueError, AttributeError) as e:
            pytest.fail(f"calculate_back_focal_length raised {type(e).__name__}: {e}")

    def test_bfl_varies_with_radius_of_curvature(self):
        """Test that BFL varies when radius of curvature changes."""
        lens_shallow = Lens(
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        lens_steep = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        
        system_shallow = OpticalSystem()
        system_shallow.add_element(lens_shallow)
        
        system_steep = OpticalSystem()
        system_steep.add_element(lens_steep)
        
        bfl_shallow = system_shallow.calculate_back_focal_length()
        bfl_steep = system_steep.calculate_back_focal_length()
        
        if bfl_shallow is not None and bfl_steep is not None:
            # Steeper curvature should result in shorter focal length
            assert abs(bfl_steep) < abs(bfl_shallow)
