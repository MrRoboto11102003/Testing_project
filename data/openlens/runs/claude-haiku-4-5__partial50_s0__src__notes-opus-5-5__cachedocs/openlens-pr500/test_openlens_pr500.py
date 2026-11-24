import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemRefreshReferences:
    """Test OpticalSystem.refresh_references method"""

    @pytest.fixture
    def lens_thin(self):
        """Create a lens with thickness 5"""
        return Lens(
            name="Thin Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )

    @pytest.fixture
    def lens_thick(self):
        """Create a lens with thickness 8"""
        return Lens(
            name="Thick Lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=8.0,
            diameter=25.0,
        )

    @pytest.fixture
    def lens_medium(self):
        """Create a lens with thickness 3"""
        return Lens(
            name="Medium Lens",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=3.0,
            diameter=20.0,
        )

    def test_refresh_swaps_lens_with_valid_lookup(self, lens_thin, lens_thick):
        """Test that refresh_references swaps lens with new one from lookup"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}

        system.refresh_references(lookup)

        assert system.elements[0].lens is lens_thick
        assert system.elements[0].lens is not lens_thin

    def test_refresh_tree_node_synced_to_new_lens(self, lens_thin, lens_thick):
        """Test that tree nodes are synced to point to new lens"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # Tree node should point to the same lens as the element
        assert system.root.children[0].element_model is system.elements[0].lens
        assert system.root.children[0].element_model is lens_thick

    def test_refresh_positions_updated_with_thicker_lens(self, lens_thin, lens_thick, lens_medium):
        """Test that element positions are recalculated when lens thickness changes"""
        system = OpticalSystem()
        system.add_lens(lens_thin)  # pos 0, thickness 5
        system.add_lens(lens_medium, air_gap_before=10)  # pos 15, thickness 3

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # First element stays at anchor position 0
        assert system.elements[0].position == pytest.approx(0.0)
        # Second element: (8 + 10) = 18
        assert system.elements[1].position == pytest.approx(18.0)

    def test_refresh_air_gap_position_updated(self, lens_thin, lens_thick, lens_medium):
        """Test that air gap positions are updated when lens thickness changes"""
        system = OpticalSystem()
        system.add_lens(lens_thin)  # thickness 5
        system.add_lens(lens_medium, air_gap_before=10)

        gap_position_before = system.air_gaps[0].position
        assert gap_position_before == pytest.approx(5.0)

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # Gap position should shift to match new lens thickness
        assert system.air_gaps[0].position == pytest.approx(8.0)

    def test_refresh_air_gap_thickness_preserved(self, lens_thin, lens_thick, lens_medium):
        """Test that air gap thicknesses are NOT modified during refresh"""
        system = OpticalSystem()
        system.add_lens(lens_thin)
        system.add_lens(lens_medium, air_gap_before=10)

        gap_thickness_before = system.air_gaps[0].thickness

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # Gap thickness must remain unchanged
        assert system.air_gaps[0].thickness == gap_thickness_before
        assert system.air_gaps[0].thickness == pytest.approx(10.0)

    def test_refresh_lens_id_field_unchanged(self, lens_thin, lens_thick):
        """Test that element.lens_id is NOT updated after refresh"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        lens_id_before = system.elements[0].lens_id
        lookup = {lens_id_before: lens_thick}
        system.refresh_references(lookup)

        # lens_id field should remain unchanged even though lens object was swapped
        assert system.elements[0].lens_id == lens_id_before

    def test_refresh_empty_lookup_no_swap(self, lens_thin):
        """Test that empty lookup does not swap lenses"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        original_lens = system.elements[0].lens
        system.refresh_references({})

        assert system.elements[0].lens is original_lens

    def test_refresh_none_lookup_no_swap(self, lens_thin):
        """Test that None lookup does not swap lenses"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        original_lens = system.elements[0].lens
        system.refresh_references(None)

        assert system.elements[0].lens is original_lens

    def test_refresh_partial_lookup_only_matching_swapped(self, lens_thin, lens_thick, lens_medium):
        """Test that only elements matching lookup keys are swapped"""
        system = OpticalSystem()
        system.add_lens(lens_thin)
        system.add_lens(lens_medium)

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # First element should be swapped
        assert system.elements[0].lens is lens_thick
        # Second element should remain unchanged
        assert system.elements[1].lens is lens_medium

    def test_refresh_non_matching_lookup_keys_ignored(self, lens_thin, lens_thick):
        """Test that non-matching keys in lookup are silently ignored"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        original_lens = system.elements[0].lens
        lookup = {"non_existent_id": lens_thick}
        system.refresh_references(lookup)

        assert system.elements[0].lens is original_lens

    def test_refresh_multiple_elements_cascading_positions(self, lens_thin, lens_thick, lens_medium):
        """Test position cascading with multiple elements and lens swap"""
        system = OpticalSystem()
        system.add_lens(lens_thin)  # pos 0, thickness 5
        system.add_lens(lens_medium, air_gap_before=5)  # pos 10, thickness 3
        system.add_lens(lens_thin, air_gap_before=3)  # pos 16, thickness 5

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # First stays at 0, second: 8+5=13, third: 13+3+3=19
        assert system.elements[0].position == pytest.approx(0.0)
        assert system.elements[1].position == pytest.approx(13.0)
        assert system.elements[2].position == pytest.approx(19.0)

    def test_refresh_empty_system_no_error(self):
        """Test that refresh on empty system does not raise error"""
        system = OpticalSystem()
        new_lens = Lens(
            name="test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        # Should not raise
        system.refresh_references({"any_id": new_lens})

    def test_refresh_total_length_updated(self, lens_thin, lens_thick, lens_medium):
        """Test that total system length reflects new lens thickness"""
        system = OpticalSystem()
        system.add_lens(lens_thin)  # 5
        system.add_lens(lens_medium, air_gap_before=10)  # +10+3 = 18 total

        element_id = system.elements[0].lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        # New total: 8+10+3 = 21
        assert system.get_total_length() == pytest.approx(21.0)

    def test_refresh_preserves_element_object_identity(self, lens_thin, lens_thick):
        """Test that element objects remain the same (only .lens attribute changes)"""
        system = OpticalSystem()
        system.add_lens(lens_thin)

        element_before = system.elements[0]
        element_id = element_before.lens_id
        lookup = {element_id: lens_thick}
        system.refresh_references(lookup)

        element_after = system.elements[0]

        # Same element object, only lens attribute changed
        assert element_before is element_after
        assert element_after.lens is lens_thick
        assert element_after.lens is not lens_thin
