import pytest
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens


class TestOpticalSystemRefreshReferences:
    """Tests for OpticalSystem.refresh_references"""

    def create_lens(self, name, thickness=5.0, radius1=50.0, radius2=-50.0):
        """Helper to create a simple test lens"""
        return Lens(
            name=name,
            radius_of_curvature_1=radius1,
            radius_of_curvature_2=radius2,
            thickness=thickness,
            diameter=25.0
        )

    def test_refresh_swaps_lens_object(self):
        """Test that matching lens_id replaces element.lens"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        original_lens_id = system.elements[0].lens_id
        original_lens = system.elements[0].lens
        
        lookup = {original_lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert system.elements[0].lens is lens_a_new
        assert system.elements[0].lens is not original_lens

    def test_refresh_preserves_element_list_and_lens_id(self):
        """Test that element list identity and lens_id are preserved"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        original_elements_id = id(system.elements)
        original_lens_id = system.elements[0].lens_id
        
        lookup = {original_lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert id(system.elements) == original_elements_id
        assert system.elements[0].lens_id == original_lens_id

    def test_refresh_syncs_tree_node_element_model(self):
        """Test that tree nodes are updated to point at new lens"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        original_lens_id = system.elements[0].lens_id
        element_nodes = [
            node for node, _ in system.root.get_flat_list() 
            if getattr(node, "is_element", False)
        ]
        assert len(element_nodes) >= 1
        
        lookup = {original_lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        element_nodes_after = [
            node for node, _ in system.root.get_flat_list() 
            if getattr(node, "is_element", False)
        ]
        assert element_nodes_after[0].element_model is lens_a_new
        assert element_nodes_after[0].element_model is system.elements[0].lens

    def test_refresh_recalculates_positions_with_thicker_lens(self):
        """Test position updates when swapping with thicker lens"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_b = self.create_lens("B", thickness=3.0)
        lens_a_new = self.create_lens("A_new", thickness=8.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        system.add_lens(lens_b, air_gap_before=10.0)
        
        assert pytest.approx(system.elements[0].position, abs=0.01) == 0.0
        assert pytest.approx(system.elements[1].position, abs=0.01) == 15.0
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert pytest.approx(system.elements[0].position, abs=0.01) == 0.0
        assert pytest.approx(system.elements[1].position, abs=0.01) == 18.0
        assert pytest.approx(system.air_gaps[0].position, abs=0.01) == 8.0

    def test_refresh_recalculates_positions_with_thinner_lens(self):
        """Test position updates when swapping with thinner lens"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_b = self.create_lens("B", thickness=3.0)
        lens_a_new = self.create_lens("A_new", thickness=2.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        system.add_lens(lens_b, air_gap_before=10.0)
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert pytest.approx(system.elements[1].position, abs=0.01) == 12.0

    def test_refresh_preserves_air_gap_thickness(self):
        """Test that air gap thicknesses remain unchanged"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_b = self.create_lens("B", thickness=3.0)
        lens_a_new = self.create_lens("A_new", thickness=10.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        system.add_lens(lens_b, air_gap_before=12.5)
        
        original_gap_thickness = system.air_gaps[0].thickness
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert pytest.approx(system.air_gaps[0].thickness, abs=0.01) == original_gap_thickness

    def test_refresh_partial_lookup_unmatched_elements_unchanged(self):
        """Test that unmatched elements keep original lens (identity check)"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_b = self.create_lens("B", thickness=3.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        system.add_lens(lens_b)
        
        original_lens_b = system.elements[1].lens
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert system.elements[0].lens is lens_a_new
        assert system.elements[1].lens is original_lens_b

    def test_refresh_empty_lookup_no_changes(self):
        """Test that empty lookup leaves lenses unchanged"""
        lens_a = self.create_lens("A", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        original_lens = system.elements[0].lens
        
        system.refresh_references({})
        
        assert system.elements[0].lens is original_lens

    def test_refresh_non_matching_lookup_no_changes(self):
        """Test that non-matching lookup IDs leave lenses unchanged"""
        lens_a = self.create_lens("A", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        original_lens = system.elements[0].lens
        
        lookup = {"nonexistent_id": self.create_lens("Other")}
        system.refresh_references(lookup)
        
        assert system.elements[0].lens is original_lens

    def test_refresh_empty_system_no_error(self):
        """Test that refresh on empty system is safe"""
        system = OpticalSystem()
        
        lookup = {"x": self.create_lens("X")}
        system.refresh_references(lookup)
        
        assert len(system.elements) == 0

    def test_refresh_preserves_decenter_on_elements(self):
        """Test that decenter values on elements are preserved"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a, decenter_y=1.5, decenter_z=2.0)
        
        original_decenter_y = system.elements[0].decenter_y
        original_decenter_z = system.elements[0].decenter_z
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert system.elements[0].decenter_y == original_decenter_y
        assert system.elements[0].decenter_z == original_decenter_z

    def test_refresh_anchor_position_preserved(self):
        """Test that first element anchor position is maintained"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=10.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a, air_gap_before=3.0)
        
        original_position = system.elements[0].position
        assert pytest.approx(original_position, abs=0.01) == 3.0
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert pytest.approx(system.elements[0].position, abs=0.01) == 3.0

    def test_refresh_cascading_positions_multiple_elements(self):
        """Test cascading position updates through multiple elements"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_b = self.create_lens("B", thickness=3.0)
        lens_c = self.create_lens("C", thickness=4.0)
        lens_a_new = self.create_lens("A_new", thickness=10.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        system.add_lens(lens_b, air_gap_before=2.0)
        system.add_lens(lens_c, air_gap_before=3.0)
        
        lookup = {system.elements[0].lens_id: lens_a_new}
        system.refresh_references(lookup)
        
        assert pytest.approx(system.elements[0].position, abs=0.01) == 0.0
        assert pytest.approx(system.elements[1].position, abs=0.01) == 15.0
        assert pytest.approx(system.elements[2].position, abs=0.01) == 22.0

    def test_refresh_invalid_lookup_type_raises_typeerror(self):
        """Test that non-dict lookup raises TypeError when system has elements"""
        lens_a = self.create_lens("A", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        with pytest.raises(TypeError):
            system.refresh_references(5)

    def test_refresh_returns_none(self):
        """Test that refresh_references returns None"""
        lens_a = self.create_lens("A", thickness=5.0)
        lens_a_new = self.create_lens("A_new", thickness=5.0)
        
        system = OpticalSystem()
        system.add_lens(lens_a)
        
        result = system.refresh_references({system.elements[0].lens_id: lens_a_new})
        
        assert result is None
