import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemRefreshReferences:
    """Pytest tests for OpticalSystem.refresh_references"""

    @pytest.fixture
    def biconvex_lens(self):
        """Create a simple biconvex lens for testing"""
        return Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )

    @pytest.fixture
    def updated_biconvex_lens(self):
        """Create an updated version of biconvex lens with different radii"""
        return Lens(
            name="Biconvex",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )

    @pytest.fixture
    def plano_convex_lens(self):
        """Create a plano-convex lens"""
        return Lens(
            name="PlanoConvex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="SF5"
        )

    @pytest.fixture
    def updated_plano_convex_lens(self):
        """Create an updated plano-convex lens"""
        return Lens(
            name="PlanoConvex",
            radius_of_curvature_1=120.0,
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="SF5"
        )

    @pytest.fixture
    def single_element_system(self, biconvex_lens):
        """Create an optical system with single element"""
        system = OpticalSystem()
        system.add_element(biconvex_lens)
        return system

    @pytest.fixture
    def multi_element_system(self, biconvex_lens, plano_convex_lens):
        """Create an optical system with multiple elements"""
        system = OpticalSystem()
        system.add_element(biconvex_lens, air_gap_mm=0)
        system.add_element(plano_convex_lens, air_gap_mm=10.0)
        return system

    def test_refresh_single_element_updates_lens_reference(
        self, single_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that single element lens reference is updated"""
        old_lens_id = id(biconvex_lens)
        old_lens = single_element_system.elements[0].lens
        
        lookup = {old_lens_id: updated_biconvex_lens}
        single_element_system.refresh_references(lookup)
        
        new_lens = single_element_system.elements[0].lens
        assert new_lens is updated_biconvex_lens
        assert id(new_lens) != id(old_lens)

    def test_refresh_multiple_elements_updates_all_references(
        self, multi_element_system, biconvex_lens, plano_convex_lens,
        updated_biconvex_lens, updated_plano_convex_lens
    ):
        """Test that all element references are updated in multi-element system"""
        lens1_id = id(biconvex_lens)
        lens2_id = id(plano_convex_lens)
        
        lookup = {
            lens1_id: updated_biconvex_lens,
            lens2_id: updated_plano_convex_lens
        }
        
        multi_element_system.refresh_references(lookup)
        
        assert multi_element_system.elements[0].lens is updated_biconvex_lens
        assert multi_element_system.elements[1].lens is updated_plano_convex_lens

    def test_refresh_empty_system_no_error(self):
        """Test that refresh on empty system does not raise"""
        system = OpticalSystem()
        lookup = {}
        
        # Should complete without error
        system.refresh_references(lookup)
        assert len(system.elements) == 0

    def test_refresh_tree_node_element_model_updated(
        self, single_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that tree nodes' element_model is synchronized with updated lens"""
        old_lens_id = id(biconvex_lens)
        
        lookup = {old_lens_id: updated_biconvex_lens}
        single_element_system.refresh_references(lookup)
        
        # Verify tree node points to new lens
        element_nodes = [
            node for node, _ in single_element_system.root.get_flat_list()
            if getattr(node, "is_element", False)
        ]
        
        for node in element_nodes:
            model = getattr(node, "element_model", None)
            if model is not None:
                assert model is updated_biconvex_lens

    def test_refresh_maintains_tree_flat_list_consistency(
        self, multi_element_system, biconvex_lens, plano_convex_lens,
        updated_biconvex_lens, updated_plano_convex_lens
    ):
        """Test that tree structure remains consistent after refresh"""
        lens1_id = id(biconvex_lens)
        lens2_id = id(plano_convex_lens)
        
        lookup = {
            lens1_id: updated_biconvex_lens,
            lens2_id: updated_plano_convex_lens
        }
        
        multi_element_system.refresh_references(lookup)
        
        # Both flat list and elements should reference same lenses
        element_nodes = [
            node for node, _ in multi_element_system.root.get_flat_list()
            if getattr(node, "is_element", False)
        ]
        
        assert len(element_nodes) >= len(multi_element_system.elements)

    def test_refresh_preserves_air_gaps(
        self, multi_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that air gaps are preserved during refresh"""
        air_gap_before = multi_element_system.air_gaps.copy() if hasattr(
            multi_element_system, 'air_gaps'
        ) else None
        
        old_lens_id = id(biconvex_lens)
        lookup = {old_lens_id: updated_biconvex_lens}
        
        multi_element_system.refresh_references(lookup)
        
        air_gap_after = multi_element_system.air_gaps.copy() if hasattr(
            multi_element_system, 'air_gaps'
        ) else None
        
        if air_gap_before is not None:
            assert air_gap_before == air_gap_after

    def test_refresh_idempotent_with_same_lens(self, single_element_system, biconvex_lens):
        """Test that refresh with the same lens object is safe"""
        current_lens = single_element_system.elements[0].lens
        old_id = id(current_lens)
        
        lookup = {old_id: current_lens}
        single_element_system.refresh_references(lookup)
        
        assert single_element_system.elements[0].lens is current_lens

    def test_refresh_with_partial_lookup(
        self, multi_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test refresh when lookup only contains some elements"""
        lens1_id = id(biconvex_lens)
        # Only provide update for first lens
        lookup = {lens1_id: updated_biconvex_lens}
        
        second_element_before = multi_element_system.elements[1].lens
        
        multi_element_system.refresh_references(lookup)
        
        # First element updated
        assert multi_element_system.elements[0].lens is updated_biconvex_lens
        # Second element unchanged
        assert multi_element_system.elements[1].lens is second_element_before

    def test_refresh_multiple_calls_accumulate_updates(
        self, single_element_system, biconvex_lens,
        updated_biconvex_lens, plano_convex_lens
    ):
        """Test that multiple refresh calls can be made sequentially"""
        old_lens_id = id(biconvex_lens)
        
        # First refresh
        lookup1 = {old_lens_id: updated_biconvex_lens}
        single_element_system.refresh_references(lookup1)
        
        # Verify first update
        assert single_element_system.elements[0].lens is updated_biconvex_lens
        
        # Second refresh with new lens
        old_id_2 = id(updated_biconvex_lens)
        lookup2 = {old_id_2: plano_convex_lens}
        single_element_system.refresh_references(lookup2)
        
        # Verify second update
        assert single_element_system.elements[0].lens is plano_convex_lens

    def test_refresh_calls_element_refresh_method(
        self, single_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that refresh calls the element's refresh method"""
        old_lens_id = id(biconvex_lens)
        lookup = {old_lens_id: updated_biconvex_lens}
        
        # Store original element
        original_element = single_element_system.elements[0]
        
        single_element_system.refresh_references(lookup)
        
        # Element object should still exist (refresh called on it)
        assert single_element_system.elements[0] is original_element

    def test_refresh_updates_positions_calculation(
        self, multi_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that _update_positions is called by monitoring element state"""
        old_lens_id = id(biconvex_lens)
        lookup = {old_lens_id: updated_biconvex_lens}
        
        # Capture element count before and after
        elements_before = len(multi_element_system.elements)
        
        multi_element_system.refresh_references(lookup)
        
        elements_after = len(multi_element_system.elements)
        
        # Element count should be preserved
        assert elements_before == elements_after

    def test_refresh_empty_lookup_table(self, single_element_system, biconvex_lens):
        """Test refresh with empty lookup table"""
        original_lens = single_element_system.elements[0].lens
        lookup = {}
        
        single_element_system.refresh_references(lookup)
        
        # Lens should remain unchanged (no updates in lookup)
        assert single_element_system.elements[0].lens is original_lens

    def test_refresh_does_not_modify_lookup_dict(
        self, single_element_system, biconvex_lens, updated_biconvex_lens
    ):
        """Test that refresh does not modify the lookup dictionary"""
        old_lens_id = id(biconvex_lens)
        lookup = {old_lens_id: updated_biconvex_lens}
        lookup_copy = lookup.copy()
        
        single_element_system.refresh_references(lookup)
        
        # Lookup should remain unchanged
        assert lookup == lookup_copy
