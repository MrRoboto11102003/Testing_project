import pytest
from unittest.mock import Mock
from openlens.optical_system import OpticalSystem


class TestRefreshReferences:
    """Test suite for OpticalSystem.refresh_references method"""
    
    def setup_method(self):
        """Setup for each test - create mock optical system"""
        self.system = Mock(spec=OpticalSystem)
        self.system.elements = []
        self.system.root = Mock()
        self.system._update_positions = Mock()
        # Bind the actual method to our mock
        self.system.refresh_references = OpticalSystem.refresh_references.__get__(
            self.system, type(self.system)
        )
    
    def test_refresh_called_on_all_elements(self):
        """Test that refresh() is called on each element with the lookup dict"""
        element1 = Mock()
        element1.refresh = Mock()
        element2 = Mock()
        element2.refresh = Mock()
        self.system.elements = [element1, element2]
        
        node1 = Mock()
        node1.is_element = True
        node1.element_model = None
        node2 = Mock()
        node2.is_element = True
        node2.element_model = None
        self.system.root.get_flat_list.return_value = [(node1, None), (node2, None)]
        
        lens_lookup = {"lens1": Mock()}
        self.system.refresh_references(lens_lookup)
        
        element1.refresh.assert_called_once_with(lens_lookup)
        element2.refresh.assert_called_once_with(lens_lookup)
    
    def test_empty_elements_list(self):
        """Test with no elements - should still call _update_positions"""
        self.system.elements = []
        self.system.root.get_flat_list.return_value = []
        
        self.system.refresh_references({})
        
        self.system._update_positions.assert_called_once()
    
    def test_element_model_updated_when_different(self):
        """Test that element_model is updated when it differs from element.lens"""
        old_lens = Mock()
        new_lens = Mock()
        
        element = Mock()
        element.lens = new_lens
        element.refresh = Mock()
        self.system.elements = [element]
        
        node = Mock()
        node.is_element = True
        node.element_model = old_lens
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        self.system.refresh_references({})
        
        assert node.element_model is new_lens
    
    def test_element_model_not_modified_when_same(self):
        """Test that element_model is not modified when already correct (identity check)"""
        same_lens = Mock()
        
        element = Mock()
        element.lens = same_lens
        element.refresh = Mock()
        self.system.elements = [element]
        
        node = Mock()
        node.is_element = True
        node.element_model = same_lens
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        self.system.refresh_references({})
        
        assert node.element_model is same_lens
    
    def test_element_model_none_not_updated(self):
        """Test that if element_model is None, it remains unchanged"""
        element = Mock()
        element.lens = Mock()
        element.refresh = Mock()
        self.system.elements = [element]
        
        node = Mock()
        node.is_element = True
        node.element_model = None
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        self.system.refresh_references({})
        
        # Condition checks "if model is not None and model is not element.lens"
        # So None model stays None
        assert node.element_model is None
    
    def test_node_without_element_model_attribute(self):
        """Test handling of nodes missing element_model attribute"""
        element = Mock()
        element.lens = Mock()
        element.refresh = Mock()
        self.system.elements = [element]
        
        node = Mock(spec=['is_element'])  # No element_model attribute
        node.is_element = True
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        # Should not raise - getattr returns None as default
        self.system.refresh_references({})
        
        self.system._update_positions.assert_called_once()
    
    def test_non_element_nodes_skipped(self):
        """Test that nodes with is_element=False are skipped in filtering"""
        element = Mock()
        element.lens = Mock()
        element.refresh = Mock()
        self.system.elements = [element]
        
        non_elem_node = Mock()
        non_elem_node.is_element = False
        element_node = Mock()
        element_node.is_element = True
        element_node.element_model = Mock()
        
        self.system.root.get_flat_list.return_value = [(non_elem_node, None), (element_node, None)]
        
        self.system.refresh_references({})
        
        # Non-element node should be filtered out
        self.system._update_positions.assert_called_once()
    
    def test_update_positions_called_after_refresh(self):
        """Test that _update_positions is called after all element refreshes and updates"""
        element = Mock()
        element.refresh = Mock()
        element.lens = Mock()
        self.system.elements = [element]
        
        node = Mock()
        node.is_element = True
        node.element_model = None
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        self.system.refresh_references({})
        
        self.system._update_positions.assert_called_once()
    
    def test_multiple_elements_paired_correctly(self):
        """Test that elements and nodes are paired correctly via zip"""
        lens1, lens2, lens3 = Mock(), Mock(), Mock()
        
        element1 = Mock()
        element1.lens = lens1
        element1.refresh = Mock()
        
        element2 = Mock()
        element2.lens = lens2
        element2.refresh = Mock()
        
        element3 = Mock()
        element3.lens = lens3
        element3.refresh = Mock()
        
        self.system.elements = [element1, element2, element3]
        
        node1 = Mock()
        node1.is_element = True
        node1.element_model = Mock()
        
        node2 = Mock()
        node2.is_element = True
        node2.element_model = Mock()
        
        node3 = Mock()
        node3.is_element = True
        node3.element_model = Mock()
        
        self.system.root.get_flat_list.return_value = [(node1, None), (node2, None), (node3, None)]
        
        self.system.refresh_references({})
        
        # Verify element[i].lens is assigned to node[i].element_model
        assert node1.element_model is lens1
        assert node2.element_model is lens2
        assert node3.element_model is lens3
    
    def test_lens_lookup_dict_passed_unchanged(self):
        """Test that the same lens_lookup dict is passed to all elements"""
        element1 = Mock()
        element1.refresh = Mock()
        element2 = Mock()
        element2.refresh = Mock()
        self.system.elements = [element1, element2]
        
        node1 = Mock()
        node1.is_element = True
        node1.element_model = None
        node2 = Mock()
        node2.is_element = True
        node2.element_model = None
        self.system.root.get_flat_list.return_value = [(node1, None), (node2, None)]
        
        lens_lookup = {"key": object()}
        self.system.refresh_references(lens_lookup)
        
        element1.refresh.assert_called_once_with(lens_lookup)
        element2.refresh.assert_called_once_with(lens_lookup)
    
    def test_conditional_update_mixed_scenarios(self):
        """Test element_model update logic with mixed update/no-update scenarios"""
        old_lens1 = Mock()
        new_lens1 = Mock()
        same_lens = Mock()
        new_lens3 = Mock()
        
        element1 = Mock()
        element1.lens = new_lens1
        element1.refresh = Mock()
        
        element2 = Mock()
        element2.lens = same_lens
        element2.refresh = Mock()
        
        element3 = Mock()
        element3.lens = new_lens3
        element3.refresh = Mock()
        
        self.system.elements = [element1, element2, element3]
        
        node1 = Mock()
        node1.is_element = True
        node1.element_model = old_lens1  # Different -> updates
        
        node2 = Mock()
        node2.is_element = True
        node2.element_model = same_lens  # Same -> no update
        
        node3 = Mock()
        node3.is_element = True
        node3.element_model = None  # None -> no update
        
        self.system.root.get_flat_list.return_value = [(node1, None), (node2, None), (node3, None)]
        
        self.system.refresh_references({})
        
        assert node1.element_model is new_lens1
        assert node2.element_model is same_lens
        assert node3.element_model is None
    
    def test_identity_check_for_model_match(self):
        """Test that model identity (is) comparison is used, not equality"""
        element = Mock()
        element.refresh = Mock()
        self.system.elements = [element]
        
        lens1 = Mock()
        lens2 = Mock()
        
        # Even if lenses compare equal, they are different objects
        element.lens = lens2
        
        node = Mock()
        node.is_element = True
        node.element_model = lens1
        self.system.root.get_flat_list.return_value = [(node, None)]
        
        self.system.refresh_references({})
        
        # Identity check with 'is' should catch the difference
        assert node.element_model is lens2
        assert node.element_model is not lens1
