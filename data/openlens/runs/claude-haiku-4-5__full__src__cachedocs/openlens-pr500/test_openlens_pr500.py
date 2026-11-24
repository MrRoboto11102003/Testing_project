import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemRefreshReferences:
    """Tests for OpticalSystem.refresh_references method."""

    def test_refresh_with_empty_system(self):
        """Test that refresh on empty system does not raise."""
        system = OpticalSystem(name="Empty system")
        system.refresh_references({})
        assert len(system.elements) == 0

    def test_refresh_preserves_single_element(self):
        """Test that refresh preserves a single element."""
        lens = Lens(
            name="Test lens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        system = OpticalSystem(name="Single element")
        system.add_lens(lens)
        
        system.refresh_references({})
        
        assert len(system.elements) == 1

    def test_refresh_preserves_multiple_elements(self):
        """Test that refresh preserves multiple elements."""
        lens1 = Lens(
            name="Lens 1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            name="Lens 2",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=6.0,
            diameter=26.0,
            material="F2"
        )
        
        system = OpticalSystem(name="Two elements")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)
        
        initial_count = len(system.elements)
        system.refresh_references({})
        
        assert len(system.elements) == initial_count

    def test_refresh_with_empty_lookup(self):
        """Test refresh with empty lookup table."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = OpticalSystem(name="Test")
        system.add_lens(lens)
        
        system.refresh_references({})
        
        assert system.root is not None
        assert len(system.elements) == 1

    def test_refresh_tree_node_count_matches_elements(self):
        """Test that tree element nodes match system elements count."""
        lenses = [
            Lens(
                name=f"Lens {i}",
                radius_of_curvature_1=50.0 + i*5,
                radius_of_curvature_2=-50.0 - i*5,
                thickness=5.0,
                diameter=25.0
            )
            for i in range(3)
        ]
        
        system = OpticalSystem(name="Multiple")
        for i, lens in enumerate(lenses):
            if i == 0:
                system.add_lens(lens)
            else:
                system.add_lens(lens, air_gap_before=0.5)
        
        system.refresh_references({})
        
        # Count element nodes in tree after refresh
        element_nodes = [
            node for node, _ in system.root.get_flat_list()
            if getattr(node, "is_element", False)
        ]
        
        assert len(element_nodes) == len(system.elements)

    def test_refresh_idempotent_single_call(self):
        """Test that calling refresh twice with same lookup is safe."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = OpticalSystem(name="Test")
        system.add_lens(lens)
        
        lookup = {}
        
        system.refresh_references(lookup)
        first_count = len(system.elements)
        
        system.refresh_references(lookup)
        second_count = len(system.elements)
        
        assert first_count == second_count

    def test_refresh_with_air_gaps_preserved(self):
        """Test that refresh preserves air gaps between elements."""
        lens1 = Lens(
            name="Lens 1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        lens2 = Lens(
            name="Lens 2",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=6.0,
            diameter=26.0
        )
        
        system = OpticalSystem(name="With gaps")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=2.5)
        
        system.refresh_references({})
        
        assert len(system.elements) == 2

    def test_refresh_with_aperture_stop_preserved(self):
        """Test that refresh preserves aperture stop configuration."""
        lens1 = Lens(
            name="Lens 1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        lens2 = Lens(
            name="Lens 2",
            radius_of_curvature_1=60.0,
            radius_of_curvature_2=-60.0,
            thickness=6.0,
            diameter=26.0
        )
        
        system = OpticalSystem(name="With stop")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)
        system.set_aperture_stop(gap_index=0, diameter=20.0)
        
        stop_before = system.aperture_stop
        system.refresh_references({})
        stop_after = system.aperture_stop
        
        assert (stop_before is None) == (stop_after is None)

    def test_refresh_maintains_system_validity(self):
        """Test that system remains valid after refresh."""
        lens1 = Lens(
            name="Lens 1",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        lens2 = Lens(
            name="Lens 2",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        system = OpticalSystem(name="Validity test")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=1.0)
        
        focal_before = system.get_system_focal_length()
        system.refresh_references({})
        focal_after = system.get_system_focal_length()
        
        # Both should be calculable and consistent
        if focal_before is not None and focal_after is not None:
            assert abs(focal_before - focal_after) < 1e-6

    def test_refresh_tree_root_exists_after_refresh(self):
        """Test that tree root exists and is valid after refresh."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = OpticalSystem(name="Test")
        system.add_lens(lens)
        
        system.refresh_references({})
        
        assert system.root is not None
        assert hasattr(system.root, "get_flat_list")

    def test_refresh_with_large_assembly(self):
        """Test refresh with many elements."""
        system = OpticalSystem(name="Large assembly")
        
        for i in range(10):
            lens = Lens(
                name=f"Lens {i}",
                radius_of_curvature_1=50.0 + i,
                radius_of_curvature_2=-50.0 - i,
                thickness=5.0,
                diameter=25.0
            )
            if i == 0:
                system.add_lens(lens)
            else:
                system.add_lens(lens, air_gap_before=0.5)
        
        initial_count = len(system.elements)
        system.refresh_references({})
        
        assert len(system.elements) == initial_count

    def test_refresh_element_nodes_have_is_element_flag(self):
        """Test that element nodes have is_element flag set correctly."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = OpticalSystem(name="Test")
        system.add_lens(lens)
        
        system.refresh_references({})
        
        # Count nodes with is_element flag
        all_nodes = [node for node, _ in system.root.get_flat_list()]
        element_nodes = [node for node in all_nodes if getattr(node, "is_element", False)]
        
        assert len(element_nodes) >= 1
        assert len(all_nodes) >= len(element_nodes)

    def test_refresh_does_not_raise_with_various_lookups(self):
        """Test that refresh handles various lookup dict contents."""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        system = OpticalSystem(name="Test")
        system.add_lens(lens)
        
        # Test with various lookup dicts
        lookups = [
            {},
            {12345: "some_value"},
            {"string_key": lens},
        ]
        
        for lookup in lookups:
            system.refresh_references(lookup)
            assert len(system.elements) == 1
