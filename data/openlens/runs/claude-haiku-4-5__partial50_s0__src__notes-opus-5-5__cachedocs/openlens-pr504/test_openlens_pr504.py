import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import (
    create_doublet_optimizer,
    OptimizationVariable,
    OptimizationTarget,
)
from openlens.constants import MIN_EDGE_THICKNESS


class TestCreateDoubletOptimizer:
    """Tests for openlens.optimizer.create_doublet_optimizer."""

    @pytest.fixture
    def basic_doublet(self):
        """Create a basic two-element optical system (crown + flint)."""
        system = OpticalSystem()
        system.add_lens(
            Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0)
        )
        system.add_lens(
            Lens(radius_of_curvature_1=-50.0, radius_of_curvature_2=-150.0),
            air_gap_before=0.0,
        )
        return system

    # === Basic structure and references ===

    def test_returns_lens_optimizer_instance(self, basic_doublet):
        """Test that the function returns a LensOptimizer."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert hasattr(optimizer, "system")
        assert hasattr(optimizer, "variables")
        assert hasattr(optimizer, "targets")
        assert hasattr(optimizer, "merit_function")

    def test_system_is_same_object_not_copied(self, basic_doublet):
        """Test that optimizer holds a reference to the same system (not a copy)."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert optimizer.system is basic_doublet

    def test_merit_function_references_same_system(self, basic_doublet):
        """Test that merit function system is the same object."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert optimizer.merit_function.system is basic_doublet

    def test_merit_function_targets_reference_same_list(self, basic_doublet):
        """Test that merit function targets reference the same list as optimizer."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert optimizer.merit_function.targets is optimizer.targets

    # === Variables validation ===

    def test_exactly_three_variables(self, basic_doublet):
        """Test that exactly 3 optimization variables are created."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert len(optimizer.variables) == 3

    def test_variable_r1_crown_all_properties(self, basic_doublet):
        """Test R1 Crown variable has correct name, bounds, element, and parameter."""
        optimizer = create_doublet_optimizer(basic_doublet)
        var = optimizer.variables[0]

        assert var.name == "R1 Crown"
        assert var.element_index == 0
        assert var.parameter == "radius_of_curvature_1"
        assert var.current_value == basic_doublet.elements[0].lens.radius_of_curvature_1
        assert var.min_value == 20.0
        assert var.max_value == 500.0
        assert var.step == 10.0
        assert var.linked_targets == []

    def test_variable_r2_crown_with_cemented_interface_link(self, basic_doublet):
        """Test R2 Crown variable has correct link to flint's front surface."""
        optimizer = create_doublet_optimizer(basic_doublet)
        var = optimizer.variables[1]

        assert var.name == "R2 Crown"
        assert var.element_index == 0
        assert var.parameter == "radius_of_curvature_2"
        assert var.current_value == basic_doublet.elements[0].lens.radius_of_curvature_2
        assert var.min_value == -500.0
        assert var.max_value == -20.0
        assert var.step == 10.0
        # Must link crown's back surface to flint's front surface
        assert len(var.linked_targets) == 1
        assert var.linked_targets[0] == (1, "radius_of_curvature_1")

    def test_variable_r2_flint_all_properties(self, basic_doublet):
        """Test R2 Flint variable has correct bounds and properties."""
        optimizer = create_doublet_optimizer(basic_doublet)
        var = optimizer.variables[2]

        assert var.name == "R2 Flint"
        assert var.element_index == 1
        assert var.parameter == "radius_of_curvature_2"
        assert var.current_value == basic_doublet.elements[1].lens.radius_of_curvature_2
        assert var.min_value == -500.0
        assert var.max_value == -20.0
        assert var.step == 10.0
        assert var.linked_targets == []

    # === Targets validation ===

    def test_exactly_two_targets(self, basic_doublet):
        """Test that exactly 2 optimization targets are created."""
        optimizer = create_doublet_optimizer(basic_doublet)
        assert len(optimizer.targets) == 2

    def test_target_chromatic_aberration_minimize(self, basic_doublet):
        """Test chromatic aberration target is set to minimize with correct weight."""
        optimizer = create_doublet_optimizer(basic_doublet)
        target = optimizer.targets[0]

        assert target.name == "chromatic_aberration"
        assert target.target_value == 0.0
        assert target.weight == 1.0
        assert target.target_type == "minimize"

    def test_target_focal_length_default_100(self, basic_doublet):
        """Test focal length target with default value of 100.0."""
        optimizer = create_doublet_optimizer(basic_doublet)
        target = optimizer.targets[1]

        assert target.name == "focal_length"
        assert target.target_value == 100.0
        assert target.weight == 100.0
        assert target.target_type == "target"

    def test_target_focal_length_custom_value(self, basic_doublet):
        """Test focal length target respects custom target value."""
        custom_fl = 80.0
        optimizer = create_doublet_optimizer(basic_doublet, target_focal_length=custom_fl)
        target = optimizer.targets[1]

        assert target.name == "focal_length"
        assert target.target_value == 80.0
        assert target.weight == 100.0
        assert target.target_type == "target"

    # === Error handling ===

    def test_empty_system_raises_index_error(self):
        """Test that an empty OpticalSystem raises IndexError."""
        system = OpticalSystem()
        with pytest.raises(IndexError):
            create_doublet_optimizer(system)

    def test_single_element_system_raises_index_error(self):
        """Test that a system with only one element raises IndexError."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0))
        with pytest.raises(IndexError):
            create_doublet_optimizer(system)

    # === Edge cases ===

    def test_system_with_more_than_two_elements_accepted(self):
        """Test that systems with >2 elements are accepted (only 0 and 1 referenced)."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0))
        system.add_lens(
            Lens(radius_of_curvature_1=-50.0, radius_of_curvature_2=-150.0),
            air_gap_before=0.0,
        )
        system.add_lens(
            Lens(radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0),
            air_gap_before=0.0,
        )

        optimizer = create_doublet_optimizer(system)
        assert len(optimizer.variables) == 3
        # Verify only elements 0 and 1 are used
        element_indices = {var.element_index for var in optimizer.variables}
        assert element_indices == {0, 1}

    def test_default_constraints_in_merit_function(self, basic_doublet):
        """Test that merit function has correct default constraint values."""
        optimizer = create_doublet_optimizer(basic_doublet)
        constraints = optimizer.merit_function.constraints

        assert constraints["min_center_thickness"] == 1.0
        assert constraints["max_center_thickness"] == 100.0
        assert constraints["min_edge_thickness"] == MIN_EDGE_THICKNESS
        assert constraints["min_air_gap"] == 0.1
        assert constraints["min_edge_clearance"] == 0.1
