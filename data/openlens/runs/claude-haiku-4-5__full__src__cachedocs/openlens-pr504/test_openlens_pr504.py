import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.optimizer import create_doublet_optimizer, LensOptimizer


class TestCreateDoubletOptimizer:
    """Test suite for create_doublet_optimizer function."""

    @pytest.fixture
    def doublet_system(self):
        """Create a two-element doublet system (crown + flint)."""
        crown = Lens(
            name="Crown",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        flint = Lens(
            name="Flint",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=-100.0,
            thickness=4.0,
            diameter=25.0,
            material="SF11",
        )
        system = OpticalSystem(name="Test Doublet")
        system.add_lens(crown)
        system.add_lens(flint, air_gap_before=0.5)
        return system

    def test_returns_lens_optimizer_instance(self, doublet_system):
        """Test function returns LensOptimizer."""
        result = create_doublet_optimizer(doublet_system)
        assert isinstance(result, LensOptimizer)

    def test_default_target_focal_length_is_100mm(self, doublet_system):
        """Test default focal length target is 100.0 mm."""
        optimizer = create_doublet_optimizer(doublet_system)
        focal_targets = [t for t in optimizer.targets if t.metric == "focal_length"]
        assert len(focal_targets) == 1
        assert focal_targets[0].target_value == 100.0

    def test_custom_target_focal_length_applied(self, doublet_system):
        """Test custom target focal length overrides default."""
        for target_fl in [50.0, 75.0, 150.0]:
            optimizer = create_doublet_optimizer(doublet_system, target_focal_length=target_fl)
            focal_targets = [t for t in optimizer.targets if t.metric == "focal_length"]
            assert focal_targets[0].target_value == target_fl

    def test_optimizer_has_exactly_three_variables(self, doublet_system):
        """Test optimizer defines 3 curvature variables."""
        optimizer = create_doublet_optimizer(doublet_system)
        assert len(optimizer.variables) == 3

    def test_optimizer_has_exactly_two_targets(self, doublet_system):
        """Test optimizer has chromatic aberration and focal length targets."""
        optimizer = create_doublet_optimizer(doublet_system)
        assert len(optimizer.targets) == 2

    def test_variable_names_for_crown_and_flint(self, doublet_system):
        """Test variables cover crown R1, crown R2, and flint R2."""
        optimizer = create_doublet_optimizer(doublet_system)
        var_names = {v.name for v in optimizer.variables}
        assert var_names == {"R1 Crown", "R2 Crown", "R2 Flint"}

    def test_variables_target_correct_elements(self, doublet_system):
        """Test variables reference element 0 (crown) and element 1 (flint)."""
        optimizer = create_doublet_optimizer(doublet_system)
        var_map = {v.name: v for v in optimizer.variables}
        
        assert var_map["R1 Crown"].element_index == 0
        assert var_map["R2 Crown"].element_index == 0
        assert var_map["R2 Flint"].element_index == 1

    def test_variables_target_radius_parameters(self, doublet_system):
        """Test variables operate on radius_of_curvature_1 and radius_of_curvature_2."""
        optimizer = create_doublet_optimizer(doublet_system)
        var_map = {v.name: v for v in optimizer.variables}
        
        assert var_map["R1 Crown"].parameter == "radius_of_curvature_1"
        assert var_map["R2 Crown"].parameter == "radius_of_curvature_2"
        assert var_map["R2 Flint"].parameter == "radius_of_curvature_2"

    def test_r2_crown_links_to_flint_r1_for_cemented_interface(self, doublet_system):
        """Test R2 Crown (cemented surface) is linked to Flint R1."""
        optimizer = create_doublet_optimizer(doublet_system)
        r2_crown = next(v for v in optimizer.variables if v.name == "R2 Crown")
        
        assert r2_crown.linked_targets is not None
        assert len(r2_crown.linked_targets) == 1
        element_idx, param_name = r2_crown.linked_targets[0]
        assert element_idx == 1
        assert param_name == "radius_of_curvature_1"

    def test_chromatic_aberration_target_configured_to_minimize(self, doublet_system):
        """Test chromatic aberration is minimized toward zero."""
        optimizer = create_doublet_optimizer(doublet_system)
        ca_target = next(t for t in optimizer.targets if t.metric == "chromatic_aberration")
        
        assert ca_target.target_type == "minimize"
        assert ca_target.target_value == 0.0
        assert ca_target.weight == 1.0

    def test_focal_length_target_has_higher_weight(self, doublet_system):
        """Test focal length is prioritized (weight 100x chromatic aberration)."""
        optimizer = create_doublet_optimizer(doublet_system)
        targets_by_metric = {t.metric: t for t in optimizer.targets}
        
        assert targets_by_metric["focal_length"].weight == 100.0
        assert targets_by_metric["chromatic_aberration"].weight == 1.0

    def test_all_variables_have_valid_bounds(self, doublet_system):
        """Test all variables have lower < upper bounds."""
        optimizer = create_doublet_optimizer(doublet_system)
        
        for var in optimizer.variables:
            assert isinstance(var.lower_bound, (int, float))
            assert isinstance(var.upper_bound, (int, float))
            assert var.lower_bound < var.upper_bound
            assert var.step > 0
