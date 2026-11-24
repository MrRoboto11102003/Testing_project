import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem, SystemElement
from openlens.optimizer import create_doublet_optimizer, LensOptimizer


class TestCreateDoubletOptimizer:
    """Test suite for create_doublet_optimizer function"""

    @pytest.fixture
    def doublet_system(self):
        """Create a simple achromatic doublet optical system for testing"""
        crown = Lens(
            name="Crown",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-40.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

        flint = Lens(
            name="Flint",
            radius_of_curvature_1=-40.0,
            radius_of_curvature_2=-131.0,
            thickness=4.0,
            diameter=25.0,
            material="SF5",
        )

        system = OpticalSystem(
            name="Test Doublet",
            elements=[
                SystemElement(lens=crown, air_gap_after=0.1),
                SystemElement(lens=flint, air_gap_after=0.0),
            ],
        )
        return system

    def test_returns_lens_optimizer_instance(self, doublet_system):
        """Test that create_doublet_optimizer returns a LensOptimizer"""
        optimizer = create_doublet_optimizer(doublet_system)
        assert isinstance(optimizer, LensOptimizer)

    def test_optimizer_has_three_variables(self, doublet_system):
        """Test that optimizer has exactly 3 variables (R1 Crown, R2 Crown, R2 Flint)"""
        optimizer = create_doublet_optimizer(doublet_system)
        assert len(optimizer.variables) == 3

    def test_optimizer_has_two_targets(self, doublet_system):
        """Test that optimizer has exactly 2 targets (chromatic aberration and focal length)"""
        optimizer = create_doublet_optimizer(doublet_system)
        assert len(optimizer.targets) == 2

    def test_variable_names_correct(self, doublet_system):
        """Test that all variables are named as documented"""
        optimizer = create_doublet_optimizer(doublet_system)
        var_names = {var.name for var in optimizer.variables}
        expected_names = {"R1 Crown", "R2 Crown", "R2 Flint"}
        assert var_names == expected_names

    def test_target_metric_names_correct(self, doublet_system):
        """Test that targets have documented metric names"""
        optimizer = create_doublet_optimizer(doublet_system)
        target_metrics = {target.metric_name for target in optimizer.targets}
        expected_metrics = {"chromatic_aberration", "focal_length"}
        assert target_metrics == expected_metrics

    def test_chromatic_aberration_target_configured_for_minimization(self, doublet_system):
        """Test that chromatic aberration target is configured to minimize"""
        optimizer = create_doublet_optimizer(doublet_system)
        chromatic_target = next(
            (t for t in optimizer.targets if t.metric_name == "chromatic_aberration"), None
        )
        assert chromatic_target is not None
        assert chromatic_target.target_value == 0.0
        assert chromatic_target.target_type == "minimize"
        assert chromatic_target.weight == 1.0

    def test_focal_length_target_default_value_100mm(self, doublet_system):
        """Test that focal length target defaults to 100.0 mm"""
        optimizer = create_doublet_optimizer(doublet_system)
        fl_target = next(
            (t for t in optimizer.targets if t.metric_name == "focal_length"), None
        )
        assert fl_target is not None
        assert fl_target.target_value == 100.0
        assert fl_target.target_type == "target"

    def test_focal_length_target_weight_prioritized(self, doublet_system):
        """Test that focal length target has higher weight than chromatic aberration"""
        optimizer = create_doublet_optimizer(doublet_system)
        fl_target = next(
            (t for t in optimizer.targets if t.metric_name == "focal_length"), None
        )
        chromatic_target = next(
            (t for t in optimizer.targets if t.metric_name == "chromatic_aberration"), None
        )
        assert fl_target.weight > chromatic_target.weight

    def test_focal_length_target_custom_value(self, doublet_system):
        """Test that custom focal length target value is applied"""
        custom_fl = 50.0
        optimizer = create_doublet_optimizer(doublet_system, target_focal_length=custom_fl)
        fl_target = next(
            (t for t in optimizer.targets if t.metric_name == "focal_length"), None
        )
        assert fl_target.target_value == custom_fl

    def test_r2_crown_has_linked_target_to_flint_r1(self, doublet_system):
        """Test that R2 Crown has linked target to flint's R1 (cemented interface)"""
        optimizer = create_doublet_optimizer(doublet_system)
        r2_crown_var = next(
            (v for v in optimizer.variables if v.name == "R2 Crown"), None
        )
        assert r2_crown_var is not None
        assert r2_crown_var.linked_targets is not None
        assert len(r2_crown_var.linked_targets) > 0
        assert any(
            link[0] == 1 and link[1] == "radius_of_curvature_1"
            for link in r2_crown_var.linked_targets
        )

    def test_initial_values_from_system_state(self, doublet_system):
        """Test that initial variable values match the optical system state"""
        optimizer = create_doublet_optimizer(doublet_system)

        r1_crown_var = next((v for v in optimizer.variables if v.name == "R1 Crown"), None)
        assert (
            r1_crown_var.initial_value
            == doublet_system.elements[0].lens.radius_of_curvature_1
        )

        r2_crown_var = next((v for v in optimizer.variables if v.name == "R2 Crown"), None)
        assert (
            r2_crown_var.initial_value
            == doublet_system.elements[0].lens.radius_of_curvature_2
        )

        r2_flint_var = next((v for v in optimizer.variables if v.name == "R2 Flint"), None)
        assert (
            r2_flint_var.initial_value
            == doublet_system.elements[1].lens.radius_of_curvature_2
        )

    def test_all_variables_have_bounds(self, doublet_system):
        """Test that all variables have valid min/max bounds"""
        optimizer = create_doublet_optimizer(doublet_system)
        for var in optimizer.variables:
            assert var.min_bound is not None
            assert var.max_bound is not None
            assert var.min_bound < var.max_bound

    def test_r2_flint_bounds_negative(self, doublet_system):
        """Test that R2 Flint has negative bounds (rear concave surface per documentation)"""
        optimizer = create_doublet_optimizer(doublet_system)
        r2_flint_var = next((v for v in optimizer.variables if v.name == "R2 Flint"), None)
        assert r2_flint_var is not None
        assert r2_flint_var.min_bound < 0
        assert r2_flint_var.max_bound < 0
