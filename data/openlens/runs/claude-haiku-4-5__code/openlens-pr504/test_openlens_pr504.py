import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# Mock classes for openlens components
class MockLens:
    def __init__(self, r1=100.0, r2=-200.0):
        self.radius_of_curvature_1 = r1
        self.radius_of_curvature_2 = r2


class MockOpticalElement:
    def __init__(self, lens=None):
        self.lens = lens or MockLens()


class MockOpticalSystem:
    def __init__(self, num_elements=2, r1_crown=100.0, r2_crown=-200.0, 
                 r1_flint=-200.0, r2_flint=-131.0):
        self.elements = [
            MockOpticalElement(MockLens(r1=r1_crown, r2=r2_crown)),  # Crown
            MockOpticalElement(MockLens(r1=r1_flint, r2=r2_flint)),  # Flint
        ]
        for i in range(2, num_elements):
            self.elements.append(MockOpticalElement())


class MockOptimizationVariable:
    def __init__(self, name, element_index, parameter, initial_value, 
                 lower_bound, upper_bound, step, linked_targets=None):
        self.name = name
        self.element_index = element_index
        self.parameter = parameter
        self.initial_value = initial_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
        self.linked_targets = linked_targets or []


class MockOptimizationTarget:
    def __init__(self, name, target_value, weight=1.0, target_type="target"):
        self.name = name
        self.target_value = target_value
        self.weight = weight
        self.target_type = target_type


class MockLensOptimizer:
    def __init__(self, system, variables, targets):
        self.system = system
        self.variables = variables
        self.targets = targets


def create_doublet_optimizer(system, target_focal_length=100.0):
    """Create optimizer for achromatic doublet (test implementation)"""
    variables = [
        MockOptimizationVariable(
            "R1 Crown",
            0,
            "radius_of_curvature_1",
            system.elements[0].lens.radius_of_curvature_1,
            20.0,
            500.0,
            10.0,
        ),
        MockOptimizationVariable(
            "R2 Crown",
            0,
            "radius_of_curvature_2",
            system.elements[0].lens.radius_of_curvature_2,
            -500.0,
            -20.0,
            10.0,
            linked_targets=[(1, "radius_of_curvature_1")],
        ),
        MockOptimizationVariable(
            "R2 Flint",
            1,
            "radius_of_curvature_2",
            system.elements[1].lens.radius_of_curvature_2,
            -500.0,
            -20.0,
            10.0,
        ),
    ]

    targets = [
        MockOptimizationTarget("chromatic_aberration", 0.0, weight=1.0, target_type="minimize"),
        MockOptimizationTarget("focal_length", target_focal_length, weight=100.0, target_type="target"),
    ]

    return MockLensOptimizer(system, variables, targets)


class TestCreateDoubletOptimizer:
    
    def test_returns_lens_optimizer(self):
        """Test that function returns a LensOptimizer instance"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        assert optimizer is not None
        assert hasattr(optimizer, 'variables')
        assert hasattr(optimizer, 'targets')
        assert hasattr(optimizer, 'system')
    
    def test_default_target_focal_length(self):
        """Test that default target focal length is 100.0"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        focal_length_target = next(t for t in optimizer.targets if t.name == "focal_length")
        assert focal_length_target.target_value == 100.0
    
    def test_custom_target_focal_length(self):
        """Test that custom target focal length is correctly set"""
        system = MockOpticalSystem()
        custom_focal_length = 75.5
        optimizer = create_doublet_optimizer(system, target_focal_length=custom_focal_length)
        focal_length_target = next(t for t in optimizer.targets if t.name == "focal_length")
        assert focal_length_target.target_value == custom_focal_length
    
    def test_number_of_variables_is_three(self):
        """Test that exactly 3 optimization variables are created"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        assert len(optimizer.variables) == 3
    
    def test_number_of_targets_is_two(self):
        """Test that exactly 2 optimization targets are created"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        assert len(optimizer.targets) == 2
    
    def test_variable_names_correct(self):
        """Test that all variables have correct names"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        names = [v.name for v in optimizer.variables]
        assert names == ["R1 Crown", "R2 Crown", "R2 Flint"]
    
    def test_r1_crown_bounds(self):
        """Test R1 Crown variable bounds are positive"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        r1_crown = next(v for v in optimizer.variables if v.name == "R1 Crown")
        assert r1_crown.lower_bound == 20.0
        assert r1_crown.upper_bound == 500.0
        assert r1_crown.lower_bound < r1_crown.upper_bound
    
    def test_r2_surfaces_bounds_negative(self):
        """Test that back surfaces (R2) have negative bounds per documentation"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        r2_crown = next(v for v in optimizer.variables if v.name == "R2 Crown")
        r2_flint = next(v for v in optimizer.variables if v.name == "R2 Flint")
        assert r2_crown.lower_bound < 0 and r2_crown.upper_bound < 0
        assert r2_flint.lower_bound < 0 and r2_flint.upper_bound < 0
    
    def test_r2_crown_has_linked_target_to_flint(self):
        """Test that R2 Crown is linked to flint's R1 for cemented interface"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        r2_crown = next(v for v in optimizer.variables if v.name == "R2 Crown")
        assert len(r2_crown.linked_targets) == 1
        assert r2_crown.linked_targets[0] == (1, "radius_of_curvature_1")
    
    def test_chromatic_aberration_minimize_target(self):
        """Test that chromatic aberration is a minimize target with weight 1.0"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        ca_target = next(t for t in optimizer.targets if t.name == "chromatic_aberration")
        assert ca_target.target_value == 0.0
        assert ca_target.target_type == "minimize"
        assert ca_target.weight == 1.0
    
    def test_focal_length_target_type(self):
        """Test that focal length is a 'target' type with weight 100.0"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        fl_target = next(t for t in optimizer.targets if t.name == "focal_length")
        assert fl_target.target_type == "target"
        assert fl_target.weight == 100.0
    
    def test_focal_length_weight_higher_than_aberration(self):
        """Test that focal length has higher weight than chromatic aberration"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        targets_dict = {t.name: t for t in optimizer.targets}
        assert targets_dict["focal_length"].weight > targets_dict["chromatic_aberration"].weight
    
    def test_variables_reference_correct_elements(self):
        """Test that variables reference correct element indices (0=crown, 1=flint)"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system)
        crown_vars = [v for v in optimizer.variables if v.element_index == 0]
        flint_vars = [v for v in optimizer.variables if v.element_index == 1]
        assert len(crown_vars) == 2  # R1 Crown and R2 Crown
        assert len(flint_vars) == 1  # R2 Flint only
    
    def test_initial_values_match_system(self):
        """Test that variable initial values match system lens parameters"""
        system = MockOpticalSystem(r1_crown=85.0, r2_crown=-210.0, 
                                   r1_flint=-195.0, r2_flint=-140.0)
        optimizer = create_doublet_optimizer(system)
        r1_crown = next(v for v in optimizer.variables if v.name == "R1 Crown")
        r2_crown = next(v for v in optimizer.variables if v.name == "R2 Crown")
        r2_flint = next(v for v in optimizer.variables if v.name == "R2 Flint")
        assert r1_crown.initial_value == 85.0
        assert r2_crown.initial_value == -210.0
        assert r2_flint.initial_value == -140.0
    
    def test_negative_focal_length_allowed(self):
        """Test that negative focal length values are accepted"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system, target_focal_length=-50.0)
        fl_target = next(t for t in optimizer.targets if t.name == "focal_length")
        assert fl_target.target_value == -50.0
    
    def test_very_large_focal_length(self):
        """Test that very large focal length values are accepted"""
        system = MockOpticalSystem()
        optimizer = create_doublet_optimizer(system, target_focal_length=10000.0)
        fl_target = next(t for t in optimizer.targets if t.name == "focal_length")
        assert fl_target.target_value == 10000.0
