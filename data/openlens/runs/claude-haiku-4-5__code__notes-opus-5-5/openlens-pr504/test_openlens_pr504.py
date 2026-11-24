import pytest
from openlens.optical_system import OpticalSystem
from openlens.lens import Lens
from openlens.optimizer import create_doublet_optimizer


class TestCreateDoubletOptimizer:
    """Tests for openlens.optimizer.create_doublet_optimizer function."""
    
    @pytest.fixture
    def doublet_system(self):
        """Create a two-element doublet system for testing."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0))
        system.add_lens(Lens(radius_of_curvature_1=-50.0, radius_of_curvature_2=-150.0), air_gap_before=0.0)
        return system
    
    def test_returns_optimizer_with_system_reference(self, doublet_system):
        """Test that function returns optimizer with same system reference."""
        opt = create_doublet_optimizer(doublet_system)
        assert opt.system is doublet_system
        assert hasattr(opt, 'variables')
        assert hasattr(opt, 'targets')
        assert hasattr(opt, 'merit_function')
    
    def test_default_target_focal_length(self, doublet_system):
        """Test that default target_focal_length is 100.0."""
        opt = create_doublet_optimizer(doublet_system)
        assert opt.targets[1].target_value == 100.0
    
    def test_custom_target_focal_length(self, doublet_system):
        """Test that custom target_focal_length is respected."""
        opt = create_doublet_optimizer(doublet_system, target_focal_length=80.0)
        assert opt.targets[1].target_value == 80.0
    
    def test_variables_configuration(self, doublet_system):
        """Test that 3 variables with correct names and parameters are created."""
        opt = create_doublet_optimizer(doublet_system)
        assert len(opt.variables) == 3
        
        assert opt.variables[0].name == "R1 Crown"
        assert opt.variables[0].parameter == "radius_of_curvature_1"
        assert opt.variables[0].element_index == 0
        
        assert opt.variables[1].name == "R2 Crown"
        assert opt.variables[1].parameter == "radius_of_curvature_2"
        assert opt.variables[1].element_index == 0
        
        assert opt.variables[2].name == "R2 Flint"
        assert opt.variables[2].parameter == "radius_of_curvature_2"
        assert opt.variables[2].element_index == 1
    
    def test_variable_current_values_match_lens_properties(self, doublet_system):
        """Test that current_value matches lens getter values."""
        opt = create_doublet_optimizer(doublet_system)
        
        assert opt.variables[0].current_value == pytest.approx(
            doublet_system.elements[0].lens.radius_of_curvature_1
        )
        assert opt.variables[1].current_value == pytest.approx(
            doublet_system.elements[0].lens.radius_of_curvature_2
        )
        assert opt.variables[2].current_value == pytest.approx(
            doublet_system.elements[1].lens.radius_of_curvature_2
        )
    
    def test_variable_bounds_and_steps(self, doublet_system):
        """Test that bounds and steps are correct for all variables."""
        opt = create_doublet_optimizer(doublet_system)
        
        # R1 Crown: positive bounds
        assert opt.variables[0].min == 20.0
        assert opt.variables[0].max == 500.0
        assert opt.variables[0].step == 10.0
        
        # R2 Crown: negative bounds
        assert opt.variables[1].min == -500.0
        assert opt.variables[1].max == -20.0
        assert opt.variables[1].step == 10.0
        
        # R2 Flint: negative bounds
        assert opt.variables[2].min == -500.0
        assert opt.variables[2].max == -20.0
        assert opt.variables[2].step == 10.0
    
    def test_linked_targets(self, doublet_system):
        """Test that R2 Crown links to flint front surface and others do not."""
        opt = create_doublet_optimizer(doublet_system)
        
        assert opt.variables[0].linked_targets == []
        assert opt.variables[1].linked_targets == [(1, "radius_of_curvature_1")]
        assert opt.variables[2].linked_targets == []
    
    def test_targets_configuration(self, doublet_system):
        """Test that 2 targets with correct configuration are created."""
        opt = create_doublet_optimizer(doublet_system)
        assert len(opt.targets) == 2
        
        # Chromatic aberration target
        assert opt.targets[0].name == "chromatic_aberration"
        assert opt.targets[0].target_value == 0.0
        assert opt.targets[0].weight == 1.0
        assert opt.targets[0].target_type == "minimize"
        
        # Focal length target
        assert opt.targets[1].name == "focal_length"
        assert opt.targets[1].weight == 100.0
        assert opt.targets[1].target_type == "target"
    
    def test_merit_function_references(self, doublet_system):
        """Test that merit_function correctly references system and targets."""
        opt = create_doublet_optimizer(doublet_system)
        assert opt.merit_function.system is doublet_system
        assert opt.merit_function.targets is opt.targets
        assert opt._merit_cache == {}
    
    def test_empty_system_raises_indexerror(self):
        """Test that empty system raises IndexError."""
        system = OpticalSystem()
        with pytest.raises(IndexError):
            create_doublet_optimizer(system)
    
    def test_single_element_system_raises_indexerror(self):
        """Test that one-element system raises IndexError."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0))
        with pytest.raises(IndexError):
            create_doublet_optimizer(system)
    
    def test_system_with_extra_elements_accepted(self):
        """Test that system with more than 2 elements is accepted."""
        system = OpticalSystem()
        system.add_lens(Lens(radius_of_curvature_1=60.0, radius_of_curvature_2=-50.0))
        system.add_lens(Lens(radius_of_curvature_1=-50.0, radius_of_curvature_2=-150.0), air_gap_before=0.0)
        system.add_lens(Lens(radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0), air_gap_before=5.0)
        opt = create_doublet_optimizer(system)
        assert len(opt.variables) == 3
        assert len(opt.targets) == 2
    
    def test_extreme_target_focal_lengths(self, doublet_system):
        """Test that extreme focal length values are accepted without validation."""
        opt_neg = create_doublet_optimizer(doublet_system, target_focal_length=-1000.0)
        assert opt_neg.targets[1].target_value == -1000.0
        
        opt_zero = create_doublet_optimizer(doublet_system, target_focal_length=0.0)
        assert opt_zero.targets[1].target_value == 0.0
