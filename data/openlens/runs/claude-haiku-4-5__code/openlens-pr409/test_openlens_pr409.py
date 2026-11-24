import pytest
from unittest.mock import Mock, MagicMock
from openlens.optimizer import LensOptimizer, OptimizationResult, OptimizationVariable


@pytest.fixture
def optimizer_one_var():
    """Create LensOptimizer with one variable."""
    opt = LensOptimizer()
    var = Mock(spec=OptimizationVariable)
    var.name = "radius"
    var.current_value = 1.0
    var.step_size = 0.1
    var.is_valid = Mock(return_value=True)
    opt.variables = [var]
    return opt


@pytest.fixture
def optimizer_two_vars():
    """Create LensOptimizer with two variables."""
    opt = LensOptimizer()
    var1 = Mock(spec=OptimizationVariable)
    var1.name = "radius"
    var1.current_value = 1.0
    var1.step_size = 0.1
    var1.is_valid = Mock(return_value=True)
    
    var2 = Mock(spec=OptimizationVariable)
    var2.name = "thickness"
    var2.current_value = 2.0
    var2.step_size = 0.2
    var2.is_valid = Mock(return_value=True)
    
    opt.variables = [var1, var2]
    return opt


def test_optimize_simplex_no_variables_raises_error():
    """Test that optimize_simplex raises ValueError when there are no variables."""
    opt = LensOptimizer()
    opt.variables = []
    
    with pytest.raises(ValueError, match="at least one variable"):
        opt.optimize_simplex()


def test_optimize_simplex_start_wrong_length_raises_error(optimizer_one_var):
    """Test that start parameter with wrong length raises ValueError."""
    with pytest.raises(ValueError, match="start has 2 value.*but there are 1"):
        optimizer_one_var.optimize_simplex(start=[1.0, 2.0])


def test_optimize_simplex_start_empty_raises_error(optimizer_one_var):
    """Test that empty start parameter raises ValueError."""
    with pytest.raises(ValueError, match="start has 0 value.*but there are 1"):
        optimizer_one_var.optimize_simplex(start=[])


def test_optimize_simplex_uses_start_values(optimizer_one_var):
    """Test that provided start values are used instead of current_value."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[2.0, 2.1])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(start=[5.0], max_iterations=1)
    
    # First evaluation should use start value
    first_call = optimizer_one_var._evaluate_design.call_args_list[0]
    assert first_call[0][0][0] == 5.0


def test_optimize_simplex_uses_current_value_as_default(optimizer_one_var):
    """Test that current_value is used when start is None."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=1)
    
    # First evaluation should use current_value
    first_call = optimizer_one_var._evaluate_design.call_args_list[0]
    assert first_call[0][0][0] == 1.0


def test_optimize_simplex_callback_invoked(optimizer_one_var):
    """Test that callback is called on each iteration with iteration count, merit, and vertex."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1, 0.95])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    callback = Mock()
    result = optimizer_one_var.optimize_simplex(callback=callback, max_iterations=5)
    
    assert callback.called
    # Check callback signature: (iteration, merit, vertex)
    call_args = callback.call_args_list[0][0]
    assert len(call_args) == 3
    assert isinstance(call_args[0], int)  # iteration
    assert isinstance(call_args[1], (int, float))  # merit
    assert isinstance(call_args[2], list)  # vertex


def test_optimize_simplex_multiple_variables(optimizer_two_vars):
    """Test that simplex creates n+1 vertices for n variables."""
    optimizer_two_vars._evaluate_design = Mock(side_effect=[1.0, 1.1, 1.2, 0.9])
    optimizer_two_vars._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_two_vars.optimize_simplex(max_iterations=1)
    
    # For 2 variables, simplex should have 3 vertices (n+1)
    assert optimizer_two_vars._evaluate_design.call_count >= 3


def test_optimize_simplex_converges_early(optimizer_one_var):
    """Test that optimization stops early when convergence criteria are met."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1, 0.95, 0.94, 0.93])
    # Converge on iteration 2
    optimizer_one_var._merit_range_converged = Mock(side_effect=[False, True])
    
    result = optimizer_one_var.optimize_simplex(max_iterations=10)
    
    assert result.iterations <= 3


def test_optimize_simplex_respects_max_iterations(optimizer_one_var):
    """Test that optimization stops at max_iterations even if not converged."""
    merits = [1.0 - 0.001*i for i in range(200)]
    optimizer_one_var._evaluate_design = Mock(side_effect=merits)
    optimizer_one_var._merit_range_converged = Mock(return_value=False)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=3)
    
    assert result.iterations <= 3


def test_optimize_simplex_asphyxiation_detection(optimizer_one_var):
    """Test that asphyxiation (vertices pinned outside bounds) is detected and fails."""
    optimizer_one_var.variables[0].is_valid = Mock(return_value=False)
    optimizer_one_var._evaluate_design = Mock(return_value=1.0)
    optimizer_one_var._merit_range_converged = Mock(return_value=False)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=5)
    
    assert result.success is False
    assert "stuck outside" in result.message


def test_optimize_simplex_result_structure(optimizer_one_var):
    """Test that OptimizationResult contains all required fields."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=1)
    
    assert isinstance(result, OptimizationResult)
    assert hasattr(result, 'success')
    assert hasattr(result, 'iterations')
    assert hasattr(result, 'initial_merit')
    assert hasattr(result, 'final_merit')
    assert hasattr(result, 'improvement')
    assert hasattr(result, 'optimized_system')
    assert hasattr(result, 'variable_history')
    assert hasattr(result, 'merit_history')
    assert hasattr(result, 'best_values')
    assert hasattr(result, 'message')


def test_optimize_simplex_variable_history_tracking(optimizer_one_var):
    """Test that variable history is tracked correctly."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=1)
    
    assert isinstance(result.variable_history, list)
    assert len(result.variable_history) >= 1
    assert "radius" in result.variable_history[0]


def test_optimize_simplex_merit_history_tracking(optimizer_one_var):
    """Test that merit history is tracked correctly."""
    optimizer_one_var._evaluate_design = Mock(side_effect=[1.0, 1.1, 0.95])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=1)
    
    assert isinstance(result.merit_history, list)
    assert len(result.merit_history) >= 1
    assert all(isinstance(m, (int, float)) for m in result.merit_history)
    assert result.merit_history[0] == 1.0


def test_optimize_simplex_initial_merit_recorded(optimizer_one_var):
    """Test that initial merit is correctly stored in result."""
    initial = 2.5
    optimizer_one_var._evaluate_design = Mock(side_effect=[initial, 2.4, 2.3])
    optimizer_one_var._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_one_var.optimize_simplex(max_iterations=1)
    
    assert result.initial_merit == initial


def test_optimize_simplex_best_values_returned(optimizer_two_vars):
    """Test that best_values are returned with correct dimensions."""
    optimizer_two_vars._evaluate_design = Mock(side_effect=[1.0, 1.1, 1.2])
    optimizer_two_vars._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_two_vars.optimize_simplex(max_iterations=1)
    
    assert isinstance(result.best_values, list)
    assert len(result.best_values) == 2  # Two variables


def test_optimize_simplex_vertices_perturbation(optimizer_two_vars):
    """Test that simplex vertices are created by perturbing each variable."""
    vertices = []
    def capture_eval(v):
        vertices.append(v[:])
        return 1.0
    
    optimizer_two_vars._evaluate_design = side_effect_function(capture_eval)
    optimizer_two_vars._merit_range_converged = Mock(return_value=True)
    
    result = optimizer_two_vars.optimize_simplex(max_iterations=1)
    
    # Verify at least 3 vertices evaluated
    assert optimizer_two_vars._evaluate_design.call_count >= 3


def side_effect_function(func):
    """Helper to create a mock with custom side effect that also appends."""
    def wrapper(v):
        return func(v)
    return Mock(side_effect=wrapper)
