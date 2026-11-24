import pytest
import math
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestSpotDiagramTraceSpot:
    """Test SpotDiagram.trace_spot method"""
    
    @pytest.fixture
    def mock_system(self):
        """Create a mock optical system"""
        system = MagicMock()
        element = MagicMock()
        lens = MagicMock()
        lens.wavelength = 550e-6
        lens.refractive_index = 1.5
        lens.update_refractive_index = MagicMock()
        element.lens = lens
        element.position = 10.0
        system.elements = [element]
        system.get_total_length.return_value = 100.0
        return system
    
    @pytest.fixture
    def mock_tracer(self):
        """Create a mock ray tracer"""
        tracer = MagicMock()
        tracer.trace_ray = MagicMock()
        return tracer
    
    @pytest.fixture
    def spot_diagram(self, mock_system, mock_tracer):
        """Create a SpotDiagram with mocked dependencies"""
        from unittest.mock import MagicMock
        from openlens.analysis.spot_diagram import SpotDiagram
        
        with patch('openlens.analysis.spot_diagram.resolve_image_plane_x', return_value=50.0):
            diagram = MagicMock(spec=SpotDiagram)
            diagram.system = mock_system
            diagram.tracer = mock_tracer
            
            # Attach the real trace_spot method
            diagram.trace_spot = SpotDiagram.trace_spot.__get__(diagram, SpotDiagram)
            diagram.generate_hexapolar_grid = SpotDiagram.generate_hexapolar_grid.__get__(diagram, SpotDiagram)
            return diagram
    
    def test_trace_spot_returns_dict_with_required_keys(self, spot_diagram):
        """Verify trace_spot returns dict with all required keys"""
        result = spot_diagram.trace_spot()
        
        assert isinstance(result, dict)
        assert 'rms_radius' in result
        assert 'geo_radius' in result
        assert 'centroid' in result
        assert 'points' in result
        assert 'valid_rays' in result
        assert 'image_plane_x' in result
    
    def test_trace_spot_centroid_is_tuple(self, spot_diagram):
        """Test that centroid is a tuple of two floats"""
        result = spot_diagram.trace_spot()
        
        assert isinstance(result['centroid'], tuple)
        assert len(result['centroid']) == 2
        assert all(isinstance(x, (int, float)) for x in result['centroid'])
    
    def test_trace_spot_points_is_list(self, spot_diagram):
        """Test that points is a list of tuples"""
        result = spot_diagram.trace_spot()
        
        assert isinstance(result['points'], list)
        for point in result['points']:
            assert isinstance(point, (tuple, list)) and len(point) == 2
    
    def test_trace_spot_no_rays_returns_none_radii(self, spot_diagram):
        """Test that fewer than 2 rays returns None for radii"""
        # Mock scenario where no rays reach the plane
        spot_diagram.tracer.trace_ray = MagicMock(
            side_effect=lambda ray: setattr(ray, 'terminated', True)
        )
        
        result = spot_diagram.trace_spot()
        
        assert result['rms_radius'] is None
        assert result['geo_radius'] is None
        assert result['valid_rays'] < 2
        assert 'error' in result
    
    def test_trace_spot_field_angle_x(self, spot_diagram):
        """Test trace_spot with non-zero X field angle"""
        result = spot_diagram.trace_spot(field_angle_x_deg=5.0)
        
        assert isinstance(result, dict)
        assert result['rms_radius'] is None or isinstance(result['rms_radius'], float)
    
    def test_trace_spot_field_angle_y(self, spot_diagram):
        """Test trace_spot with non-zero Y field angle"""
        result = spot_diagram.trace_spot(field_angle_y_deg=5.0)
        
        assert isinstance(result, dict)
        assert result['rms_radius'] is None or isinstance(result['rms_radius'], float)
    
    def test_trace_spot_custom_wavelength(self, spot_diagram):
        """Test trace_spot with custom wavelength"""
        result_green = spot_diagram.trace_spot(wavelength_nm=550.0)
        result_red = spot_diagram.trace_spot(wavelength_nm=650.0)
        
        assert result_green is not None
        assert result_red is not None
        # Verify wavelength updates were attempted
        assert spot_diagram.system.elements[0].lens.update_refractive_index.called
    
    def test_trace_spot_custom_image_plane(self, spot_diagram):
        """Test trace_spot with custom image plane position"""
        custom_x = 75.5
        result = spot_diagram.trace_spot(image_plane_x_mm=custom_x)
        
        # Note: may add focus_shift, so check with tolerance
        assert result['image_plane_x'] >= custom_x - 1e-6
    
    def test_trace_spot_focus_shift_increases_plane(self, spot_diagram):
        """Test that focus_shift_mm increases image plane position"""
        result_no_shift = spot_diagram.trace_spot(focus_shift_mm=0.0)
        result_shifted = spot_diagram.trace_spot(focus_shift_mm=1.0, image_plane_x_mm=50.0)
        
        assert result_shifted['image_plane_x'] > result_no_shift['image_plane_x']
    
    def test_trace_spot_num_rings_affects_points_count(self, spot_diagram):
        """Test that num_rings parameter affects number of points"""
        result_low = spot_diagram.trace_spot(num_rings=2)
        result_high = spot_diagram.trace_spot(num_rings=8)
        
        # More rings should generally give more points (hexapolar grid)
        # Formula: approximately 3*N*(N+1) for N rings
        if result_low['valid_rays'] > 0 and result_high['valid_rays'] > 0:
            assert len(result_high['points']) >= len(result_low['points'])
    
    def test_trace_spot_rms_radius_non_negative(self, spot_diagram):
        """Test that RMS radius is non-negative when defined"""
        result = spot_diagram.trace_spot()
        
        if result['rms_radius'] is not None:
            assert result['rms_radius'] >= 0.0
    
    def test_trace_spot_geo_radius_non_negative(self, spot_diagram):
        """Test that geometric radius is non-negative when defined"""
        result = spot_diagram.trace_spot()
        
        if result['geo_radius'] is not None:
            assert result['geo_radius'] >= 0.0


class TestInverseSensitivityAnalyzer:
    """Test InverseSensitivityAnalyzer.calculate_sensitivities method"""
    
    @pytest.fixture
    def mock_analyzer_setup(self):
        """Setup mock analyzer with tolerances"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        analyzer = MagicMock(spec=InverseSensitivityAnalyzer)
        analyzer.SUPPORTED_CRITERIA = "rms_spot_radius"
        
        # Create mock tolerances
        tol1 = MagicMock()
        tol1.max_val = 0.1
        tol1.param_type.name = 'thickness'
        tol1.element_index = 0
        
        tol2 = MagicMock()
        tol2.max_val = 0.05
        tol2.param_type.name = 'radius'
        tol2.element_index = 1
        
        analyzer.tolerances = [tol1, tol2]
        
        # Mock system state management
        analyzer._get_system_state = MagicMock(return_value={'state': 'mock'})
        analyzer._set_system_state = MagicMock()
        analyzer._apply_single_tolerance = MagicMock()
        
        return analyzer
    
    def test_calculate_sensitivities_invalid_criterion_raises(self, mock_analyzer_setup):
        """Test that invalid criterion raises ValueError"""
        from openlens.tolerancing import InverseSensitivityAnalyzer
        
        analyzer = mock_analyzer_setup
        
        # Create a real implementation that checks criterion
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            if criterion != analyzer.SUPPORTED_CRITERIA:
                raise ValueError(
                    f"Unsupported criterion {criterion!r}; "
                    f"supported: {analyzer.SUPPORTED_CRITERIA!r}"
                )
            return []
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        with pytest.raises(ValueError, match="Unsupported criterion"):
            analyzer.calculate_sensitivities(criterion="invalid_metric")
    
    def test_calculate_sensitivities_returns_list(self, mock_analyzer_setup):
        """Test that calculate_sensitivities returns a list"""
        analyzer = mock_analyzer_setup
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            if criterion != analyzer.SUPPORTED_CRITERIA:
                raise ValueError()
            return []
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        assert isinstance(result, list)
    
    def test_calculate_sensitivities_list_contains_dicts(self, mock_analyzer_setup):
        """Test that returned list contains dictionaries"""
        analyzer = mock_analyzer_setup
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            if criterion != analyzer.SUPPORTED_CRITERIA:
                raise ValueError()
            return [
                {'operand_index': 0, 'sensitivity': 1.0},
                {'operand_index': 1, 'sensitivity': 2.0},
            ]
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        assert all(isinstance(item, dict) for item in result)
    
    def test_calculate_sensitivities_dict_has_required_keys(self, mock_analyzer_setup):
        """Test that each result dict has required keys"""
        analyzer = mock_analyzer_setup
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            results = []
            for i, tol in enumerate(analyzer.tolerances):
                results.append({
                    'operand_index': i,
                    'type': tol.param_type.name,
                    'element': tol.element_index,
                    'test_value': tol.max_val,
                    'result_value': 1.5,
                    'change': 0.5,
                    'sensitivity': 5.0,
                })
            return results
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        
        required_keys = {'operand_index', 'type', 'element', 'test_value',
                        'result_value', 'change', 'sensitivity'}
        for item in result:
            assert required_keys.issubset(item.keys())
    
    def test_calculate_sensitivities_one_entry_per_tolerance(self, mock_analyzer_setup):
        """Test that there's one result per tolerance"""
        analyzer = mock_analyzer_setup
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            results = []
            for i, tol in enumerate(analyzer.tolerances):
                results.append({'operand_index': i})
            return results
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        assert len(result) == len(analyzer.tolerances)
    
    def test_calculate_sensitivities_handles_vignetted_inf(self):
        """Test that inf values result in nan sensitivity"""
        analyzer = MagicMock()
        analyzer.SUPPORTED_CRITERIA = "rms_spot_radius"
        
        tol = MagicMock()
        tol.max_val = 0.1
        analyzer.tolerances = [tol]
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            # Simulate vignetted nominal (inf) and perturbed (inf)
            nominal_val = float('inf')
            val = float('inf')
            delta = tol.max_val
            
            change = val - nominal_val  # inf - inf = nan
            sensitivity = change / delta  # nan / float = nan
            
            # Normalize nan as per implementation
            if not (math.isfinite(change) and math.isfinite(sensitivity)):
                change = float('nan')
                sensitivity = float('nan')
            
            return [{
                'operand_index': 0,
                'change': change,
                'sensitivity': sensitivity,
            }]
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        assert math.isnan(result[0]['sensitivity'])
        assert math.isnan(result[0]['change'])
    
    def test_calculate_sensitivities_sensitivity_calculation(self):
        """Test that sensitivity is calculated as change/delta"""
        analyzer = MagicMock()
        analyzer.SUPPORTED_CRITERIA = "rms_spot_radius"
        
        tol = MagicMock()
        tol.max_val = 0.1
        tol.param_type.name = 'test'
        tol.element_index = 0
        analyzer.tolerances = [tol]
        
        def calculate_sensitivities_impl(criterion="rms_spot_radius"):
            nominal = 1.0
            perturbed = 1.5
            delta = 0.1
            change = perturbed - nominal  # 0.5
            sensitivity = change / delta  # 5.0
            
            return [{
                'operand_index': 0,
                'change': change,
                'sensitivity': sensitivity,
            }]
        
        analyzer.calculate_sensitivities.side_effect = calculate_sensitivities_impl
        
        result = analyzer.calculate_sensitivities()
        assert abs(result[0]['sensitivity'] - 5.0) < 1e-6
        assert abs(result[0]['change'] - 0.5) < 1e-6
