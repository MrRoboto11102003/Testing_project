import pytest
import tempfile
import os
from unittest.mock import Mock
from pathlib import Path

try:
    from openlens.io.step_export import StepExporter
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


@pytest.fixture
def temp_step_file():
    """Create a temporary STEP file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.step', delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def mock_single_lens_system():
    """Create a mock single-lens system."""
    system = Mock()
    system.radius_of_curvature_1 = 50.0
    system.radius_of_curvature_2 = -50.0
    system.thickness = 5.0
    system.diameter = 20.0
    return system


@pytest.fixture
def mock_multi_element_system():
    """Create a mock multi-element optical system."""
    system = Mock()
    
    element1 = Mock()
    element1.lens = Mock()
    element2 = Mock()
    element2.lens = Mock()
    
    system.elements = [element1, element2]
    system.air_gaps = [5.0]
    return system


class TestStepExporterExportReturnValue:
    """Test return value structure and types."""
    
    def test_export_returns_dict_with_required_keys(self, temp_step_file, mock_single_lens_system):
        """Export returns dict containing 'shapes', 'skipped', 'skipped_parts'."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file)
        
        assert isinstance(result, dict)
        assert 'shapes' in result
        assert 'skipped' in result
        assert 'skipped_parts' in result
    
    def test_export_dict_values_have_correct_types(self, temp_step_file, mock_single_lens_system):
        """Return dict values must be int, int, and list respectively."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file)
        
        assert isinstance(result['shapes'], int)
        assert isinstance(result['skipped'], int)
        assert isinstance(result['skipped_parts'], list)
    
    def test_export_counts_are_non_negative(self, temp_step_file, mock_single_lens_system):
        """Shapes and skipped counts must be non-negative."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file)
        
        assert result['shapes'] >= 0
        assert result['skipped'] >= 0


class TestStepExporterBufferReset:
    """Test that entity buffer is reset on each export call."""
    
    def test_repeated_exports_reset_buffer(self, temp_step_file, mock_single_lens_system):
        """Calling export multiple times resets the buffer (no duplication)."""
        exporter = StepExporter(mock_single_lens_system)
        
        # First export
        exporter.export(temp_step_file)
        with open(temp_step_file, 'r') as f:
            content_first = f.read()
        
        # Second export
        exporter.export(temp_step_file)
        with open(temp_step_file, 'r') as f:
            content_second = f.read()
        
        # Same file size indicates no entity duplication
        assert len(content_first) == len(content_second)
    
    def test_buffer_state_reset_between_calls(self, temp_step_file, mock_single_lens_system):
        """Internal buffer state is reset between export calls."""
        exporter = StepExporter(mock_single_lens_system)
        
        exporter.export(temp_step_file)
        # After reset, ID counters should be zero
        assert exporter.axis_id == 0
        assert exporter.origin == 0


class TestStepExporterFileCreation:
    """Test that output file is created."""
    
    def test_export_creates_output_file(self, temp_step_file, mock_single_lens_system):
        """Export creates the output file."""
        exporter = StepExporter(mock_single_lens_system)
        exporter.export(temp_step_file)
        
        assert os.path.exists(temp_step_file)
        assert os.path.getsize(temp_step_file) > 0


class TestStepExporterHousingHandling:
    """Test housing parameter handling."""
    
    def test_export_without_housing_parameter(self, temp_step_file, mock_single_lens_system):
        """Export works without housing parameter."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file)
        
        assert result['skipped'] == 0
        assert result['shapes'] >= 1
    
    def test_export_with_none_housing(self, temp_step_file, mock_single_lens_system):
        """Export handles housing=None."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=None)
        
        assert result['skipped'] == 0
    
    def test_export_with_empty_housing_list(self, temp_step_file, mock_single_lens_system):
        """Export handles empty housing list."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=[])
        
        assert result['skipped'] == 0
    
    def test_export_with_valid_housing_part(self, temp_step_file, mock_single_lens_system):
        """Export processes valid housing parts with all required keys."""
        housing = [{
            'name': 'spacer',
            'z0': 0.0,
            'z1': 2.0,
            'r_inner': 5.0,
            'r_outer': 8.0
        }]
        
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=housing)
        
        assert result['shapes'] >= 1
        assert result['skipped'] == 0


class TestStepExporterHousingErrors:
    """Test error handling for malformed housing parts."""
    
    def test_housing_missing_required_key(self, temp_step_file, mock_single_lens_system):
        """Housing part missing required keys is skipped and reported."""
        housing = [{'name': 'incomplete', 'z0': 0.0}]  # Missing z1, r_inner, r_outer
        
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=housing)
        
        assert result['skipped'] >= 1
        assert 'incomplete' in result['skipped_parts']
    
    def test_housing_with_non_numeric_value(self, temp_step_file, mock_single_lens_system):
        """Housing part with non-numeric values is skipped and reported."""
        housing = [{
            'name': 'bad_housing',
            'z0': 'invalid_string',
            'z1': 2.0,
            'r_inner': 5.0,
            'r_outer': 8.0
        }]
        
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=housing)
        
        assert result['skipped'] >= 1
        assert 'bad_housing' in result['skipped_parts']
    
    def test_mixed_valid_invalid_housing_parts(self, temp_step_file, mock_single_lens_system):
        """Invalid housing parts are skipped while valid ones are processed."""
        housing = [
            {'name': 'valid', 'z0': 0.0, 'z1': 2.0, 'r_inner': 5.0, 'r_outer': 8.0},
            {'name': 'invalid'}  # Missing all coordinate/dimension fields
        ]
        
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file, housing=housing)
        
        assert 'invalid' in result['skipped_parts']
        assert result['skipped'] >= 1


class TestStepExporterSystemTypes:
    """Test handling of different system input types."""
    
    def test_single_lens_system(self, temp_step_file, mock_single_lens_system):
        """Single lens system (with radius_of_curvature_1) is exported."""
        exporter = StepExporter(mock_single_lens_system)
        result = exporter.export(temp_step_file)
        
        assert result['shapes'] >= 1
        assert result['skipped'] == 0
    
    def test_multi_element_system(self, temp_step_file, mock_multi_element_system):
        """Multi-element system (with elements and air_gaps) is handled."""
        exporter = StepExporter(mock_multi_element_system)
        result = exporter.export(temp_step_file)
        
        assert isinstance(result, dict)
        assert 'shapes' in result
        assert 'skipped' in result
