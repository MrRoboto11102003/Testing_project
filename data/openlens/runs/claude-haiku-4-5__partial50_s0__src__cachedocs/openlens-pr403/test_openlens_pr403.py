import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from openlens.io.step_export import StepExporter


class TestStepExporterExport:
    """Tests for StepExporter.export() method."""

    @pytest.fixture
    def temp_step_file(self):
        """Provide a temporary file path for STEP exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_export.step")

    @pytest.fixture
    def mock_lens(self):
        """Create a minimal mock lens with required attributes."""
        lens = Mock()
        lens.radius_of_curvature_1 = 50.0
        lens.radius_of_curvature_2 = -50.0
        lens.thickness = 5.0
        lens.diameter = 25.0
        lens.name = "test_lens"
        lens.conic_constant_1 = 0.0
        lens.conic_constant_2 = 0.0
        return lens

    @pytest.fixture
    def single_lens_system(self, mock_lens):
        """Mock a single lens system (no elements/air_gaps attributes)."""
        system = Mock(spec=[
            'radius_of_curvature_1', 'radius_of_curvature_2',
            'thickness', 'diameter', 'name', 'conic_constant_1', 'conic_constant_2'
        ])
        system.radius_of_curvature_1 = 50.0
        system.radius_of_curvature_2 = -50.0
        system.thickness = 5.0
        system.diameter = 25.0
        system.name = "single_lens"
        system.conic_constant_1 = 0.0
        system.conic_constant_2 = 0.0
        return system

    @pytest.fixture
    def multi_element_system(self, mock_lens):
        """Mock a multi-element optical system."""
        elem1 = Mock()
        elem1.lens = mock_lens
        elem2 = Mock()
        elem2.lens = mock_lens

        system = Mock()
        system.elements = [elem1, elem2]
        system.air_gaps = [0.1, 0.2]
        system.name = "multi_element"
        system._element_placements = Mock(return_value=[(0.0,), (5.1,)])
        return system

    @pytest.fixture
    def exporter_single(self, single_lens_system):
        """Create StepExporter for single lens system."""
        return StepExporter(single_lens_system)

    @pytest.fixture
    def exporter_multi(self, multi_element_system):
        """Create StepExporter for multi-element system."""
        return StepExporter(multi_element_system)

    # --- Return Value Structure Tests ---

    def test_export_returns_dict(self, exporter_single, temp_step_file):
        """Test that export() returns a dictionary."""
        result = exporter_single.export(temp_step_file)
        assert isinstance(result, dict)

    def test_export_dict_contains_required_keys(self, exporter_single, temp_step_file):
        """Test that return dict has 'shapes', 'skipped', and 'skipped_parts' keys."""
        result = exporter_single.export(temp_step_file)
        assert "shapes" in result
        assert "skipped" in result
        assert "skipped_parts" in result

    def test_export_dict_values_have_correct_types(self, exporter_single, temp_step_file):
        """Test that dict values have correct types: int, int, list."""
        result = exporter_single.export(temp_step_file)
        assert isinstance(result["shapes"], int)
        assert isinstance(result["skipped"], int)
        assert isinstance(result["skipped_parts"], list)

    def test_export_skipped_count_matches_skipped_parts_list(self, exporter_single, temp_step_file):
        """Test that skipped count equals length of skipped_parts list."""
        result = exporter_single.export(temp_step_file)
        assert result["skipped"] == len(result["skipped_parts"])

    # --- File Creation Tests ---

    def test_export_creates_output_file(self, exporter_single, temp_step_file):
        """Test that export() creates an output file at the specified path."""
        exporter_single.export(temp_step_file)
        assert os.path.exists(temp_step_file)

    def test_export_file_contains_data(self, exporter_single, temp_step_file):
        """Test that the created STEP file contains content."""
        exporter_single.export(temp_step_file)
        with open(temp_step_file, 'r') as f:
            content = f.read()
        assert len(content) > 0

    # --- Repeated Call Behavior Tests ---

    def test_export_safe_to_call_repeatedly_no_duplication(self, exporter_single, temp_step_file):
        """Test that calling export() twice doesn't duplicate entities (buffer reset)."""
        # First export
        result1 = exporter_single.export(temp_step_file)
        with open(temp_step_file, 'r') as f:
            content1 = f.read()
        
        # Second export
        result2 = exporter_single.export(temp_step_file)
        with open(temp_step_file, 'r') as f:
            content2 = f.read()
        
        # Results should be identical (buffer was reset)
        assert result1["shapes"] == result2["shapes"]
        assert result1["skipped"] == result2["skipped"]
        # File sizes should be very close (allowing minor variance)
        assert abs(len(content1) - len(content2)) < len(content1) * 0.1

    # --- Housing Parameter Tests ---

    def test_export_with_housing_none(self, exporter_single, temp_step_file):
        """Test that export() works with housing=None."""
        result = exporter_single.export(temp_step_file, housing=None)
        assert isinstance(result, dict)

    def test_export_with_empty_housing_list(self, exporter_single, temp_step_file):
        """Test that export() works with empty housing list."""
        result = exporter_single.export(temp_step_file, housing=[])
        assert isinstance(result, dict)

    def test_export_with_valid_housing_parts(self, exporter_single, temp_step_file):
        """Test export() with properly formatted housing parts."""
        housing = [
            {
                "name": "spacer",
                "z0": 0.0,
                "z1": 5.0,
                "r_inner": 10.0,
                "r_outer": 20.0,
            }
        ]
        result = exporter_single.export(temp_step_file, housing=housing)
        assert isinstance(result, dict)
        assert os.path.exists(temp_step_file)

    def test_export_skips_malformed_housing_missing_required_keys(self, exporter_single, temp_step_file):
        """Test that housing parts with missing required keys are skipped."""
        housing = [{"name": "incomplete_part"}]  # Missing z0, z1, r_inner, r_outer
        result = exporter_single.export(temp_step_file, housing=housing)
        
        assert result["skipped"] >= 1
        assert "incomplete_part" in result["skipped_parts"]

    def test_export_with_multiple_housing_parts(self, exporter_single, temp_step_file):
        """Test export with multiple housing parts."""
        housing = [
            {
                "name": "barrel",
                "z0": 0.0,
                "z1": 10.0,
                "r_inner": 15.0,
                "r_outer": 25.0,
            },
            {
                "name": "retainer",
                "z0": 10.0,
                "z1": 12.0,
                "r_inner": 18.0,
                "r_outer": 22.0,
            },
        ]
        result = exporter_single.export(temp_step_file, housing=housing)
        
        assert isinstance(result, dict)
        assert result["skipped"] == len(result["skipped_parts"])

    # --- System Type Tests ---

    def test_export_single_lens_system_success(self, exporter_single, temp_step_file):
        """Test successful export of a single lens system."""
        result = exporter_single.export(temp_step_file)
        assert result["shapes"] + result["skipped"] >= 1
        assert os.path.exists(temp_step_file)

    def test_export_multi_element_system_success(self, exporter_multi, temp_step_file):
        """Test successful export of a multi-element optical system."""
        result = exporter_multi.export(temp_step_file)
        assert isinstance(result, dict)
        assert os.path.exists(temp_step_file)
