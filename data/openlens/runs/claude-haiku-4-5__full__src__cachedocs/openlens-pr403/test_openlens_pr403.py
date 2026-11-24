import pytest
import tempfile
import os
from pathlib import Path

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.io.step_export import StepExporter


class TestStepExporterExport:
    """Tests for StepExporter.export method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def simple_lens(self):
        """Create a simple biconvex lens for testing."""
        return Lens(
            name="Test Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )

    @pytest.fixture
    def plano_convex_lens(self):
        """Create a plano-convex lens."""
        return Lens(
            name="Plano-Convex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=float("inf"),
            thickness=4.0,
            diameter=20.0,
            material="BK7",
        )

    @pytest.fixture
    def multi_element_system(self, simple_lens):
        """Create a two-element optical system."""
        system = OpticalSystem(name="Two-Element System")
        system.add_lens(simple_lens)
        
        lens2 = Lens(
            name="Diverging Lens",
            radius_of_curvature_1=-50.0,
            radius_of_curvature_2=50.0,
            thickness=4.0,
            diameter=25.0,
            material="BK7",
        )
        system.add_lens(lens2, air_gap_before=1.0)
        return system

    @pytest.fixture
    def exporter_single(self, simple_lens):
        """Create exporter for single lens."""
        return StepExporter(simple_lens)

    @pytest.fixture
    def exporter_system(self, multi_element_system):
        """Create exporter for multi-element system."""
        return StepExporter(multi_element_system)

    def test_export_single_lens_creates_file(self, exporter_single, temp_dir):
        """Test that export creates an output file."""
        output_file = os.path.join(temp_dir, "lens.step")
        result = exporter_single.export(output_file)
        
        assert os.path.exists(output_file), "STEP file was not created"
        assert os.path.getsize(output_file) > 0, "STEP file is empty"

    def test_export_returns_dict_structure(self, exporter_single, temp_dir):
        """Test that export returns dict with required keys."""
        output_file = os.path.join(temp_dir, "structure.step")
        result = exporter_single.export(output_file)
        
        assert isinstance(result, dict), "Result is not a dict"
        required_keys = {"shapes", "skipped", "skipped_parts"}
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    def test_export_return_values_are_valid_types(self, exporter_single, temp_dir):
        """Test that return values have correct types."""
        output_file = os.path.join(temp_dir, "types.step")
        result = exporter_single.export(output_file)
        
        assert isinstance(result["shapes"], int), "shapes should be int"
        assert isinstance(result["skipped"], int), "skipped should be int"
        assert isinstance(result["skipped_parts"], list), "skipped_parts should be list"

    def test_export_single_lens_produces_shapes(self, exporter_single, temp_dir):
        """Test that single lens export produces at least one shape."""
        output_file = os.path.join(temp_dir, "shapes.step")
        result = exporter_single.export(output_file)
        
        assert result["shapes"] >= 1, "Single lens should produce at least one shape"

    def test_export_multi_element_system(self, exporter_system, temp_dir):
        """Test export of multi-element optical system."""
        output_file = os.path.join(temp_dir, "system.step")
        result = exporter_system.export(output_file)
        
        assert result["shapes"] >= 2, "Two-element system should produce at least 2 shapes"
        assert os.path.exists(output_file)

    def test_export_with_valid_housing_parts(self, exporter_single, temp_dir):
        """Test export with valid housing specification."""
        output_file = os.path.join(temp_dir, "housing.step")
        
        housing = [
            {
                "name": "spacer",
                "z0": 10.0,
                "z1": 12.0,
                "r_inner": 5.0,
                "r_outer": 15.0,
            },
            {
                "name": "barrel",
                "z0": 12.0,
                "z1": 25.0,
                "r_inner": 5.0,
                "r_outer": 20.0,
            },
        ]
        
        result = exporter_single.export(output_file, housing=housing)
        
        # Should have lens + 2 housing parts
        assert result["shapes"] >= 3, "Should export lens plus housing parts"
        assert result["skipped"] == 0, "Valid housing should not be skipped"

    def test_export_with_malformed_housing_missing_keys(self, exporter_single, temp_dir):
        """Test that malformed housing parts are skipped."""
        output_file = os.path.join(temp_dir, "malformed.step")
        
        housing = [
            {
                "name": "incomplete",
                "z0": 10.0,
                "z1": 12.0,
                # Missing r_inner and r_outer
            }
        ]
        
        result = exporter_single.export(output_file, housing=housing)
        
        assert result["skipped"] > 0, "Malformed housing should be skipped"
        assert "incomplete" in result["skipped_parts"], "Skipped part name should be recorded"
        assert result["shapes"] >= 1, "Lens should still be exported"

    def test_export_with_invalid_housing_non_numeric(self, exporter_single, temp_dir):
        """Test that housing with non-numeric values is skipped."""
        output_file = os.path.join(temp_dir, "non_numeric.step")
        
        housing = [
            {
                "name": "bad_values",
                "z0": "not_a_number",
                "z1": 12.0,
                "r_inner": 5.0,
                "r_outer": 15.0,
            }
        ]
        
        result = exporter_single.export(output_file, housing=housing)
        
        assert result["skipped"] > 0, "Non-numeric housing should be skipped"
        assert "bad_values" in result["skipped_parts"]

    def test_export_repeated_calls_reset_buffer(self, exporter_single, temp_dir):
        """Test that buffer is reset on each export (no duplication)."""
        file1 = os.path.join(temp_dir, "export1.step")
        file2 = os.path.join(temp_dir, "export2.step")
        
        result1 = exporter_single.export(file1)
        result2 = exporter_single.export(file2)
        
        # Same exporter should produce same result
        assert result1["shapes"] == result2["shapes"], "Buffer not properly reset"
        
        # File sizes should be similar (not doubled)
        size1 = os.path.getsize(file1)
        size2 = os.path.getsize(file2)
        ratio = max(size1, size2) / min(size1, size2)
        assert ratio < 1.5, "File sizes differ too much, buffer likely not reset"

    def test_export_without_housing_parameter(self, exporter_single, temp_dir):
        """Test export with housing=None (default)."""
        output_file = os.path.join(temp_dir, "no_housing.step")
        result = exporter_single.export(output_file, housing=None)
        
        assert result["shapes"] >= 1
        assert os.path.exists(output_file)

    def test_export_with_empty_housing_list(self, exporter_single, temp_dir):
        """Test export with empty housing list."""
        output_file = os.path.join(temp_dir, "empty_housing.step")
        result = exporter_single.export(output_file, housing=[])
        
        assert result["shapes"] >= 1
        assert result["skipped"] == 0

    def test_export_skipped_parts_consistency(self, exporter_single, temp_dir):
        """Test that skipped_parts list length matches skipped count."""
        output_file = os.path.join(temp_dir, "consistency.step")
        
        housing = [
            {"name": "bad1", "z0": "x", "z1": 1.0, "r_inner": 5.0, "r_outer": 15.0},
            {"name": "bad2", "z0": 0.0, "z1": 1.0, "r_inner": 5.0},  # Missing r_outer
        ]
        
        result = exporter_single.export(output_file, housing=housing)
        
        assert len(result["skipped_parts"]) == result["skipped"], (
            f"skipped_parts length {len(result['skipped_parts'])} "
            f"does not match skipped count {result['skipped']}"
        )

    def test_export_plano_convex_lens(self, plano_convex_lens, temp_dir):
        """Test export of plano-convex lens (different profile)."""
        exporter = StepExporter(plano_convex_lens)
        output_file = os.path.join(temp_dir, "plano_convex.step")
        result = exporter.export(output_file)
        
        assert result["shapes"] >= 1
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_export_output_file_contains_step_format(self, exporter_single, temp_dir):
        """Test that exported file contains STEP format indicators."""
        output_file = os.path.join(temp_dir, "format.step")
        exporter_single.export(output_file)
        
        with open(output_file, "r") as f:
            content = f.read()
        
        # STEP files should contain expected structures
        assert "MANIFOLD_SOLID_BREP" in content or "SPHERICAL_SURFACE" in content, (
            "Output does not appear to be valid STEP format"
        )
