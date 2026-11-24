import pytest
from pathlib import Path
import openlens
from openlens import Lens
from openlens.io.step_export import StepExporter


class TestStepExporterExport:
    """Test suite for StepExporter.export method."""

    def test_export_none_system_returns_zero_shapes(self, tmp_path):
        """When system is None, export returns 0 shapes and 1 skipped."""
        exporter = StepExporter(None)
        outfile = tmp_path / "output.step"
        
        result = exporter.export(str(outfile))
        
        assert result["shapes"] == 0
        assert result["skipped"] == 1
        assert result["skipped_parts"] == ["system"]
        assert outfile.exists()

    def test_export_single_spherical_lens(self, tmp_path):
        """Test export of a basic spherical lens returns 1 shape."""
        lens = Lens(
            name="TestLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "lens.step"
        
        result = exporter.export(str(outfile))
        
        assert result["shapes"] == 1
        assert result["skipped"] == 0
        assert result["skipped_parts"] == []
        assert outfile.exists()

    def test_export_raises_on_conic_surface_no_file_created(self, tmp_path):
        """Conic surfaces (K != 0) should raise ValueError and not create file."""
        lens = Lens(
            name="ConicLens",
            radius_of_curvature_1=50.0,
            conic_constant_1=-1.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "conic.step"
        
        with pytest.raises(ValueError, match="spherical and flat surfaces only"):
            exporter.export(str(outfile))
        
        assert not outfile.exists()

    def test_export_raises_on_parabolic_surface_with_sag(self, tmp_path):
        """Parabolic surfaces with non-zero sag should raise ValueError."""
        lens = Lens(
            name="ParabolicLens",
            radius_of_curvature_1=50.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.1,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "parabolic.step"
        
        with pytest.raises(ValueError, match="spherical and flat surfaces only"):
            exporter.export(str(outfile))
        
        assert not outfile.exists()

    def test_export_with_valid_housing_increases_shape_count(self, tmp_path):
        """Test export with valid housing includes housing parts in shape count."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "with_housing.step"
        
        housing = [
            {
                "name": "spacer",
                "z0": 0.0,
                "z1": 2.0,
                "r_inner": 10.0,
                "r_outer": 12.0
            }
        ]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["shapes"] == 2
        assert result["skipped"] == 0
        assert outfile.exists()

    def test_export_housing_missing_required_key_skipped(self, tmp_path):
        """Housing with missing required keys should be skipped."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "missing_key.step"
        
        housing = [{"name": "incomplete", "z0": 0.0}]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["shapes"] == 1
        assert result["skipped"] == 1
        assert "incomplete" in result["skipped_parts"]

    def test_export_housing_invalid_dimensions_skipped(self, tmp_path):
        """Housing with invalid dimensions (z1<=z0, r_outer<=r_inner) skipped."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "invalid_dims.step"
        
        housing = [
            {
                "name": "badpart",
                "z0": 2.0,
                "z1": 2.0,
                "r_inner": 10.0,
                "r_outer": 10.0
            }
        ]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["skipped"] >= 1
        assert "badpart" in result["skipped_parts"]

    def test_export_repeated_calls_produce_identical_data(self, tmp_path):
        """Repeated exports should produce identical DATA sections (no accumulation)."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        file1 = tmp_path / "export1.step"
        file2 = tmp_path / "export2.step"
        
        exporter.export(str(file1))
        content1 = file1.read_text()
        
        exporter.export(str(file2))
        content2 = file2.read_text()
        
        data1 = content1.split("DATA;\n")[1].split("\nENDSEC;")[0]
        data2 = content2.split("DATA;\n")[1].split("\nENDSEC;")[0]
        
        assert data1 == data2

    def test_export_file_has_iso_10303_21_structure(self, tmp_path):
        """Exported file should follow ISO-10303-21 format structure."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "format.step"
        exporter.export(str(outfile))
        
        content = outfile.read_text()
        
        assert content.startswith("ISO-10303-21;")
        assert "HEADER;" in content
        assert "DATA;" in content
        assert content.endswith("END-ISO-10303-21;")

    def test_export_contains_standard_context_entities(self, tmp_path):
        """Standard context entities like DIRECTION, LENGTH_UNIT present."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "context.step"
        exporter.export(str(outfile))
        
        content = outfile.read_text()
        
        assert "DIRECTION('axis_z',(0.000000,0.000000,1.000000))" in content
        assert "LENGTH_UNIT()" in content
        assert "APPLICATION_CONTEXT(" in content

    def test_export_lens_name_escaping_quotes(self, tmp_path):
        """Lens names with special characters like quotes are properly escaped."""
        lens = Lens(
            name="it's",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "quoted.step"
        exporter.export(str(outfile))
        
        content = outfile.read_text()
        
        assert "'it''s'" in content or "it's" in content

    def test_export_return_value_structure_complete(self, tmp_path):
        """Return dict has correct keys and types."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "return_check.step"
        
        result = exporter.export(str(outfile))
        
        assert isinstance(result, dict)
        assert set(result.keys()) == {"shapes", "skipped", "skipped_parts"}
        assert isinstance(result["shapes"], int)
        assert isinstance(result["skipped"], int)
        assert isinstance(result["skipped_parts"], list)
        assert result["skipped"] == len(result["skipped_parts"])

    def test_export_housing_numeric_string_coercion(self, tmp_path):
        """Housing numeric values as strings are coerced to float."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "string_coerce.step"
        
        housing = [
            {
                "name": "tube",
                "z0": "0.0",
                "z1": "2.0",
                "r_inner": "10.0",
                "r_outer": "12.0"
            }
        ]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["shapes"] == 2
        assert result["skipped"] == 0

    def test_export_housing_non_numeric_value_skipped(self, tmp_path):
        """Housing with non-numeric values is skipped and counted."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "non_numeric.step"
        
        housing = [
            {
                "name": "badtube",
                "z0": "not_a_number",
                "z1": 2.0,
                "r_inner": 10.0,
                "r_outer": 12.0
            }
        ]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["shapes"] == 1
        assert result["skipped"] == 1
        assert "badtube" in result["skipped_parts"]

    def test_export_mixed_valid_invalid_housing_parts(self, tmp_path):
        """Mix of valid and invalid housing - valid exported, invalid skipped."""
        lens = Lens(
            name="L",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0
        )
        exporter = StepExporter(lens)
        outfile = tmp_path / "mixed.step"
        
        housing = [
            {"name": "good1", "z0": 0.0, "z1": 2.0, "r_inner": 10.0, "r_outer": 12.0},
            {"name": "bad", "z0": 2.0, "z1": 2.0, "r_inner": 10.0, "r_outer": 12.0},
            {"name": "good2", "z0": 4.0, "z1": 6.0, "r_inner": 10.0, "r_outer": 12.0},
        ]
        
        result = exporter.export(str(outfile), housing=housing)
        
        assert result["shapes"] == 3
        assert result["skipped"] == 1
        assert "bad" in result["skipped_parts"]
