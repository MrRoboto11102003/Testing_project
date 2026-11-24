import pytest
from pathlib import Path
from openlens.lens import Lens
from openlens.io.step_export import StepExporter


class TestStepExporterExport:
    """Tests for StepExporter.export() method."""

    def test_export_none_system_returns_skipped_summary(self, tmp_path):
        """Test export with None system returns skipped summary dict."""
        output_file = tmp_path / "output.step"
        exporter = StepExporter(None)
        result = exporter.export(str(output_file))

        assert result["shapes"] == 0
        assert result["skipped"] == 1
        assert result["skipped_parts"] == ["system"]
        assert output_file.exists()

    def test_export_single_spherical_lens_success(self, tmp_path):
        """Test successful export of a single spherical lens."""
        lens = Lens(
            name="BiconvexTest",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        output_file = tmp_path / "lens.step"
        exporter = StepExporter(lens)
        result = exporter.export(str(output_file))

        assert result["shapes"] == 1
        assert result["skipped"] == 0
        assert result["skipped_parts"] == []
        assert output_file.exists()

    def test_export_file_iso10303_21_format(self, tmp_path):
        """Test exported file has correct ISO 10303-21 format."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "format.step"
        exporter = StepExporter(lens)
        exporter.export(str(output_file))

        content = output_file.read_text()
        assert content.startswith("ISO-10303-21;")
        assert "HEADER;" in content
        assert "DATA;" in content
        assert content.strip().endswith("END-ISO-10303-21;")
        assert "MANIFOLD_SOLID_BREP" in content

    def test_export_context_entities_always_present(self, tmp_path):
        """Test that mandatory context entities #1-#11 are present."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "context.step"
        exporter = StepExporter(lens)
        exporter.export(str(output_file))

        content = output_file.read_text()
        assert "#1=DIRECTION('axis_z',(0.000000,0.000000,1.000000))" in content
        assert "#2=DIRECTION('axis_x',(1.000000,0.000000,0.000000))" in content
        assert "#3=DIRECTION('axis_y',(0.000000,1.000000,0.000000))" in content
        assert "#4=CARTESIAN_POINT('origin',(0.000000,0.000000,0.000000))" in content

    def test_export_parabolic_surface_raises_valueerror(self, tmp_path):
        """Test that parabolic surfaces raise ValueError and create no file."""
        lens = Lens(
            name="ParabolicLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.1,
        )
        output_file = tmp_path / "parabolic.step"
        exporter = StepExporter(lens)

        with pytest.raises(ValueError, match="spherical and flat surfaces only"):
            exporter.export(str(output_file))

        assert not output_file.exists()

    def test_export_conic_surface_raises_valueerror(self, tmp_path):
        """Test that conic surfaces raise ValueError."""
        lens = Lens(
            name="ConicLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            conic_constant_1=-1.0,
        )
        output_file = tmp_path / "conic.step"
        exporter = StepExporter(lens)

        with pytest.raises(ValueError, match="conic"):
            exporter.export(str(output_file))

        assert not output_file.exists()

    def test_export_repeated_calls_no_entity_accumulation(self, tmp_path):
        """Test repeated exports do not accumulate entities."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        exporter = StepExporter(lens)

        file1 = tmp_path / "export1.step"
        file2 = tmp_path / "export2.step"

        result1 = exporter.export(str(file1))
        result2 = exporter.export(str(file2))

        content1 = file1.read_text()
        content2 = file2.read_text()

        data1 = content1.split("DATA;\n")[1].split("\nENDSEC;")[0]
        data2 = content2.split("DATA;\n")[1].split("\nENDSEC;")[0]

        entity_count1 = data1.count("#")
        entity_count2 = data2.count("#")
        assert entity_count1 == entity_count2
        assert result1 == result2

    def test_export_with_valid_housing_parts(self, tmp_path):
        """Test export with valid housing parts."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "with_housing.step"
        exporter = StepExporter(lens)

        housing = [
            {
                "name": "spacer",
                "z0": 0.0,
                "z1": 2.0,
                "r_inner": 10.0,
                "r_outer": 12.0,
            }
        ]

        result = exporter.export(str(output_file), housing=housing)

        assert result["shapes"] == 2
        assert result["skipped"] == 0
        assert output_file.exists()

    def test_export_housing_invalid_dimensions_skipped(self, tmp_path):
        """Test that housing with invalid dimensions is skipped."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "invalid_housing.step"
        exporter = StepExporter(lens)

        housing = [
            {
                "name": "bad_spacer",
                "z0": 2.0,
                "z1": 2.0,
                "r_inner": 10.0,
                "r_outer": 12.0,
            }
        ]

        result = exporter.export(str(output_file), housing=housing)

        assert result["shapes"] == 1
        assert result["skipped"] == 1
        assert "bad_spacer" in result["skipped_parts"]

    def test_export_housing_missing_keys_skipped(self, tmp_path):
        """Test that housing with missing required keys is skipped."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "missing_keys.step"
        exporter = StepExporter(lens)

        housing = [
            {
                "name": "incomplete",
                "z0": 0.0,
            }
        ]

        result = exporter.export(str(output_file), housing=housing)

        assert result["shapes"] == 1
        assert result["skipped"] == 1
        assert "incomplete" in result["skipped_parts"]

    def test_export_empty_housing_list(self, tmp_path):
        """Test export with empty housing list."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "no_housing.step"
        exporter = StepExporter(lens)

        result = exporter.export(str(output_file), housing=[])

        assert result["shapes"] == 1
        assert result["skipped"] == 0
        assert result["skipped_parts"] == []

    def test_export_housing_none_equivalent_to_empty(self, tmp_path):
        """Test that housing=None behaves same as housing=[]."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        file1 = tmp_path / "none_housing.step"
        file2 = tmp_path / "empty_housing.step"
        exporter1 = StepExporter(lens)
        exporter2 = StepExporter(lens)

        result1 = exporter1.export(str(file1), housing=None)
        result2 = exporter2.export(str(file2), housing=[])

        assert result1 == result2

    def test_export_multiple_housing_parts(self, tmp_path):
        """Test export with multiple housing parts."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "multi_housing.step"
        exporter = StepExporter(lens)

        housing = [
            {
                "name": "spacer1",
                "z0": 0.0,
                "z1": 1.0,
                "r_inner": 10.0,
                "r_outer": 12.0,
            },
            {
                "name": "spacer2",
                "z0": 10.0,
                "z1": 12.0,
                "r_inner": 10.0,
                "r_outer": 12.0,
            },
        ]

        result = exporter.export(str(output_file), housing=housing)

        assert result["shapes"] == 3
        assert result["skipped"] == 0

    def test_export_return_dict_has_required_keys(self, tmp_path):
        """Test that return dict has correct structure and types."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
        )
        output_file = tmp_path / "return_check.step"
        exporter = StepExporter(lens)
        result = exporter.export(str(output_file))

        assert isinstance(result, dict)
        assert set(result.keys()) == {"shapes", "skipped", "skipped_parts"}
        assert isinstance(result["shapes"], int)
        assert isinstance(result["skipped"], int)
        assert isinstance(result["skipped_parts"], list)
        assert result["skipped"] == len(result["skipped_parts"])
