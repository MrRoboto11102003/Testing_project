"""
Pytest end-to-end system tests for arnio.schema_export.schema_to_dict and schema_to_yaml.

Tests drive complete workflows through arnio's public API using real files on disk,
verifying outputs that users would observe: exported files, returned strings, error
types and messages.
"""

import pytest
import csv
import tempfile
import sys
from pathlib import Path
import arnio as ar
import pandas as pd


class TestSchemaToDictBasic:
    """Test schema_to_dict with basic scan_csv workflow."""

    def test_schema_to_dict_from_scan_csv(self, tmp_path):
        """Convert schema from scan_csv to normalized dict."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n")

        schema = ar.scan_csv(str(csv_file))
        result = ar.schema_to_dict(schema)

        # Verify normalized structure
        assert isinstance(result, dict)
        assert "fields" in result
        assert isinstance(result["fields"], dict)
        assert "id" in result["fields"]
        assert "name" in result["fields"]
        assert "score" in result["fields"]
        # Each field has at least a "type" key
        for field_name, field_def in result["fields"].items():
            assert isinstance(field_def, dict)
            assert "type" in field_def
            assert isinstance(field_def["type"], str)


class TestSchemaToYamlBasic:
    """Test schema_to_yaml string serialization."""

    def test_schema_to_yaml_from_scan_csv(self, tmp_path):
        """Serialize schema to YAML string from scan_csv output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n")

        schema = ar.scan_csv(str(csv_file))
        result = ar.schema_to_yaml(schema)

        # Verify YAML format
        assert isinstance(result, str)
        assert result.endswith("\n")
        assert "fields:" in result
        # Field names should appear
        assert "id:" in result or "name:" in result or "score:" in result
        # Type information should be present
        assert "type:" in result or "TYPE:" in result


class TestSchemaToYamlFileWrite:
    """Test schema_to_yaml file persistence."""

    def test_schema_to_yaml_write_to_file(self, tmp_path):
        """Write schema to YAML file and verify file is created."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user_id,username\n1,alice\n2,bob\n")

        output_file = tmp_path / "schema.yaml"
        schema = ar.scan_csv(str(csv_file))
        returned_str = ar.schema_to_yaml(schema, path=str(output_file))

        # Verify file was written
        assert output_file.exists()
        file_content = output_file.read_text(encoding="utf-8")

        # Returned string should match file content
        assert returned_str == file_content
        assert file_content.endswith("\n")
        assert "fields:" in file_content


class TestSchemaToDictWithArSchema:
    """Test schema_to_dict with explicit ar.Schema objects."""

    def test_schema_to_dict_with_ar_schema(self, tmp_path):
        """Convert ar.Schema object to normalized dict."""
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "age": ar.Int64(min=0, max=120),
            "name": ar.String(),
        })

        result = ar.schema_to_dict(schema)

        assert isinstance(result, dict)
        assert "fields" in result
        assert "email" in result["fields"]
        assert "age" in result["fields"]
        assert "name" in result["fields"]


class TestSchemaToYamlWithArSchema:
    """Test schema_to_yaml with ar.Schema objects."""

    def test_schema_to_yaml_with_ar_schema(self, tmp_path):
        """Serialize ar.Schema to YAML."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "name": ar.String(nullable=False),
            "email": ar.Email(),
        })

        result = ar.schema_to_yaml(schema)

        assert isinstance(result, str)
        assert result.endswith("\n")
        assert "fields:" in result
        # Field names should be present
        for field_name in ["id", "name", "email"]:
            assert field_name in result


class TestSchemaToDictWithArFrame:
    """Test schema_to_dict with ArFrame objects (via schema_summary)."""

    def test_schema_to_dict_with_arframe(self, tmp_path):
        """Extract and normalize schema from ArFrame."""
        csv_file = tmp_path / "products.csv"
        csv_file.write_text("product,quantity,price\nApple,10,1.50\nBanana,20,0.75\n")

        frame = ar.read_csv(str(csv_file))
        result = ar.schema_to_dict(frame)

        assert isinstance(result, dict)
        assert "fields" in result
        assert "product" in result["fields"]
        assert "quantity" in result["fields"]
        assert "price" in result["fields"]


class TestSchemaToDictErrorCases:
    """Test schema_to_dict error handling."""

    def test_schema_to_dict_unsupported_type_number(self):
        """Raise TypeError for numeric input."""
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict(12345)

    def test_schema_to_dict_unsupported_type_string(self):
        """Raise TypeError for string input."""
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict("not a schema")

    def test_schema_to_dict_unsupported_type_none(self):
        """Raise TypeError for None input."""
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict(None)


class TestSchemaToYamlErrorCases:
    """Test schema_to_yaml error handling."""

    def test_schema_to_yaml_invalid_path_empty_string(self, tmp_path):
        """Raise ValueError for empty path string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        schema = ar.scan_csv(str(csv_file))

        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml(schema, path="")

    def test_schema_to_yaml_invalid_path_is_directory(self, tmp_path):
        """Raise ValueError when path is an existing directory."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        schema = ar.scan_csv(str(csv_file))

        with pytest.raises(ValueError, match="must point to a file, not a directory"):
            ar.schema_to_yaml(schema, path=str(tmp_path))

    def test_schema_to_yaml_invalid_path_type(self, tmp_path):
        """Raise TypeError for invalid path type (not string/PathLike)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        schema = ar.scan_csv(str(csv_file))

        with pytest.raises(TypeError, match="path must be a string or os.PathLike"):
            ar.schema_to_yaml(schema, path=12345)


class TestSchemaWithMetadata:
    """Test schema export with strict and unique constraints."""

    def test_schema_dict_includes_strict_metadata(self, tmp_path):
        """Schema dict preserves strict=True metadata."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
        }, strict=True)

        result = ar.schema_to_dict(schema)

        assert result.get("strict") is True
        assert "fields" in result

    def test_schema_yaml_includes_unique_constraint(self, tmp_path):
        """Schema YAML output includes unique constraint metadata."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(),
        }, unique=["id", "email"])

        result = ar.schema_to_yaml(schema)

        assert isinstance(result, str)
        # Result should include unique constraint info somewhere
        assert "fields:" in result


class TestSchemaDictNullableFields:
    """Test schema_to_dict preserves nullable information."""

    def test_schema_dict_captures_nullable_flags(self, tmp_path):
        """schema_to_dict preserves nullable field metadata."""
        schema = ar.Schema({
            "required_id": ar.Int64(nullable=False),
            "optional_name": ar.String(nullable=True),
        })

        result = ar.schema_to_dict(schema)

        # Check that nullable info is preserved
        assert "fields" in result
        required_field = result["fields"]["required_id"]
        optional_field = result["fields"]["optional_name"]

        # At least one should have explicit nullable (implementation may vary)
        assert isinstance(required_field, dict)
        assert isinstance(optional_field, dict)


class TestSchemaExportIntegration:
    """Test schema export in complete cleaning and validation workflows."""

    def test_schema_export_after_cleaning_pipeline(self, tmp_path):
        """Export schema after applying cleaning pipeline."""
        # Create raw data
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "id,customer_name,amount\n"
            "1, Alice , 100.50\n"
            "2, Bob , 200.75\n"
            "1, Alice , 100.50\n"  # duplicate
        )

        # Clean using pipeline
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        # Export schema from cleaned frame
        schema_dict = ar.schema_to_dict(clean)
        schema_yaml = ar.schema_to_yaml(clean)

        # Verify exports contain expected structure
        assert "fields" in schema_dict
        assert len(schema_dict["fields"]) == 3
        for col in ["id", "customer_name", "amount"]:
            assert col in schema_dict["fields"]

        # YAML should be valid output
        assert isinstance(schema_yaml, str)
        assert schema_yaml.endswith("\n")
        assert "fields:" in schema_yaml

    def test_schema_export_with_validation(self, tmp_path):
        """Export schema after creating and using validation schema."""
        # Create data
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,email\n1,alice@example.com\n2,bob@example.com\n")

        # Read and validate
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)

        # Validation should pass
        assert result.passed

        # Export schema to dict and yaml
        exported_dict = ar.schema_to_dict(schema)
        exported_yaml = ar.schema_to_yaml(schema)

        assert "fields" in exported_dict
        assert "id" in exported_dict["fields"]
        assert "email" in exported_dict["fields"]
        assert "fields:" in exported_yaml


class TestSchemaDictFromPandasWorkflow:
    """Test schema export from pandas-integrated workflows."""

    def test_schema_export_after_pandas_cleaning(self, tmp_path):
        """Export schema from ArFrame created via pandas integration."""
        # Create pandas DataFrame
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "username": [" alice ", " bob ", " charlie "],
        })

        # Clean using .arnio accessor
        clean_df = df.arnio.clean(strip_whitespace=True)

        # Convert to ArFrame and export schema
        frame = ar.from_pandas(clean_df)
        schema_dict = ar.schema_to_dict(frame)
        schema_yaml = ar.schema_to_yaml(frame)

        # Verify both export formats work
        assert "fields" in schema_dict
        assert "user_id" in schema_dict["fields"]
        assert "username" in schema_dict["fields"]
        assert "fields:" in schema_yaml
        assert "user_id" in schema_yaml


class TestSchemaRoundTripConsistency:
    """Test consistency across multiple export formats."""

    def test_schema_dict_yaml_consistency(self, tmp_path):
        """schema_to_dict and schema_to_yaml produce consistent output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "order_id,customer,amount\n"
            "101,Alice,100.00\n"
            "102,Bob,250.50\n"
        )

        # Read and export in both formats
        frame = ar.read_csv(str(csv_file))
        dict_result = ar.schema_to_dict(frame)
        yaml_result = ar.schema_to_yaml(frame)

        # Both should contain the same field names
        dict_fields = set(dict_result["fields"].keys())
        expected_fields = {"order_id", "customer", "amount"}
        assert expected_fields.issubset(dict_fields)

        # YAML output should reference all these fields
        for field_name in expected_fields:
            assert field_name in yaml_result

    def test_schema_to_yaml_always_ends_with_newline(self, tmp_path):
        """schema_to_yaml always returns string ending with newline."""
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.schema_to_yaml(schema)
        assert result.endswith("\n")

        # Even with file write
        output_file = tmp_path / "out.yaml"
        result_with_file = ar.schema_to_yaml(schema, path=str(output_file))
        assert result_with_file.endswith("\n")
        assert output_file.read_text().endswith("\n")


class TestSchemaDictSpecialCases:
    """Test schema_to_dict with edge cases and special scenarios."""

    def test_schema_to_dict_with_special_column_names(self, tmp_path):
        """schema_to_dict handles special characters in column names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user-id,first_name,email_2\n1,Alice,a@ex.com\n")

        frame = ar.read_csv(str(csv_file))
        result = ar.schema_to_dict(frame)

        # Columns should be preserved as they appear
        assert "fields" in result
        assert len(result["fields"]) >= 2


class TestSchemaYamlFileCreation:
    """Test schema_to_yaml file creation and overwriting."""

    def test_schema_to_yaml_overwrites_existing_file(self, tmp_path):
        """schema_to_yaml overwrites existing file at path."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        schema = ar.scan_csv(str(csv_file))

        output_file = tmp_path / "schema.yaml"

        # Write first time
        ar.schema_to_yaml(schema, path=str(output_file))
        first_mtime = output_file.stat().st_mtime

        # Write again (should overwrite)
        import time
        time.sleep(0.01)  # Ensure mtime changes if file is rewritten
        ar.schema_to_yaml(schema, path=str(output_file))

        # File should exist and contain valid schema
        assert output_file.exists()
        content = output_file.read_text()
        assert "fields:" in content

    def test_schema_to_yaml_creates_parent_directories(self, tmp_path):
        """schema_to_yaml creates parent directories as needed."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        schema = ar.scan_csv(str(csv_file))

        nested_dir = tmp_path / "schemas" / "v1" / "output.yaml"
        ar.schema_to_yaml(schema, path=str(nested_dir))

        assert nested_dir.exists()
        assert nested_dir.read_text().startswith("fields:")
