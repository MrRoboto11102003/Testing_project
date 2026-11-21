import pytest
import tempfile
from pathlib import Path
import pandas as pd
import arnio as ar


class TestSchemaToDictBasics:
    """Test schema_to_dict with various input formats."""

    def test_flat_dict_from_scan_csv(self):
        """Convert flat dict (like output of ar.scan_csv)."""
        schema = {"id": "int64", "name": "string", "active": "bool"}
        result = ar.schema_to_dict(schema)

        assert "fields" in result
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["name"]["type"] == "STRING"
        assert result["fields"]["active"]["type"] == "BOOL"

    def test_structured_dict_with_metadata(self):
        """Preserve structured dict with strict and unique metadata."""
        schema = {
            "fields": {"user_id": "int64", "email": "string"},
            "strict": True,
            "unique": ["user_id"],
        }
        result = ar.schema_to_dict(schema)

        assert result["fields"]["user_id"]["type"] == "INT64"
        assert result["fields"]["email"]["type"] == "STRING"
        assert result["strict"] is True
        assert result["unique"] == ["user_id"]

    def test_schema_object_with_field_types(self, tmp_path):
        """Convert ar.Schema object to dict."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,alice\n2,bob\n")

        schema = ar.Schema(
            {"id": ar.Int64(nullable=False), "name": ar.String()}
        )
        result = ar.schema_to_dict(schema)

        assert "fields" in result
        assert "id" in result["fields"]
        assert "name" in result["fields"]

    def test_invalid_input_raises_typeerror(self):
        """Raise TypeError for unsupported input types."""
        with pytest.raises(TypeError):
            ar.schema_to_dict(123)

        with pytest.raises(TypeError):
            ar.schema_to_dict(None)

    def test_schema_with_custom_rules_raises_valueerror(self):
        """Reject Schema with custom callables (not serializable)."""
        def custom_rule(df):
            return []

        schema = ar.Schema({"id": ar.Int64()}, rules=[custom_rule])

        with pytest.raises(ValueError, match="custom rules"):
            ar.schema_to_dict(schema)


class TestSchemaToYamlBasics:
    """Test schema_to_yaml string and file output."""

    def test_returns_yaml_string(self):
        """Return valid YAML string with correct structure."""
        schema = {"id": "int64", "name": "string"}
        result = ar.schema_to_yaml(schema)

        assert isinstance(result, str)
        assert "fields:" in result
        assert "id:" in result
        assert "name:" in result
        assert result.endswith("\n")

    def test_writes_yaml_file_and_returns_string(self, tmp_path):
        """Write YAML to file and also return string."""
        schema = {"user_id": "int64", "email": "string"}
        yaml_file = tmp_path / "schema.yaml"

        result = ar.schema_to_yaml(schema, path=str(yaml_file))

        assert yaml_file.exists()
        file_content = yaml_file.read_text()
        assert file_content == result
        assert "user_id:" in file_content
        assert "email:" in file_content

    def test_accepts_pathlib_path(self, tmp_path):
        """Accept pathlib.Path in addition to string."""
        schema = {"value": "float64"}
        yaml_file = tmp_path / "output.yaml"

        ar.schema_to_yaml(schema, path=yaml_file)

        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert "value:" in content

    def test_creates_parent_directories_automatically(self, tmp_path):
        """Create missing parent directories."""
        schema = {"id": "int64"}
        nested_path = tmp_path / "nested" / "dirs" / "schema.yaml"

        ar.schema_to_yaml(schema, path=str(nested_path))

        assert nested_path.exists()
        assert "fields:" in nested_path.read_text()

    def test_invalid_path_type_raises_typeerror(self):
        """Raise TypeError for non-string/PathLike path."""
        schema = {"id": "int64"}

        with pytest.raises(TypeError):
            ar.schema_to_yaml(schema, path=123)

    def test_empty_path_string_raises_valueerror(self):
        """Raise ValueError for empty path string."""
        schema = {"id": "int64"}

        with pytest.raises(ValueError):
            ar.schema_to_yaml(schema, path="")

    def test_directory_path_raises_valueerror(self, tmp_path):
        """Raise ValueError if path points to existing directory."""
        schema = {"id": "int64"}

        with pytest.raises(ValueError):
            ar.schema_to_yaml(schema, path=str(tmp_path))


class TestEndToEndWorkflows:
    """Test complete workflows through real file I/O and processing."""

    def test_read_csv_scan_dict_yaml_workflow(self, tmp_path):
        """Read CSV → scan → convert to dict → export to YAML."""
        # Create test CSV with various types
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "customer_id,email,revenue,active\n"
            "1,alice@example.com,1000.50,true\n"
            "2,bob@example.com,2500.75,false\n"
        )

        # Scan to get inferred schema
        schema_dict = ar.scan_csv(str(csv_file))

        # Convert to standardized dict
        normalized = ar.schema_to_dict(schema_dict)
        assert "fields" in normalized
        assert "customer_id" in normalized["fields"]

        # Export to YAML
        yaml_file = tmp_path / "inferred_schema.yaml"
        yaml_content = ar.schema_to_yaml(normalized, path=str(yaml_file))

        assert yaml_file.exists()
        assert "customer_id" in yaml_content
        assert "email" in yaml_content
        assert "revenue" in yaml_content
        assert "active" in yaml_content

    def test_validate_pipeline_cleanup_export_schema(self, tmp_path):
        """Read → validate → clean → export updated schema."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "  name  , score \n"
            "  alice ,  95 \n"
            "  bob   ,  87 \n"
        )

        # Read CSV
        frame = ar.read_csv(str(csv_file))

        # Create and apply validation schema
        schema = ar.Schema(
            {
                "name": ar.String(nullable=False),
                "score": ar.Int64(nullable=False, min=0, max=100),
            }
        )
        result = ar.validate(frame, schema)
        assert result.passed

        # Clean the data
        clean = ar.pipeline(
            frame, [("strip_whitespace",), ("trim_column_names",)]
        )

        # Export cleaned schema to YAML
        schema_dict = ar.schema_to_dict(schema)
        yaml_file = tmp_path / "cleaned_schema.yaml"
        ar.schema_to_yaml(schema_dict, path=str(yaml_file))

        assert yaml_file.exists()
        yaml_text = yaml_file.read_text()
        assert "name:" in yaml_text
        assert "score:" in yaml_text

    def test_schema_metadata_strict_unique_roundtrip(self, tmp_path):
        """Verify strict and unique metadata persist through dict→YAML."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("user_id,email\n1,a@ex.com\n2,b@ex.com\n")

        # Create schema with constraints
        schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
            },
            strict=True,
            unique=["user_id"],
        )

        # Convert through dict
        schema_dict = ar.schema_to_dict(schema)
        assert schema_dict.get("strict") is True
        assert schema_dict.get("unique") == ["user_id"]

        # Export to YAML and verify metadata persists
        yaml_file = tmp_path / "user_schema.yaml"
        yaml_text = ar.schema_to_yaml(schema, path=str(yaml_file))

        # Verify file and string both have metadata
        file_text = yaml_file.read_text()
        assert "strict" in file_text
        assert "unique" in file_text
        assert yaml_text == file_text

    def test_profile_suggest_then_export_schema(self, tmp_path):
        """Profile CSV → get suggestions → export schema."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n1,alice,95\n2,bob,88\n3,bob,88\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        # Get cleaning suggestions
        suggestions = ar.suggest_cleaning(frame)
        assert isinstance(suggestions, list)

        # Extract schema dict and export
        schema_dict = ar.schema_to_dict(frame)
        yaml_file = tmp_path / "data_schema.yaml"
        ar.schema_to_yaml(schema_dict, path=str(yaml_file))

        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert "id:" in content
        assert "name:" in content
        assert "score:" in content
