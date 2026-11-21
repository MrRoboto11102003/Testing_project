import pytest
import pathlib
from pathlib import Path
import yaml

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestSchemaToDict:
    """Test ar.schema_to_dict with various input types and edge cases."""
    
    def test_flat_dict_input(self):
        """Test with flat dict from scan_csv."""
        schema = {"id": "INT64", "name": "STRING", "age": "INT32"}
        result = ar.schema_to_dict(schema)
        
        assert isinstance(result, dict)
        assert "fields" in result
        assert len(result["fields"]) == 3
        # Fields must be sorted
        assert list(result["fields"].keys()) == ["age", "id", "name"]
        assert result["fields"]["id"]["type"] == "INT64"
    
    def test_structured_dict_with_metadata(self):
        """Test structured dict with fields key and additional metadata."""
        schema = {
            "fields": {
                "id": {"type": "INT64", "nullable": False},
                "name": {"type": "STRING", "nullable": True},
            },
            "strict": True,
            "unique": ["id"],
        }
        result = ar.schema_to_dict(schema)
        
        assert result["fields"]["id"]["nullable"] is False
        assert result["fields"]["name"]["nullable"] is True
        assert result["strict"] is True
        assert result["unique"] == ["id"]
    
    def test_field_names_sorted_alphabetically(self):
        """Test that output field names are sorted regardless of input order."""
        schema = {"zebra": "STRING", "apple": "INT64", "monkey": "FLOAT64"}
        result = ar.schema_to_dict(schema)
        
        field_names = list(result["fields"].keys())
        assert field_names == ["apple", "monkey", "zebra"]
    
    def test_invalid_type_raises_typeerror(self):
        """Test that unsupported input types raise TypeError."""
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict(12345)
        
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict(None)
    
    def test_non_string_metadata_key_raises_typeerror(self):
        """Test that non-string metadata keys raise TypeError."""
        schema = {"fields": {"id": "INT64"}, 123: "value"}
        with pytest.raises(TypeError, match="schema metadata keys must be strings"):
            ar.schema_to_dict(schema)


class TestSchemaToYaml:
    """Test ar.schema_to_yaml YAML serialization and file I/O."""
    
    def test_returns_valid_yaml_string(self):
        """Test that schema_to_yaml returns valid YAML string."""
        schema = {"id": "INT64", "name": "STRING"}
        result = ar.schema_to_yaml(schema)
        
        assert isinstance(result, str)
        assert result.endswith("\n")
        # Verify it's valid YAML
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "fields" in parsed
    
    def test_yaml_output_has_sorted_fields(self):
        """Test that YAML contains sorted field names."""
        schema = {"z_col": "STRING", "a_col": "INT64", "m_col": "FLOAT64"}
        yaml_str = ar.schema_to_yaml(schema)
        
        parsed = yaml.safe_load(yaml_str)
        assert list(parsed["fields"].keys()) == ["a_col", "m_col", "z_col"]
    
    def test_writes_to_file_when_path_provided(self, tmp_path):
        """Test that schema is written to file and return value matches file content."""
        schema = {"id": "INT64", "name": "STRING"}
        output_file = tmp_path / "schema.yaml"
        
        result = ar.schema_to_yaml(schema, path=output_file)
        
        assert output_file.exists()
        assert isinstance(result, str)
        # Verify file contents match return value
        file_content = output_file.read_text(encoding="utf-8")
        assert file_content == result
    
    def test_overwrites_existing_file(self, tmp_path):
        """Test that existing file is overwritten."""
        output_file = tmp_path / "schema.yaml"
        output_file.write_text("old content\n", encoding="utf-8")
        
        schema = {"id": "INT64"}
        ar.schema_to_yaml(schema, path=output_file)
        
        content = output_file.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "fields" in content
        assert "id" in content
    
    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        schema = {"id": "INT64"}
        output_file = tmp_path / "nested" / "dir" / "schema.yaml"
        
        ar.schema_to_yaml(schema, path=output_file)
        
        assert output_file.exists()
        assert output_file.is_file()
    
    def test_accepts_string_and_pathlib_paths(self, tmp_path):
        """Test that path parameter accepts both string and pathlib.Path."""
        schema = {"id": "INT64"}
        
        # String path
        str_path = str(tmp_path / "schema1.yaml")
        ar.schema_to_yaml(schema, path=str_path)
        assert Path(str_path).exists()
        
        # Pathlib.Path
        path_obj = tmp_path / "schema2.yaml"
        ar.schema_to_yaml(schema, path=path_obj)
        assert path_obj.exists()
    
    def test_invalid_path_types_and_values_raise_errors(self, tmp_path):
        """Test that invalid path types and values raise appropriate errors."""
        schema = {"id": "INT64"}
        
        # Invalid type (not str or PathLike)
        with pytest.raises(TypeError, match="path must be a string or os.PathLike"):
            ar.schema_to_yaml(schema, path=123)
        
        # Empty string
        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml(schema, path="")
        
        # Whitespace-only string
        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml(schema, path="   ")
        
        # Directory instead of file
        with pytest.raises(ValueError, match="path must point to a file, not a directory"):
            ar.schema_to_yaml(schema, path=tmp_path)


class TestIntegration:
    """Integration tests exercising complete user workflows."""
    
    def test_dict_to_yaml_to_dict_roundtrip(self):
        """Test that dict→yaml→dict preserves schema structure."""
        original = {
            "fields": {
                "id": {"type": "INT64", "nullable": False},
                "name": {"type": "STRING", "nullable": True},
            },
            "strict": True,
            "unique": ["id"],
        }
        
        # Convert to YAML
        yaml_str = ar.schema_to_yaml(original)
        
        # Parse YAML back to dict
        reparsed = yaml.safe_load(yaml_str)
        
        # Convert through schema_to_dict
        reimported = ar.schema_to_dict(reparsed)
        
        # Verify key attributes survived roundtrip
        assert reimported["fields"]["id"]["type"] == "INT64"
        assert reimported["fields"]["id"]["nullable"] is False
        assert reimported["fields"]["name"]["nullable"] is True
        assert reimported["strict"] is True
        assert reimported["unique"] == ["id"]
    
    def test_scan_csv_schema_export_to_yaml_file(self, tmp_path):
        """Test end-to-end workflow: scan CSV, export schema to YAML file."""
        # Create sample CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95.5\n2,Bob,87.3\n", encoding="utf-8")
        
        # Scan CSV to get schema
        schema = ar.scan_csv(str(csv_file))
        
        # Export schema to YAML file
        yaml_file = tmp_path / "schema.yaml"
        yaml_str = ar.schema_to_yaml(schema, path=yaml_file)
        
        # Verify file was created and contains valid YAML
        assert yaml_file.exists()
        parsed = yaml.safe_load(yaml_str)
        assert "fields" in parsed
        assert "id" in parsed["fields"]
        assert "name" in parsed["fields"]
        assert "score" in parsed["fields"]
    
    def test_flat_dict_export_produces_sorted_yaml(self, tmp_path):
        """Test that flat dict is correctly converted to sorted YAML."""
        # Flat dict as returned by scan_csv
        schema = {
            "user_id": "INT64",
            "user_email": "STRING",
            "user_name": "STRING",
        }
        
        yaml_file = tmp_path / "schema.yaml"
        yaml_str = ar.schema_to_yaml(schema, path=yaml_file)
        
        # Read and parse YAML
        content = yaml_file.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        
        # Verify structure and sorting
        assert "fields" in parsed
        field_names = list(parsed["fields"].keys())
        assert field_names == sorted(field_names)
        assert field_names == ["user_email", "user_id", "user_name"]
        # Each field should have a type
        for field in parsed["fields"].values():
            assert isinstance(field, dict)
            assert "type" in field
