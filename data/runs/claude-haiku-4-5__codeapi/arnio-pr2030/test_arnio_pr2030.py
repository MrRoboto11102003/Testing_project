import pytest
import tempfile
import pathlib
import os
import pandas as pd
import yaml
import arnio as ar
from arnio.schema import Schema, String, Int64


class TestSchemaToDictRawDict:
    """Tests for schema_to_dict with raw dicts."""
    
    def test_flat_dict_from_scan_csv_style(self):
        """Test conversion of flat dict (scan_csv output style)."""
        schema = {"name": "STRING", "age": "INT64", "score": "FLOAT64"}
        result = ar.schema_to_dict(schema)
        
        assert isinstance(result, dict)
        assert "fields" in result
        assert result["fields"]["name"]["type"] == "STRING"
        assert result["fields"]["age"]["type"] == "INT64"
    
    def test_dict_fields_sorted_alphabetically(self):
        """Verify that output fields are sorted alphabetically."""
        schema = {"zebra": "STRING", "apple": "INT64", "banana": "FLOAT64"}
        result = ar.schema_to_dict(schema)
        
        fields_keys = list(result["fields"].keys())
        assert fields_keys == ["apple", "banana", "zebra"]
    
    def test_structured_dict_with_fields_and_metadata(self):
        """Test dict with explicit 'fields' key and metadata."""
        schema = {
            "fields": {"id": "INT64", "name": "STRING"},
            "strict": True,
            "unique": ["id"],
        }
        result = ar.schema_to_dict(schema)
        
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["strict"] is True
        assert result["unique"] == ["id"]


class TestSchemaToDictSchemaObject:
    """Tests for schema_to_dict with Schema objects."""
    
    def test_schema_object_with_field_definitions(self):
        """Test conversion of Schema object with Field objects."""
        schema = Schema(fields={
            "id": Int64(),
            "name": String(),
        })
        result = ar.schema_to_dict(schema)
        
        assert "fields" in result
        assert "id" in result["fields"]
        assert "name" in result["fields"]
    
    def test_schema_metadata_strict_and_unique_preserved(self):
        """Test that Schema metadata (strict, unique) is preserved."""
        schema = Schema(
            fields={"id": Int64(), "name": String()},
            strict=True,
            unique=["id"],
        )
        result = ar.schema_to_dict(schema)
        
        assert result.get("strict") is True
        assert result.get("unique") == ["id"]
    
    def test_schema_with_custom_rules_raises_valueerror(self):
        """Test that Schema with custom rules raises ValueError."""
        def rule(df):
            return []
        
        schema = Schema(fields={"x": String()}, rules=[rule])
        
        with pytest.raises(ValueError, match="custom rules"):
            ar.schema_to_dict(schema)


class TestSchemaToDictArFrame:
    """Tests for schema_to_dict with ArFrame objects."""
    
    def test_arframe_schema_summary_conversion(self):
        """Test conversion from ArFrame using schema_summary."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        frame = ar.from_pandas(df)
        result = ar.schema_to_dict(frame)
        
        assert "fields" in result
        assert "name" in result["fields"]
        assert "age" in result["fields"]


class TestSchemaToDictErrors:
    """Tests for schema_to_dict error handling."""
    
    def test_unsupported_type_raises_typeerror(self):
        """Test that unsupported schema type raises TypeError."""
        with pytest.raises(TypeError, match="dict, an ArFrame"):
            ar.schema_to_dict(42)
    
    def test_string_input_raises_typeerror(self):
        """Test that string input raises TypeError."""
        with pytest.raises(TypeError, match="dict, an ArFrame"):
            ar.schema_to_dict("invalid")


class TestSchemaToYamlStringOutput:
    """Tests for schema_to_yaml string output."""
    
    def test_returns_yaml_string_with_newline(self):
        """Test that YAML string is returned and ends with newline."""
        schema = {"name": "STRING", "age": "INT64"}
        result = ar.schema_to_yaml(schema)
        
        assert isinstance(result, str)
        assert result.endswith("\n")
    
    def test_yaml_output_is_valid_and_parseable(self):
        """Test that output is valid YAML that can be parsed."""
        schema = {"name": "STRING", "age": "INT64"}
        result = ar.schema_to_yaml(schema)
        parsed = yaml.safe_load(result)
        
        assert parsed is not None
        assert "fields" in parsed
        assert parsed["fields"]["name"]["type"] == "STRING"


class TestSchemaToYamlFileWriting:
    """Tests for schema_to_yaml file writing."""
    
    def test_write_to_file_path_string(self):
        """Test writing YAML to a file path."""
        schema = {"name": "STRING"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "schema.yaml")
            result = ar.schema_to_yaml(schema, path=path)
            
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
            assert content == result
    
    def test_create_parent_directories_automatically(self):
        """Test that parent directories are created automatically."""
        schema = {"x": "STRING"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "schema.yaml")
            ar.schema_to_yaml(schema, path=path)
            
            assert os.path.exists(path)
    
    def test_pathlib_path_object_accepted(self):
        """Test that pathlib.Path objects are accepted."""
        schema = {"x": "STRING"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "schema.yaml"
            result = ar.schema_to_yaml(schema, path=path)
            
            assert path.exists()
            assert isinstance(result, str)


class TestSchemaToYamlErrorHandling:
    """Tests for schema_to_yaml error handling."""
    
    def test_directory_path_raises_valueerror(self):
        """Test that passing a directory path raises ValueError."""
        schema = {"x": "STRING"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="must point to a file"):
                ar.schema_to_yaml(schema, path=tmpdir)
    
    def test_empty_path_string_raises_valueerror(self):
        """Test that empty path string raises ValueError."""
        schema = {"x": "STRING"}
        with pytest.raises(ValueError, match="must not be empty"):
            ar.schema_to_yaml(schema, path="")
    
    def test_invalid_path_type_raises_typeerror(self):
        """Test that non-string/PathLike path raises TypeError."""
        schema = {"x": "STRING"}
        with pytest.raises(TypeError, match="path must be"):
            ar.schema_to_yaml(schema, path=123)
