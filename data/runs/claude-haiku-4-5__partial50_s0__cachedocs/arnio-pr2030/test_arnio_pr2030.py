import pytest
import arnio as ar
from arnio import schema_export
import pathlib
import tempfile
import yaml


class TestSchemaToDict:
    """Tests for arnio.schema_export.schema_to_dict"""
    
    def test_empty_schema_to_dict(self):
        """Convert empty schema to dict"""
        schema = ar.Schema({})
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)
    
    def test_simple_schema_to_dict(self):
        """Convert schema with single field to dict"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False)
        })
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)
    
    def test_schema_with_multiple_types_to_dict(self):
        """Convert schema with various field types to dict"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "score": ar.Float64(nullable=True),
            "active": ar.Bool(),
            "email": ar.Email(),
        })
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)
    
    def test_schema_with_field_constraints_to_dict(self):
        """Convert schema with field constraints to dict"""
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=20),
            "age": ar.Int64(min=0, max=150),
        })
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)
    
    def test_plain_dict_to_schema_to_dict(self):
        """Pass plain dict directly to schema_to_dict"""
        schema_dict = {"id": "int64", "name": "string"}
        result = schema_export.schema_to_dict(schema_dict)
        assert isinstance(result, dict)
    
    def test_schema_with_unique_constraints_to_dict(self):
        """Convert schema with unique constraints to dict"""
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "email": ar.Email(),
        }, unique=["id"])
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)
    
    def test_schema_with_datetime_to_dict(self):
        """Convert schema with DateTime field to dict"""
        schema = ar.Schema({
            "created_at": ar.DateTime(format="%Y-%m-%d", nullable=False),
        })
        result = schema_export.schema_to_dict(schema)
        assert isinstance(result, dict)


class TestSchemaToYaml:
    """Tests for arnio.schema_export.schema_to_yaml"""
    
    def test_schema_to_yaml_returns_string(self):
        """schema_to_yaml returns non-empty string"""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema_export.schema_to_yaml(schema)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_yaml_output_is_valid(self):
        """Returned YAML string is parseable"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = schema_export.schema_to_yaml(schema)
        parsed = yaml.safe_load(result)
        assert parsed is not None
    
    def test_yaml_with_file_write_pathlib(self):
        """schema_to_yaml writes to file with pathlib.Path"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "test_schema.yaml"
            result = schema_export.schema_to_yaml(schema, path=filepath)
            
            assert isinstance(result, str)
            assert filepath.exists()
            
            with open(filepath) as f:
                content = f.read()
                parsed = yaml.safe_load(content)
                assert parsed is not None
    
    def test_yaml_with_file_write_string_path(self):
        """schema_to_yaml writes to file with string path"""
        schema = ar.Schema({"id": ar.Int64()})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(pathlib.Path(tmpdir) / "schema.yaml")
            result = schema_export.schema_to_yaml(schema, path=filepath)
            assert pathlib.Path(filepath).exists()
    
    def test_yaml_without_path_returns_string(self):
        """schema_to_yaml without path returns string, no file written"""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema_export.schema_to_yaml(schema)
        
        assert isinstance(result, str)
        assert len(result) > 0
        parsed = yaml.safe_load(result)
        assert parsed is not None
    
    def test_yaml_complex_schema(self):
        """schema_to_yaml handles complex schema with multiple constraints"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(min=0, max=150),
            "created_at": ar.DateTime(format="%Y-%m-%d"),
            "country": ar.CountryCode(),
        })
        result = schema_export.schema_to_yaml(schema)
        assert isinstance(result, str)
        parsed = yaml.safe_load(result)
        assert parsed is not None
    
    def test_yaml_with_dict_input(self):
        """schema_to_yaml works with plain dict input"""
        schema_dict = {"id": "int64", "name": "string"}
        result = schema_export.schema_to_yaml(schema_dict)
        assert isinstance(result, str)
        parsed = yaml.safe_load(result)
        assert parsed is not None
