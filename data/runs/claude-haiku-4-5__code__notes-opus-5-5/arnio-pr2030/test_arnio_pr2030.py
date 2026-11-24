import pytest
import re
from pathlib import Path
from types import SimpleNamespace

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestSchemaToDict:
    """Tests for arnio.schema_export.schema_to_dict"""
    
    def test_empty_dict(self):
        """Empty dict should return {"fields": {}}"""
        result = ar.schema_to_dict({})
        assert result == {"fields": {}}
    
    def test_flat_dict_sorts_fields(self):
        """Flat dict fields should be sorted by name"""
        result = ar.schema_to_dict({"b": "int64", "a": "string"})
        assert list(result["fields"].keys()) == ["a", "b"]
        assert result["fields"]["a"]["type"] == "string"
        assert result["fields"]["b"]["type"] == "int64"
    
    def test_structured_dict_with_metadata(self):
        """Structured dict sorts fields and preserves metadata"""
        schema = {
            "fields": {"z": "int64", "a": "string"},
            "strict": True,
            "unique": ("id",)
        }
        result = ar.schema_to_dict(schema)
        assert list(result["fields"].keys()) == ["a", "z"]
        assert result["strict"] is True
        assert result["unique"] == ["id"]  # Tuple normalized to list
    
    def test_non_str_field_name_raises(self):
        """Non-string field name raises TypeError"""
        with pytest.raises(TypeError, match=re.escape("schema field names must be strings")):
            ar.schema_to_dict({1: "int64"})
    
    def test_simple_namespace_with_fields(self):
        """SimpleNamespace with fields attribute is processed"""
        schema = SimpleNamespace(
            fields={"a": "int64"},
            strict=False,
            unique=("id",)
        )
        result = ar.schema_to_dict(schema)
        assert result["fields"]["a"]["type"] == "int64"
        assert result["strict"] is False
        assert result["unique"] == ("id",)  # Tuple preserved in schema path
    
    def test_empty_list(self):
        """Empty list returns {"fields": {}}"""
        result = ar.schema_to_dict([])
        assert result == {"fields": {}}
    
    def test_list_of_column_summary_like_objects(self):
        """List of objects with name, dtype, nullable attributes"""
        col_a = SimpleNamespace(name="a", dtype="int64", nullable=False)
        col_b = SimpleNamespace(name="b", dtype="string", nullable=True)
        result = ar.schema_to_dict([col_a, col_b])
        
        # Type is uppercased in list path
        assert result["fields"]["a"]["type"] == "INT64"
        assert result["fields"]["a"]["nullable"] is False
        assert result["fields"]["b"]["type"] == "STRING"
        assert result["fields"]["b"]["nullable"] is True
        # Order preserved from input (not sorted)
        assert list(result["fields"].keys()) == ["a", "b"]
    
    def test_list_of_dicts_with_dtype(self):
        """List of dicts with name and dtype keys"""
        result = ar.schema_to_dict([{"name": "x", "dtype": "float64"}])
        assert result["fields"]["x"]["type"] == "FLOAT64"
        assert "nullable" not in result["fields"]["x"]
    
    def test_list_invalid_entry_raises(self):
        """List entry without required name/dtype raises TypeError"""
        with pytest.raises(TypeError, match=re.escape("Unsupported list item type:")):
            ar.schema_to_dict([{"dtype": "int64"}])
    
    def test_schema_with_truthy_rules_raises(self):
        """Schema with truthy rules raises ValueError"""
        schema = SimpleNamespace(fields={}, rules=[lambda df: []])
        msg = "Remove schema.rules before exporting"
        with pytest.raises(ValueError, match=re.escape(msg)):
            ar.schema_to_dict(schema)
    
    def test_unsupported_type_raises(self):
        """Unsupported input type raises TypeError"""
        with pytest.raises(TypeError, match=re.escape("Expected a dict, an ArFrame")):
            ar.schema_to_dict(42)
    
    def test_non_str_metadata_key_raises(self):
        """Non-string metadata key in structured dict raises TypeError"""
        with pytest.raises(TypeError, match=re.escape("schema metadata keys must be strings")):
            ar.schema_to_dict({"fields": {}, 1: "value"})


class TestSchemaToYaml:
    """Tests for arnio.schema_export.schema_to_yaml"""
    
    def test_simple_yaml_output(self):
        """Simple schema produces valid YAML string"""
        result = ar.schema_to_yaml({"id": "INT64", "name": "STRING"})
        assert result.startswith("fields:")
        assert "id:" in result
        assert "name:" in result
        assert result.endswith("\n")
    
    def test_empty_dict_yaml(self):
        """Empty dict produces 'fields: {}'"""
        result = ar.schema_to_yaml({})
        assert result == "fields: {}\n"
    
    def test_yaml_with_metadata_sorted_keys(self):
        """YAML top-level keys are sorted (fields, strict, unique)"""
        result = ar.schema_to_yaml({
            "fields": {"b": "int"},
            "strict": True,
            "unique": None
        })
        f_idx = result.index("fields:")
        s_idx = result.index("strict:")
        u_idx = result.index("unique:")
        assert f_idx < s_idx < u_idx
        assert result.endswith("\n")
    
    def test_yaml_always_ends_with_newline(self):
        """YAML output always ends with exactly one newline"""
        for schema in [{}, {"a": "int"}, []]:
            result = ar.schema_to_yaml(schema)
            assert result.endswith("\n")
            assert not result.endswith("\n\n")
    
    def test_write_to_file(self, tmp_path):
        """schema_to_yaml writes to file when path is provided"""
        output_file = tmp_path / "schema.yaml"
        schema = {"id": "INT64"}
        
        result = ar.schema_to_yaml(schema, path=output_file)
        
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == result
    
    def test_write_creates_parent_directories(self, tmp_path):
        """schema_to_yaml creates missing parent directories"""
        output_file = tmp_path / "nested" / "sub" / "schema.yaml"
        result = ar.schema_to_yaml({}, path=output_file)
        
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == result
    
    def test_write_overwrites_existing_file(self, tmp_path):
        """schema_to_yaml overwrites existing file"""
        output_file = tmp_path / "schema.yaml"
        output_file.write_text("old content", encoding="utf-8")
        
        result = ar.schema_to_yaml({"a": "int"}, path=output_file)
        
        assert output_file.read_text(encoding="utf-8") == result
        assert "old" not in output_file.read_text(encoding="utf-8")
    
    def test_path_type_must_be_string_or_pathlike(self, tmp_path):
        """path must be string or os.PathLike"""
        with pytest.raises(TypeError, match=re.escape("path must be a string or os.PathLike")):
            ar.schema_to_yaml({}, path=123)
    
    def test_path_cannot_be_empty_string(self):
        """path cannot be empty or whitespace-only string"""
        with pytest.raises(ValueError, match=re.escape("path must not be empty")):
            ar.schema_to_yaml({}, path="")
        
        with pytest.raises(ValueError, match=re.escape("path must not be empty")):
            ar.schema_to_yaml({}, path="   ")
    
    def test_path_cannot_be_directory(self, tmp_path):
        """path cannot point to a directory"""
        with pytest.raises(ValueError, match=re.escape("path must point to a file, not a directory")):
            ar.schema_to_yaml({}, path=tmp_path)
    
    def test_schema_error_raised_before_path_validation(self, tmp_path):
        """Schema conversion error is raised before path checks"""
        # Schema error (bad type) with invalid path (directory)
        with pytest.raises(TypeError, match=re.escape("Expected a dict, an ArFrame")):
            ar.schema_to_yaml(42, path=tmp_path)
