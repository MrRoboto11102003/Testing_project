import pytest
import tempfile
import pathlib
from unittest.mock import Mock
import arnio as ar


class TestSchemaToDictConversion:
    """Test schema_to_dict type conversion and normalization"""
    
    def test_flat_dict_input(self):
        """Convert flat dict with uppercase type normalization"""
        result = ar.schema_export.schema_to_dict({"id": "int64", "name": "string"})
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["name"]["type"] == "STRING"
    
    def test_structured_dict_with_metadata(self):
        """Preserve strict and unique metadata in structured dict"""
        schema = {
            "fields": {"id": "int64"},
            "strict": True,
            "unique": ["id"]
        }
        result = ar.schema_export.schema_to_dict(schema)
        assert result["strict"] is True
        assert result["unique"] == ["id"]
    
    def test_sorted_output_with_nullable(self):
        """Output fields are sorted and preserve nullable attribute"""
        schema = {
            "z": {"type": "string", "nullable": True},
            "a": "int64"
        }
        result = ar.schema_export.schema_to_dict(schema)
        assert list(result["fields"].keys()) == ["a", "z"]
        assert result["fields"]["z"]["nullable"] is True


class TestSchemaToDictObjects:
    """Test schema_to_dict with non-dict object types"""
    
    def test_schema_object_and_metadata(self):
        """Accept Schema object with fields and metadata"""
        mock_schema = Mock()
        mock_schema.fields = {"id": "int64"}
        mock_schema.strict = False
        result = ar.schema_export.schema_to_dict(mock_schema)
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["strict"] is False
    
    def test_arframe_and_list_inputs(self):
        """Accept ArFrame with schema_summary and list of objects"""
        # Test ArFrame
        mock_frame = Mock()
        mock_frame.schema_summary = {"id": "int64", "value": "float64"}
        result1 = ar.schema_export.schema_to_dict(mock_frame)
        assert "id" in result1["fields"]
        assert "value" in result1["fields"]
        
        # Test list of ColumnSummary-like objects
        col = Mock(name="id", dtype="int64", nullable=False)
        result2 = ar.schema_export.schema_to_dict([col])
        assert result2["fields"]["id"]["nullable"] is False


class TestSchemaToDictErrors:
    """Test schema_to_dict error handling"""
    
    def test_error_cases(self):
        """Raise errors for invalid inputs: unsupported type, bad list items, bad keys"""
        # Unsupported type
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_export.schema_to_dict(123)
        
        # Invalid list item
        with pytest.raises(TypeError, match="Unsupported list item type"):
            ar.schema_export.schema_to_dict([{"invalid": "entry"}])
        
        # Non-string metadata key
        with pytest.raises(TypeError, match="metadata keys must be strings"):
            ar.schema_export.schema_to_dict({"fields": {"id": "int64"}, 123: "val"})
    
    def test_schema_with_custom_rules_error(self):
        """Reject Schema with custom rules (non-serializable)"""
        mock_schema = Mock()
        mock_schema.fields = {"id": "int64"}
        mock_schema.rules = Mock()
        with pytest.raises(ValueError, match="custom rules"):
            ar.schema_export.schema_to_dict(mock_schema)


class TestSchemaToYamlConversion:
    """Test schema_to_yaml string output"""
    
    def test_yaml_format_and_structure(self):
        """YAML output has correct format with newline termination"""
        result = ar.schema_export.schema_to_yaml({"id": "int64"})
        assert isinstance(result, str)
        assert result.endswith("\n")
        assert "fields:" in result
        assert "INT64" in result


class TestSchemaToYamlFileOutput:
    """Test schema_to_yaml file writing"""
    
    def test_file_writing_and_sync(self):
        """Write YAML to file and verify content matches return value"""
        schema = {"id": "int64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "schema.yaml"
            result = ar.schema_export.schema_to_yaml(schema, path=filepath)
            assert filepath.exists()
            assert filepath.read_text(encoding="utf-8") == result
    
    def test_creates_parent_directories(self):
        """Create parent directories if they don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "a" / "b" / "schema.yaml"
            ar.schema_export.schema_to_yaml({"id": "int64"}, path=filepath)
            assert filepath.exists()


class TestSchemaToYamlErrors:
    """Test schema_to_yaml error handling"""
    
    def test_error_cases(self):
        """Raise errors for invalid path: wrong type, empty, directory"""
        schema = {"id": "int64"}
        
        # Invalid path type
        with pytest.raises(TypeError, match="path must be a string"):
            ar.schema_export.schema_to_yaml(schema, path=123)
        
        # Empty path string
        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_export.schema_to_yaml(schema, path="")
        
        # Directory path
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="must point to a file"):
                ar.schema_export.schema_to_yaml(schema, path=tmpdir)
