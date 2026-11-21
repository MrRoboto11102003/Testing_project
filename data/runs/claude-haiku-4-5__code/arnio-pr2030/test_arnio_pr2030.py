import pytest
import tempfile
import pathlib
import os
from unittest.mock import Mock
import arnio as ar


class TestSchemaToDict:
    """Test arnio.schema_export.schema_to_dict function."""

    def test_flat_dict_simple(self):
        """Test conversion of flat dict with string type values."""
        schema = {"id": "INT64", "name": "STRING"}
        result = ar.schema_export.schema_to_dict(schema)
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["name"]["type"] == "STRING"

    def test_flat_dict_fields_sorted(self):
        """Test that output fields are sorted alphabetically."""
        schema = {"zebra": "STRING", "apple": "INT64"}
        result = ar.schema_export.schema_to_dict(schema)
        assert list(result["fields"].keys()) == ["apple", "zebra"]

    def test_structured_dict_with_metadata(self):
        """Test structured dict preserves strict and unique metadata."""
        schema = {
            "fields": {"id": "INT64"},
            "strict": True,
            "unique": ["id"]
        }
        result = ar.schema_export.schema_to_dict(schema)
        assert result["strict"] is True
        assert result["unique"] == ["id"]

    def test_schema_object_with_fields_attribute(self):
        """Test Schema object with fields attribute and strict/unique."""
        mock_schema = Mock()
        mock_schema.fields = {"id": "INT64", "name": "STRING"}
        mock_schema.rules = None
        mock_schema.strict = False
        
        result = ar.schema_export.schema_to_dict(mock_schema)
        assert "id" in result["fields"]
        assert result.get("strict") is False

    def test_arframe_with_schema_summary(self):
        """Test ArFrame extracts and converts schema_summary attribute."""
        mock_arframe = Mock()
        mock_arframe.schema_summary = {"value": "FLOAT64"}
        
        result = ar.schema_export.schema_to_dict(mock_arframe)
        assert result["fields"]["value"]["type"] == "FLOAT64"

    def test_list_of_columnsummary_objects(self):
        """Test list of ColumnSummary-like objects with dtype converted to uppercase."""
        mock_col = Mock()
        mock_col.name = "id"
        mock_col.dtype = "int64"
        mock_col.nullable = False
        
        result = ar.schema_export.schema_to_dict([mock_col])
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["id"]["nullable"] is False

    def test_list_of_dicts_with_name_dtype(self):
        """Test list of dicts with name and dtype keys."""
        schema = [
            {"name": "id", "dtype": "int64", "nullable": True},
            {"name": "value", "dtype": "float64"}
        ]
        result = ar.schema_export.schema_to_dict(schema)
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["id"]["nullable"] is True
        assert result["fields"]["value"]["type"] == "FLOAT64"

    def test_error_schema_with_rules_raises_valueerror(self):
        """Test ValueError when Schema has custom rules (not serializable)."""
        mock_schema = Mock()
        mock_schema.fields = {"id": "INT64"}
        mock_schema.rules = {"validation": lambda x: True}
        
        with pytest.raises(ValueError, match="schema_to_yaml does not support Schema objects with custom rules"):
            ar.schema_export.schema_to_dict(mock_schema)

    def test_error_unsupported_type_raises_typeerror(self):
        """Test TypeError for unsupported schema type."""
        with pytest.raises(TypeError, match="Expected a dict, an ArFrame, or an object with a 'fields' attribute"):
            ar.schema_export.schema_to_dict(42)

    def test_error_invalid_list_item_raises_typeerror(self):
        """Test TypeError for list item without required name/dtype attributes."""
        with pytest.raises(TypeError, match="Unsupported list item type"):
            ar.schema_export.schema_to_dict(["invalid_item"])

    def test_error_non_string_metadata_key_raises_typeerror(self):
        """Test TypeError for non-string metadata keys in dict."""
        schema = {"fields": {"id": "INT64"}, 999: "bad_key"}
        with pytest.raises(TypeError, match="schema metadata keys must be strings"):
            ar.schema_export.schema_to_dict(schema)


class TestSchemaToYaml:
    """Test arnio.schema_export.schema_to_yaml function."""

    def test_yaml_string_output(self):
        """Test basic YAML string generation."""
        schema = {"id": "INT64"}
        result = ar.schema_export.schema_to_yaml(schema)
        assert isinstance(result, str)
        assert "fields:" in result

    def test_yaml_always_ends_with_newline(self):
        """Test that returned YAML string always ends with newline."""
        schema = {"field": "STRING"}
        result = ar.schema_export.schema_to_yaml(schema)
        assert result.endswith("\n")

    def test_yaml_write_to_file_string_path(self):
        """Test writing YAML to file with string path."""
        schema = {"id": "INT64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "schema.yaml")
            result = ar.schema_export.schema_to_yaml(schema, path=filepath)
            
            assert os.path.exists(filepath)
            assert pathlib.Path(filepath).read_text(encoding="utf-8") == result

    def test_yaml_write_creates_parent_directories(self):
        """Test that parent directories are created when needed."""
        schema = {"id": "INT64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nested", "dir", "schema.yaml")
            ar.schema_export.schema_to_yaml(schema, path=filepath)
            assert os.path.exists(filepath)

    def test_yaml_write_with_pathlib_path(self):
        """Test path parameter as pathlib.Path object."""
        schema = {"id": "INT64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "schema.yaml"
            result = ar.schema_export.schema_to_yaml(schema, path=filepath)
            assert filepath.exists()
            assert filepath.read_text(encoding="utf-8") == result

    def test_yaml_error_invalid_path_type(self):
        """Test TypeError when path is not string or PathLike."""
        schema = {"id": "INT64"}
        with pytest.raises(TypeError, match="path must be a string or os.PathLike"):
            ar.schema_export.schema_to_yaml(schema, path=123)

    def test_yaml_error_empty_path_string(self):
        """Test ValueError when path is empty string."""
        schema = {"id": "INT64"}
        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_export.schema_to_yaml(schema, path="")

    def test_yaml_error_path_is_directory(self):
        """Test ValueError when path points to an existing directory."""
        schema = {"id": "INT64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="path must point to a file, not a directory"):
                ar.schema_export.schema_to_yaml(schema, path=tmpdir)
