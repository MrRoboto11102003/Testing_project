import pytest
import re
from types import SimpleNamespace

import pandas as pd
import arnio as ar


class TestSchemaToDictAndYaml:
    """Comprehensive tests for schema_to_dict and schema_to_yaml."""

    # ─── schema_to_dict: basic paths ───
    def test_flat_dict_normalized_and_sorted(self):
        """Flat dict is normalized to sorted fields."""
        result = ar.schema_to_dict({"b": "int64", "a": "string"})
        assert list(result["fields"]) == ["a", "b"]
        assert result["fields"]["a"]["type"] == "string"
        assert result["fields"]["b"]["type"] == "int64"

    def test_structured_dict_with_metadata(self):
        """Structured dict preserves and normalizes metadata."""
        schema = {"fields": {"x": "int64"}, "strict": True, "unique": ("id",)}
        result = ar.schema_to_dict(schema)
        assert result["fields"]["x"]["type"] == "int64"
        assert result["strict"] is True
        assert result["unique"] == ["id"]  # Top-level tuple converted to list

    def test_list_path_uppercases_types_and_includes_nullable(self):
        """List path uppercases dtypes and includes nullable when present."""
        schema = [
            {"name": "x", "dtype": "int64", "nullable": False},
            {"name": "y", "dtype": "string"},
        ]
        result = ar.schema_to_dict(schema)
        assert result["fields"]["x"]["type"] == "INT64"
        assert result["fields"]["x"]["nullable"] is False
        assert "nullable" not in result["fields"]["y"]

    def test_schema_object_sorts_fields_preserves_unique_as_tuple(self):
        """Schema-like object sorts fields; unique stays as tuple (not normalized)."""
        schema = SimpleNamespace(
            fields={"z": "int64", "a": "string"},
            strict=False,
            unique=("a",),
        )
        result = ar.schema_to_dict(schema)
        assert list(result["fields"]) == ["a", "z"]
        assert result["unique"] == ("a",)  # Tuple NOT converted on Schema path

    def test_arframe_converted_via_schema_summary(self):
        """ArFrame input is converted via schema_summary attribute."""
        df = pd.DataFrame({"b": [1], "a": ["x"]})
        frame = ar.from_pandas(df)
        result = ar.schema_to_dict(frame)
        
        assert "fields" in result
        assert list(result["fields"]) == ["b", "a"]  # Input order preserved

    def test_empty_inputs_return_empty_fields(self):
        """Empty dict and empty list both return empty fields."""
        assert ar.schema_to_dict({}) == {"fields": {}}
        assert ar.schema_to_dict([]) == {"fields": {}}

    # ─── schema_to_dict: error conditions ───
    def test_schema_to_dict_invalid_input_types_raise_typeerror(self):
        """Non-schema input types raise TypeError with correct message."""
        with pytest.raises(TypeError, match=re.escape("got <class 'int'>.")):
            ar.schema_to_dict(42)
        
        with pytest.raises(TypeError, match="Unsupported list item type"):
            ar.schema_to_dict([{"name": "a"}])  # Missing 'dtype'

    def test_schema_to_dict_non_string_field_and_metadata_names_raise(self):
        """Non-string field or metadata keys raise TypeError."""
        with pytest.raises(TypeError, match="field names must be strings"):
            ar.schema_to_dict({1: "int64"})
        
        with pytest.raises(TypeError, match="metadata keys must be strings"):
            ar.schema_to_dict({"fields": {}, 1: "x"})

    def test_schema_to_dict_with_callable_rules_raises_valueerror(self):
        """Schema with truthy rules attribute raises ValueError."""
        schema = SimpleNamespace(fields={}, rules=[lambda: None])
        with pytest.raises(ValueError, match="custom rules"):
            ar.schema_to_dict(schema)

    # ─── schema_to_yaml: basic functionality ───
    def test_schema_to_yaml_produces_deterministic_sorted_output(self):
        """schema_to_yaml produces sorted keys and ends with newline."""
        result = ar.schema_to_yaml({"b": "int64", "a": "string"})
        
        assert result.endswith("\n")
        # Verify 'a' comes before 'b' (sorted)
        a_idx = result.index("a:")
        b_idx = result.index("b:")
        assert a_idx < b_idx

    def test_schema_to_yaml_empty_input(self):
        """Empty dict produces minimal YAML output."""
        result = ar.schema_to_yaml({})
        assert result == "fields: {}\n"

    # ─── schema_to_yaml: file I/O ───
    def test_schema_to_yaml_writes_file_when_path_provided(self, tmp_path):
        """schema_to_yaml writes to file and returns same string."""
        output_file = tmp_path / "schema.yaml"
        result = ar.schema_to_yaml({"x": "int64"}, path=output_file)
        
        assert output_file.exists()
        file_content = output_file.read_text(encoding="utf-8")
        assert file_content == result

    def test_schema_to_yaml_creates_parents_and_overwrites_file(self, tmp_path):
        """schema_to_yaml creates parent dirs and overwrites existing files."""
        output_file = tmp_path / "sub" / "dir" / "schema.yaml"
        
        ar.schema_to_yaml({"new": "int64"}, path=output_file)
        assert output_file.exists()
        
        # Verify it overwrites
        output_file.write_text("old content\n", encoding="utf-8")
        ar.schema_to_yaml({"newer": "string"}, path=output_file)
        
        content = output_file.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "newer" in content

    # ─── schema_to_yaml: error conditions ───
    def test_schema_to_yaml_invalid_schema_raises_typeerror(self):
        """Invalid schema input raises TypeError."""
        with pytest.raises(TypeError, match="got <class 'int'>"):
            ar.schema_to_yaml(42)

    def test_schema_to_yaml_invalid_path_raises_error(self, tmp_path):
        """Invalid path raises appropriate errors."""
        # Invalid path type
        with pytest.raises(TypeError, match="path must be a string"):
            ar.schema_to_yaml({}, path=123)
        
        # Empty path
        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml({}, path="")
        
        # Directory instead of file
        with pytest.raises(ValueError, match="path must point to a file"):
            ar.schema_to_yaml({}, path=tmp_path)
