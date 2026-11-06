import pytest
import tempfile
from pathlib import Path
import arnio as ar
import pandas as pd


class TestSchemaToDictBasic:
    """Basic functionality for schema_to_dict"""

    def test_flat_dict_from_scan_csv(self):
        """Flat dict (scan_csv-style) should be normalized with fields wrapper."""
        schema = {"id": "int64", "name": "string", "active": "bool"}
        result = ar.schema_to_dict(schema)

        assert "fields" in result
        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["name"]["type"] == "STRING"
        assert result["fields"]["active"]["type"] == "BOOL"

    def test_fields_sorted_by_name(self):
        """Output fields should be sorted alphabetically."""
        schema = {"zebra": "string", "apple": "int64", "monkey": "float64"}
        result = ar.schema_to_dict(schema)

        field_names = list(result["fields"].keys())
        assert field_names == ["apple", "monkey", "zebra"]

    def test_structured_dict_with_metadata(self):
        """Dict with explicit 'fields' key preserves metadata."""
        schema = {
            "fields": {"user_id": "int64", "email": "string"},
            "strict": True,
            "unique": ["user_id"],
        }
        result = ar.schema_to_dict(schema)

        assert result["strict"] is True
        assert result["unique"] == ["user_id"]
        assert "fields" in result


class TestSchemaToDictObjects:
    """schema_to_dict with object types"""

    def test_schema_object_with_fields(self):
        """Schema object should extract and normalize fields."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        result = ar.schema_to_dict(schema)

        assert "fields" in result
        assert "id" in result["fields"]
        assert "name" in result["fields"]

    def test_schema_object_with_strict_unique(self):
        """Schema metadata (strict, unique) should be preserved."""
        schema = ar.Schema(
            {"id": ar.Int64()},
            strict=True,
            unique=["id"],
        )
        result = ar.schema_to_dict(schema)

        assert result.get("strict") is True
        assert result.get("unique") == ["id"]

    def test_arframe_object(self):
        """ArFrame with schema_summary should be converted."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        frame = ar.from_pandas(df)

        result = ar.schema_to_dict(frame)

        assert "fields" in result
        assert "id" in result["fields"]
        assert "name" in result["fields"]

    def test_list_of_dicts_with_name_dtype(self):
        """List of column dicts should be normalized."""
        schema = [
            {"name": "id", "dtype": "int64", "nullable": False},
            {"name": "email", "dtype": "string", "nullable": True},
        ]
        result = ar.schema_to_dict(schema)

        assert result["fields"]["id"]["type"] == "INT64"
        assert result["fields"]["id"]["nullable"] is False
        assert result["fields"]["email"]["type"] == "STRING"
        assert result["fields"]["email"]["nullable"] is True


class TestSchemaToDictErrors:
    """Error handling for schema_to_dict"""

    def test_schema_object_with_custom_rules_raises_error(self):
        """Schema with custom rules should raise ValueError."""
        def dummy_rule(df):
            return []

        schema = ar.Schema({"id": ar.Int64()}, rules=[dummy_rule])

        with pytest.raises(ValueError, match="does not support Schema objects with custom rules"):
            ar.schema_to_dict(schema)

    def test_unsupported_type_raises_typeerror(self):
        """Non-dict/non-object types should raise TypeError."""
        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict(12345)

        with pytest.raises(TypeError, match="Expected a dict"):
            ar.schema_to_dict("invalid_string")

    def test_non_string_metadata_key_raises_error(self):
        """Metadata keys must be strings."""
        schema = {
            "fields": {"id": "int64"},
            123: "value",  # non-string key
        }

        with pytest.raises(TypeError, match="schema metadata keys must be strings"):
            ar.schema_to_dict(schema)


class TestSchemaToYamlBasic:
    """Basic functionality for schema_to_yaml"""

    def test_returns_yaml_string(self):
        """Function should return YAML string."""
        schema = {"id": "int64", "name": "string"}
        result = ar.schema_to_yaml(schema)

        assert isinstance(result, str)
        assert "fields:" in result

    def test_yaml_ends_with_newline(self):
        """YAML output always ends with newline."""
        schema = {"id": "int64"}
        result = ar.schema_to_yaml(schema)

        assert result.endswith("\n")

    def test_yaml_contains_type_info(self):
        """YAML should contain type information."""
        schema = {"id": "int64", "active": "bool"}
        result = ar.schema_to_yaml(schema)

        # Should contain type references (exact format from YAML emitter)
        assert "INT64" in result or "int64" in result
        assert "BOOL" in result or "bool" in result


class TestSchemaToYamlFile:
    """File writing for schema_to_yaml"""

    def test_write_to_file(self):
        """YAML should be written to file when path provided."""
        schema = {"id": "int64"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.yaml"
            yaml_str = ar.schema_to_yaml(schema, path=filepath)

            assert filepath.exists()
            assert filepath.read_text(encoding="utf-8") == yaml_str

    def test_file_overwritten_if_exists(self):
        """Existing file should be overwritten."""
        schema = {"id": "int64"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.yaml"
            filepath.write_text("old content\n")

            ar.schema_to_yaml(schema, path=filepath)

            content = filepath.read_text(encoding="utf-8")
            assert "old content" not in content
            assert "fields:" in content

    def test_creates_parent_directories(self):
        """Parent directories should be created automatically."""
        schema = {"id": "int64"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "sub" / "nested" / "schema.yaml"

            ar.schema_to_yaml(schema, path=filepath)

            assert filepath.exists()
            assert filepath.parent.exists()

    def test_file_encoding_utf8(self):
        """File should be written with UTF-8 encoding."""
        schema = {"name": "string"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.yaml"
            ar.schema_to_yaml(schema, path=filepath)

            # Should read without encoding errors
            content = filepath.read_text(encoding="utf-8")
            assert isinstance(content, str)


class TestSchemaToYamlErrors:
    """Error handling for schema_to_yaml"""

    def test_path_not_string_or_pathlike_raises_error(self):
        """path must be string or PathLike."""
        schema = {"id": "int64"}

        with pytest.raises(TypeError, match="path must be a string or os.PathLike"):
            ar.schema_to_yaml(schema, path=123)

        with pytest.raises(TypeError, match="path must be a string or os.PathLike"):
            ar.schema_to_yaml(schema, path=[])

    def test_empty_string_path_raises_error(self):
        """Empty or whitespace-only path should raise ValueError."""
        schema = {"id": "int64"}

        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml(schema, path="")

        with pytest.raises(ValueError, match="path must not be empty"):
            ar.schema_to_yaml(schema, path="   ")

    def test_directory_path_raises_error(self):
        """path pointing to directory should raise ValueError."""
        schema = {"id": "int64"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="path must point to a file"):
                ar.schema_to_yaml(schema, path=tmpdir)


class TestSchemaToYamlRoundTrip:
    """schema_to_yaml with various schema types"""

    def test_schema_object_to_yaml(self):
        """Schema object should convert to YAML."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(),
        })
        result = ar.schema_to_yaml(schema)

        assert "fields:" in result
        assert result.endswith("\n")

    def test_dict_and_file_consistency(self):
        """Return value and file content should be identical."""
        schema = {"id": "int64", "name": "string"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.yaml"
            yaml_str = ar.schema_to_yaml(schema, path=filepath)
            file_content = filepath.read_text(encoding="utf-8")

            assert yaml_str == file_content
