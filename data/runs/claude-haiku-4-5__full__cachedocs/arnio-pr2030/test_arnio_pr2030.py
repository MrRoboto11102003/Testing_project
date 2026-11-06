import pytest
import tempfile
import pathlib
import arnio as ar
import pandas as pd
import yaml
import os


class TestSchemaToDict:
    """Tests for ar.schema_to_dict()"""

    def test_dict_input_returns_dict(self):
        """schema_to_dict accepts a plain dict and returns a dict."""
        schema = {"id": "int64", "name": "string"}
        result = ar.schema_to_dict(schema)
        assert isinstance(result, dict)
        assert "id" in result or len(result) > 0

    def test_scan_csv_dict_returns_dict(self):
        """schema_to_dict accepts output from ar.scan_csv()."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name\n1,Alice\n2,Bob\n")
            f.flush()
            try:
                schema = ar.scan_csv(f.name)
                result = ar.schema_to_dict(schema)
                assert isinstance(result, dict)
            finally:
                os.unlink(f.name)

    def test_schema_object_input(self):
        """schema_to_dict accepts arnio Schema object with fields."""
        schema_obj = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True),
        })
        result = ar.schema_to_dict(schema_obj)
        assert isinstance(result, dict)

    def test_returns_stdlib_serializable_values(self):
        """Result contains only stdlib-serializable values."""
        schema = {"id": "int64", "name": "string", "active": "bool"}
        result = ar.schema_to_dict(schema)
        import json
        json.dumps(result)  # Should not raise

    def test_unsupported_type_int_raises_typeerror(self):
        """Unsupported type (int) raises TypeError."""
        with pytest.raises(TypeError):
            ar.schema_to_dict(12345)

    def test_unsupported_type_none_raises_typeerror(self):
        """Unsupported type (None) raises TypeError."""
        with pytest.raises(TypeError):
            ar.schema_to_dict(None)

    def test_unsupported_type_string_raises_typeerror(self):
        """Unsupported type (string) raises TypeError."""
        with pytest.raises(TypeError):
            ar.schema_to_dict("not a schema")

    def test_unsupported_type_list_raises_typeerror(self):
        """Unsupported type (list) raises TypeError."""
        with pytest.raises(TypeError):
            ar.schema_to_dict([1, 2, 3])

    def test_arframe_input(self):
        """schema_to_dict accepts ArFrame objects."""
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        frame = ar.from_pandas(df)
        result = ar.schema_to_dict(frame)
        assert isinstance(result, dict)


class TestSchemaToYaml:
    """Tests for ar.schema_to_yaml()"""

    def test_dict_input_returns_yaml_string(self):
        """schema_to_yaml returns string for dict input."""
        schema = {"id": "int64", "name": "string"}
        result = ar.schema_to_yaml(schema)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_yaml_string_ends_with_newline(self):
        """Returned YAML string ends with newline character."""
        schema = {"id": "int64"}
        result = ar.schema_to_yaml(schema)
        assert result.endswith("\n")

    def test_schema_object_input_returns_yaml(self):
        """schema_to_yaml accepts arnio Schema object."""
        schema_obj = ar.Schema({"id": ar.Int64(nullable=False)})
        result = ar.schema_to_yaml(schema_obj)
        assert isinstance(result, str)
        assert result.endswith("\n")

    def test_yaml_output_is_valid_yaml(self):
        """Returned string is valid YAML and parseable."""
        schema = {"id": "int64", "name": "string"}
        result = ar.schema_to_yaml(schema)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)

    def test_writes_to_file_when_path_provided(self):
        """schema_to_yaml writes YAML to file when path is provided."""
        schema = {"id": "int64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "schema.yaml"
            ar.schema_to_yaml(schema, path=path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_returns_string_even_when_path_provided(self):
        """schema_to_yaml returns string regardless of path parameter."""
        schema = {"id": "int64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "schema.yaml"
            result = ar.schema_to_yaml(schema, path=path)
            assert isinstance(result, str)
            assert result.endswith("\n")

    def test_file_content_matches_return_value(self):
        """Content written to file matches the returned string."""
        schema = {"id": "int64", "name": "string"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "schema.yaml"
            result = ar.schema_to_yaml(schema, path=path)
            file_content = path.read_text(encoding="utf-8")
            assert result == file_content

    def test_overwrites_existing_file(self):
        """Writing to existing file overwrites its content."""
        schema = {"id": "int64"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "schema.yaml"
            path.write_text("old content\n", encoding="utf-8")
            ar.schema_to_yaml(schema, path=path)
            content = path.read_text(encoding="utf-8")
            assert "old content" not in content
            assert "id" in content or "int64" in content
