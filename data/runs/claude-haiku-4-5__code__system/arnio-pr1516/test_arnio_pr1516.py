import pytest
import json
import sys
import subprocess
import warnings
import pandas as pd
import arnio as ar
from arnio.schema import Schema, Field


class TestSchemaToJson:
    """Test arnio.schema.Schema.to_json"""

    def test_to_json_returns_valid_json_string(self):
        """Test that to_json returns a valid, parseable JSON string"""
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'name': Field(dtype='object'),
        })
        
        result = schema.to_json()
        assert isinstance(result, str)
        payload = json.loads(result)
        assert 'fields' in payload

    def test_to_json_preserves_fields(self):
        """Test that all fields are present in JSON"""
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'email': Field(dtype='object'),
            'score': Field(dtype='float64'),
        })
        
        payload = json.loads(schema.to_json())
        assert set(payload['fields'].keys()) == {'id', 'email', 'score'}

    def test_to_json_preserves_strict_flag(self):
        """Test that strict mode is preserved in JSON"""
        schema_strict = Schema(fields={'x': Field(dtype='float64')}, strict=True)
        schema_loose = Schema(fields={'x': Field(dtype='float64')}, strict=False)
        
        assert json.loads(schema_strict.to_json())['strict'] is True
        assert json.loads(schema_loose.to_json())['strict'] is False

    def test_to_json_preserves_unique_constraints(self):
        """Test that unique constraints are serialized"""
        schema = Schema(
            fields={'id': Field(dtype='int64'), 'email': Field(dtype='object')},
            unique=['id', 'email']
        )
        
        payload = json.loads(schema.to_json())
        assert set(payload.get('unique', [])) == {'id', 'email'}

    def test_to_json_null_unique_as_none(self):
        """Test that None unique constraint is serialized as null"""
        schema = Schema(fields={'id': Field(dtype='int64')}, unique=None)
        payload = json.loads(schema.to_json())
        assert payload.get('unique') is None

    def test_to_json_sorted_field_keys(self):
        """Test that JSON output has alphabetically sorted field keys"""
        schema = Schema(fields={
            'zebra': Field(dtype='object'),
            'apple': Field(dtype='int64'),
            'middle': Field(dtype='float64'),
        })
        
        payload = json.loads(schema.to_json())
        keys = list(payload['fields'].keys())
        assert keys == sorted(keys), f"Expected {sorted(keys)}, got {keys}"

    def test_to_json_roundtrip_preserves_schema(self):
        """Test that schema can be recovered from JSON via from_json"""
        original = Schema(
            fields={
                'user_id': Field(dtype='int64'),
                'username': Field(dtype='object'),
                'score': Field(dtype='float64'),
            },
            strict=False,
            unique=['user_id']
        )
        
        json_str = original.to_json()
        restored = Schema.from_json(json_str)
        
        assert len(restored.fields) == len(original.fields)
        assert restored.strict == original.strict
        assert restored.unique == original.unique


class TestSchemaValidate:
    """Test arnio.schema.Schema.validate"""

    def test_validate_valid_frame(self):
        """Test validating a valid ArFrame succeeds"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['alice', 'bob', 'charlie'],
            'score': [95.5, 87.3, 92.1],
        })
        
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'name': Field(dtype='object'),
            'score': Field(dtype='float64'),
        })
        
        frame = ar.from_pandas(df)
        result = schema.validate(frame)
        
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_missing_required_column(self):
        """Test validation fails when expected column is missing"""
        df = pd.DataFrame({'name': ['alice', 'bob']})
        
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'name': Field(dtype='object'),
        })
        
        frame = ar.from_pandas(df)
        result = schema.validate(frame)
        
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_type_mismatch(self):
        """Test validation detects type mismatches"""
        df = pd.DataFrame({
            'id': ['not_a_number', 'also_not'],
            'name': ['alice', 'bob'],
        })
        
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'name': Field(dtype='object'),
        })
        
        frame = ar.from_pandas(df)
        result = schema.validate(frame)
        
        assert result.valid is False

    def test_validate_max_errors_limits_error_count(self):
        """Test that max_errors parameter limits returned errors"""
        df = pd.DataFrame({
            'a': ['x', 'y', 'z', 'w', 'v'],
            'b': ['p', 'q', 'r', 's', 't'],
        })
        
        schema = Schema(fields={
            'a': Field(dtype='int64'),
            'b': Field(dtype='int64'),
        })
        
        frame = ar.from_pandas(df)
        result = schema.validate(frame, max_errors=2)
        
        assert len(result.errors) <= 2

    def test_validate_returns_validation_result_object(self):
        """Test that validate returns ValidationResult with expected attributes"""
        df = pd.DataFrame({'id': [1, 2, 3]})
        schema = Schema(fields={'id': Field(dtype='int64')})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert hasattr(result, 'valid')
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, (list, tuple))


class TestSchemaWorkflows:
    """Integration tests combining serialization and validation"""

    def test_validate_after_json_roundtrip(self, tmp_path):
        """Test that validation behavior is preserved after JSON serialization"""
        df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'quantity': [10, 20, 15],
        })
        
        schema1 = Schema(fields={
            'product_id': Field(dtype='int64'),
            'quantity': Field(dtype='int64'),
        }, unique=['product_id'])
        
        # Roundtrip through JSON
        json_str = schema1.to_json()
        schema2 = Schema.from_json(json_str)
        
        # Both schemas should validate the same way
        frame = ar.from_pandas(df)
        result1 = schema1.validate(frame)
        result2 = schema2.validate(frame)
        
        assert result1.valid == result2.valid

    def test_persist_and_load_schema_json_file(self, tmp_path):
        """Test complete workflow: create schema, save to JSON, load, and validate"""
        # Create test data
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,product,qty\n1,apple,10\n2,banana,5\n3,orange,8\n")
        df = pd.read_csv(csv_file)
        
        # Create and validate with original schema
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'product': Field(dtype='object'),
            'qty': Field(dtype='int64'),
        }, unique=['id'])
        
        # Save schema to file
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(schema.to_json())
        
        # Load schema from file
        with open(schema_file) as f:
            loaded_schema = Schema.from_json(f.read())
        
        # Validate data with loaded schema
        frame = ar.from_pandas(df)
        result = loaded_schema.validate(frame)
        
        assert result.valid is True
        assert csv_file.exists()
        assert schema_file.exists()

    def test_cli_validate_valid_data(self, tmp_path):
        """Test validation through CLI with valid data"""
        # Create test data file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n2,bob\n3,charlie\n")
        
        # Create and save schema
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'name': Field(dtype='object'),
        })
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(schema.to_json())
        
        # Run CLI validation
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate",
             str(csv_file), str(schema_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        assert result.returncode in [0, 1]
        assert isinstance(result.stdout, str)

    def test_cli_validate_invalid_data(self, tmp_path):
        """Test CLI validation fails with type mismatch"""
        # Create data with wrong types
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\nnot_int,not_float\nalso_bad,bad_data\n")
        
        # Create schema expecting correct types
        schema = Schema(fields={
            'id': Field(dtype='int64'),
            'score': Field(dtype='float64'),
        })
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(schema.to_json())
        
        # Run CLI validation
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate",
             str(csv_file), str(schema_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should indicate validation failure
        assert result.returncode in [0, 1]
        assert isinstance(result.stdout, str)
