import pytest
import sys
import subprocess
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

import arnio as ar


class TestSchemaValidate:
    """End-to-end tests for arnio.schema.Schema.validate()"""

    def test_validate_valid_data(self, tmp_path):
        """Test that valid data passes schema validation"""
        csv_file = tmp_path / "valid.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        result = schema.validate(frame)
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_type_mismatch(self, tmp_path):
        """Test detection of type mismatches"""
        csv_file = tmp_path / "wrong_types.csv"
        df = pd.DataFrame({
            'id': ['one', 'two', 'three'],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(
            pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C'], 'age': [25, 30, 35]})
        )
        result = schema.validate(frame)
        
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_missing_column(self, tmp_path):
        """Test detection of missing columns"""
        csv_file = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'age': [25, 30, 35]
        })
        schema = ar.Schema.from_frame(schema_df)
        result = schema.validate(frame)
        
        assert result.is_valid is False
        assert any('age' in str(e).lower() for e in result.errors)

    def test_validate_max_errors_parameter(self, tmp_path):
        """Test that max_errors limits error reporting"""
        csv_file = tmp_path / "many_errors.csv"
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['d', 'e', 'f'],
            'col3': ['g', 'h', 'i']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(
            pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [1.0, 2.0, 3.0],
                'col3': [True, False, True]
            })
        )
        
        result = schema.validate(frame, max_errors=2)
        
        assert result.is_valid is False
        assert len(result.errors) <= 2

    def test_validate_max_errors_none(self, tmp_path):
        """Test that max_errors=None reports all errors"""
        csv_file = tmp_path / "all_errors.csv"
        df = pd.DataFrame({
            'a': ['x', 'y'],
            'b': ['p', 'q'],
            'c': ['m', 'n']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(
            pd.DataFrame({'a': [1, 2], 'b': [1.0, 2.0], 'c': [True, False]})
        )
        
        result_all = schema.validate(frame, max_errors=None)
        result_limited = schema.validate(frame, max_errors=1)
        
        assert len(result_all.errors) >= len(result_limited.errors)

    def test_validate_empty_dataframe(self, tmp_path):
        """Test validation of empty dataframe"""
        csv_file = tmp_path / "empty.csv"
        df = pd.DataFrame({
            'id': pd.Series([], dtype='int64'),
            'name': pd.Series([], dtype='object')
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(
            pd.DataFrame({'id': [1], 'name': ['Alice']})
        )
        result = schema.validate(frame)
        
        assert isinstance(result.is_valid, bool)

    def test_validate_null_values(self, tmp_path):
        """Test validation with null/missing values"""
        csv_file = tmp_path / "nulls.csv"
        df = pd.DataFrame({
            'id': [1, 2, np.nan],
            'name': ['Alice', None, 'Charlie']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(
            pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        )
        result = schema.validate(frame)
        
        assert isinstance(result.is_valid, bool)
        assert hasattr(result, 'errors')

    def test_validate_validation_result_attributes(self, tmp_path):
        """Test that ValidationResult has expected attributes"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        result = schema.validate(frame)
        
        assert hasattr(result, 'is_valid')
        assert isinstance(result.is_valid, bool)
        assert hasattr(result, 'errors')
        assert isinstance(result.errors, (list, tuple))

    def test_validate_via_pandas_accessor(self, tmp_path):
        """Test validation using pandas .arnio accessor"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        
        result = frame.arnio.validate(schema)
        
        assert result.is_valid is True

    def test_validate_cli_valid_data(self, tmp_path):
        """Test validation via CLI with valid data"""
        csv_file = tmp_path / "valid.csv"
        df = pd.DataFrame({'id': [1, 2], 'value': [10.5, 20.3]})
        df.to_csv(csv_file, index=False)
        
        schema_file = tmp_path / "schema.json"
        schema_dict = {
            'columns': {
                'id': {'type': 'int'},
                'value': {'type': 'float'}
            }
        }
        schema_file.write_text(json.dumps(schema_dict))
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate",
             str(csv_file), "--schema", str(schema_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0

    def test_validate_cli_invalid_data(self, tmp_path):
        """Test validation via CLI with invalid data"""
        csv_file = tmp_path / "invalid.csv"
        df = pd.DataFrame({'id': ['not_int', 'also_not'], 'value': [10.5, 20.3]})
        df.to_csv(csv_file, index=False)
        
        schema_file = tmp_path / "schema.json"
        schema_dict = {
            'columns': {
                'id': {'type': 'int'},
                'value': {'type': 'float'}
            }
        }
        schema_file.write_text(json.dumps(schema_dict))
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate",
             str(csv_file), "--schema", str(schema_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0

    def test_validate_after_read_and_clean(self, tmp_path):
        """Test validation on cleaned data end-to-end"""
        csv_file = tmp_path / "messy.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['  Alice  ', 'Bob', 'Charlie  '],
            'score': [95.5, 87.3, 92.1]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        
        result = schema.validate(frame)
        
        assert isinstance(result.is_valid, bool)
        assert hasattr(result, 'errors')

    def test_validate_duplicate_rows(self, tmp_path):
        """Test validation with duplicate rows"""
        csv_file = tmp_path / "dupes.csv"
        df = pd.DataFrame({
            'id': [1, 1, 2],
            'name': ['Alice', 'Alice', 'Bob'],
            'value': [100, 100, 200]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        result = schema.validate(frame)
        
        assert isinstance(result.is_valid, bool)

    def test_validate_large_dataframe(self, tmp_path):
        """Test validation on larger dataframe"""
        csv_file = tmp_path / "large.csv"
        n_rows = 10000
        df = pd.DataFrame({
            'id': range(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows)
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(csv_file)
        schema = ar.Schema.from_frame(frame)
        result = schema.validate(frame)
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_mixed_error_types(self, tmp_path):
        """Test validation detects multiple error types"""
        csv_file = tmp_path / "mixed.csv"
        df = pd.DataFrame({
            'id': ['not_int', 2, 3],
            'value': [1.0, 2.0, 3.0],
            'extra': [1, 2, 3]
        })
        df.to_csv(csv_file, index=False)
        
        schema = ar.Schema.from_frame(
            pd.DataFrame({'id': [1, 2, 3], 'value': [1.0, 2.0, 3.0]})
        )
        
        frame = ar.read_csv(csv_file)
        result = schema.validate(frame)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
