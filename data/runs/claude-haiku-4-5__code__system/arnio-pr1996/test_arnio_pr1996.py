import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
import csv
from pathlib import Path
import arnio as ar


class TestDropEmptyColumnsEndToEnd:
    """End-to-end system tests for arnio.cleaning.drop_empty_columns"""

    def test_basic_single_empty_column(self, tmp_path):
        """Remove a single fully empty column from CSV"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'empty_col': [None, None, None]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'empty_col' not in result_df.columns
        assert list(result_df.columns) == ['name', 'age']
        assert len(result_df) == 3

    def test_multiple_empty_columns(self, tmp_path):
        """Remove multiple empty columns"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'empty1': [None, None],
            'value': [100, 200],
            'empty2': [None, None],
            'city': ['NYC', 'LA']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert list(result_df.columns) == ['name', 'value', 'city']

    def test_whitespace_only_strings_are_empty(self, tmp_path):
        """Columns with only whitespace strings are treated as empty"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'whitespace_col'])
            writer.writerow(['Alice', '   '])
            writer.writerow(['Bob', '\t'])
            writer.writerow(['Charlie', '  \t  '])
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'whitespace_col' not in result_df.columns
        assert 'name' in result_df.columns
        assert len(result_df) == 3

    def test_no_empty_columns(self, tmp_path):
        """When no columns are empty, all columns are retained"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30],
            'city': ['NYC', 'LA']
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert list(result_df.columns) == ['name', 'age', 'city']

    def test_all_columns_empty(self, tmp_path):
        """When all columns are empty, result has no columns"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'empty1': [None, None, None],
            'empty2': [None, None, None]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3

    def test_empty_frame_zero_rows(self, tmp_path):
        """Empty frame (0 rows) preserves all columns"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'name': pd.Series([], dtype=str),
            'age': pd.Series([], dtype=int),
            'empty': pd.Series([], dtype=object)
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ['name', 'age', 'empty']

    def test_mixed_null_and_data_kept(self, tmp_path):
        """Columns with some null values but also data are kept"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'value': [100, 200, 300],
            'all_null': [None, None, None]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'name' in result_df.columns
        assert 'value' in result_df.columns
        assert 'all_null' not in result_df.columns
        assert len(result_df) == 3

    def test_write_cleaned_csv_file(self, tmp_path):
        """Write cleaned data back to CSV file"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'empty': [None, None, None],
            'name': ['A', 'B', 'C']
        })
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.drop_empty_columns(frame)
        ar.write_csv(result, str(output_file))
        
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert 'empty' not in output_df.columns
        assert len(output_df) == 3
        assert list(output_df.columns) == ['id', 'name']

    def test_cli_drop_empty_columns(self, tmp_path):
        """Test drop_empty_columns through CLI"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'empty': [None, None],
            'value': [100, 200]
        })
        df.to_csv(input_file, index=False)
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", 
             "drop-empty-columns", str(input_file), "-o", str(output_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        
        output_df = pd.read_csv(output_file)
        assert 'empty' not in output_df.columns
        assert 'name' in output_df.columns
        assert 'value' in output_df.columns

    def test_column_order_preserved(self, tmp_path):
        """Column order is preserved after dropping empty columns"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'z': [1, 2],
            'empty1': [None, None],
            'a': [3, 4],
            'empty2': [None, None],
            'b': [5, 6]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert list(result_df.columns) == ['z', 'a', 'b']

    def test_empty_strings_treated_as_empty(self, tmp_path):
        """Empty strings and null values in a column cause it to be dropped"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'empty_str_col'])
            writer.writerow(['Alice', ''])
            writer.writerow(['Bob', ''])
            writer.writerow(['Charlie', ''])
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'empty_str_col' not in result_df.columns
        assert 'name' in result_df.columns

    def test_data_integrity_preserved(self, tmp_path):
        """Data integrity is maintained in remaining columns"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'empty': [None, None, None],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert result_df['id'].tolist() == [1, 2, 3]
        assert result_df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
        np.testing.assert_array_almost_equal(
            result_df['value'].values, [10.5, 20.3, 30.1]
        )

    def test_mixed_empty_null_whitespace(self, tmp_path):
        """Column with mixed null, empty string, and whitespace is dropped"""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'mixed'])
            writer.writerow(['Alice', None])
            writer.writerow(['Bob', '   '])
            writer.writerow(['Charlie', ''])
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'mixed' not in result_df.columns
        assert len(result_df) == 3

    def test_single_empty_column_multiple_rows(self, tmp_path):
        """Empty column with many rows is correctly identified and dropped"""
        csv_file = tmp_path / "data.csv"
        data = {'data': list(range(100)), 'empty': [None] * 100}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        result_df = result.to_pandas()
        
        assert 'empty' not in result_df.columns
        assert 'data' in result_df.columns
        assert len(result_df) == 100
