"""
End-to-end pytest tests for arnio.frame.ArFrame.drop_columns and select_columns.

These tests drive complete user workflows through arnio's public API, using
real CSV files on disk, cleaning/validation, and observable outcomes.
"""

import pytest
import pandas as pd
import arnio as ar
import csv
from pathlib import Path


class TestDropColumnsBasic:
    """Test basic drop_columns workflow with CSV I/O."""

    def test_drop_columns_basic_workflow(self, tmp_path):
        """Load CSV, drop columns, verify result, write output, convert to pandas."""
        # Create test CSV
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email', 'age', 'salary'])
            writer.writerow(['1', 'Alice', 'alice@example.com', '30', '100000'])
            writer.writerow(['2', 'Bob', 'bob@example.com', '25', '95000'])

        # Workflow: load → drop columns → verify → write → check output
        frame = ar.read_csv(csv_path)
        assert frame.columns == ['id', 'name', 'email', 'age', 'salary']

        # Drop sensitive columns
        result = frame.drop_columns(['email', 'salary'])
        assert result.columns == ['id', 'name', 'age']

        # Verify row count preserved
        assert result.shape == (2, 3)

        # Convert to pandas and verify data integrity
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'age']
        assert df['name'].tolist() == ['Alice', 'Bob']

        # Write to output CSV and verify
        out_path = tmp_path / "cleaned.csv"
        ar.write_csv(result, out_path)
        output_df = pd.read_csv(out_path)
        assert list(output_df.columns) == ['id', 'name', 'age']
        assert len(output_df) == 2

    def test_drop_columns_preserves_order(self, tmp_path):
        """Dropping columns from middle preserves order of remaining columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['col_a', 'col_b', 'col_c', 'col_d', 'col_e'])
            writer.writerow(['1', '2', '3', '4', '5'])

        frame = ar.read_csv(csv_path)
        # Drop middle columns
        result = frame.drop_columns(['col_b', 'col_d'])

        assert result.columns == ['col_a', 'col_c', 'col_e']
        df = ar.to_pandas(result)
        assert df['col_a'].iloc[0] == '1'
        assert df['col_c'].iloc[0] == '3'
        assert df['col_e'].iloc[0] == '5'

    def test_drop_columns_with_duplicates_in_input(self, tmp_path):
        """Duplicate column names in drop list are silently deduplicated."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email', 'age'])
            writer.writerow(['1', 'Alice', 'alice@example.com', '30'])

        frame = ar.read_csv(csv_path)
        # Duplicates in drop list
        result = frame.drop_columns(['email', 'email', 'age', 'age'])

        assert result.columns == ['id', 'name']
        df = ar.to_pandas(result)
        assert len(df) == 1

    def test_drop_columns_empty_list_returns_copy(self, tmp_path):
        """Empty drop list returns unchanged copy of frame."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email'])
            writer.writerow(['1', 'Alice', 'alice@example.com'])

        frame = ar.read_csv(csv_path)
        original_cols = frame.columns[:]
        result = frame.drop_columns([])

        assert result.columns == original_cols
        assert result is not frame  # Verify it's a new frame

    def test_drop_columns_tuple_argument(self, tmp_path):
        """drop_columns accepts tuple in addition to list."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email'])
            writer.writerow(['1', 'Alice', 'alice@example.com'])

        frame = ar.read_csv(csv_path)

        # With tuple
        result = frame.drop_columns(('email',))
        assert result.columns == ['id', 'name']
        df = ar.to_pandas(result)
        assert list(df.columns) == ['id', 'name']


class TestSelectColumnsBasic:
    """Test basic select_columns workflow with CSV I/O."""

    def test_select_columns_basic_workflow(self, tmp_path):
        """Load CSV, select specific columns, verify result and output."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email', 'age', 'salary'])
            writer.writerow(['1', 'Alice', 'alice@example.com', '30', '100000'])
            writer.writerow(['2', 'Bob', 'bob@example.com', '25', '95000'])

        # Workflow: load → select columns → verify → write → check output
        frame = ar.read_csv(csv_path)
        result = frame.select_columns(['id', 'name', 'age'])

        assert result.columns == ['id', 'name', 'age']
        assert result.shape == (2, 3)

        # Convert to pandas
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df['age'].tolist() == ['30', '25']

        # Write and verify
        out_path = tmp_path / "selected.csv"
        ar.write_csv(result, out_path)
        output_df = pd.read_csv(out_path)
        assert list(output_df.columns) == ['id', 'name', 'age']

    def test_select_columns_custom_order(self, tmp_path):
        """select_columns preserves requested column order, not file order."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email', 'age'])
            writer.writerow(['1', 'Alice', 'alice@example.com', '30'])

        frame = ar.read_csv(csv_path)
        # Select in different order
        result = frame.select_columns(['age', 'name', 'id'])

        assert result.columns == ['age', 'name', 'id']
        df = ar.to_pandas(result)
        assert list(df.columns) == ['age', 'name', 'id']
        # Verify values are correct (not misaligned)
        assert df['age'].iloc[0] == '30'
        assert df['name'].iloc[0] == 'Alice'


class TestDropColumnsErrors:
    """Test drop_columns error handling and validation."""

    def test_drop_columns_error_nonexistent_column(self, tmp_path):
        """ValueError raised when dropping nonexistent column."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(ValueError) as exc_info:
            frame.drop_columns(['nonexistent'])
        assert "Unknown column" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_drop_columns_error_all_columns(self, tmp_path):
        """ValueError raised when attempting to drop all columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(ValueError) as exc_info:
            frame.drop_columns(['id', 'name'])
        assert "cannot remove all columns" in str(exc_info.value).lower()

    def test_drop_columns_error_wrong_type_string(self, tmp_path):
        """TypeError raised if cols argument is a string instead of list."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(TypeError) as exc_info:
            frame.drop_columns('id')
        assert "list or tuple" in str(exc_info.value)

    def test_drop_columns_error_wrong_type_dict(self, tmp_path):
        """TypeError raised if cols argument is not list/tuple."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(TypeError) as exc_info:
            frame.drop_columns({'id'})
        assert "list or tuple" in str(exc_info.value)

    def test_drop_columns_error_non_string_element(self, tmp_path):
        """TypeError raised if list contains non-string elements."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email'])
            writer.writerow(['1', 'Alice', 'alice@example.com'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(TypeError) as exc_info:
            frame.drop_columns(['id', 123])
        assert "must be strings" in str(exc_info.value)


class TestSelectColumnsErrors:
    """Test select_columns error handling and validation."""

    def test_select_columns_error_empty_selection(self, tmp_path):
        """ValueError raised if selection is empty."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(ValueError) as exc_info:
            frame.select_columns([])
        assert "empty" in str(exc_info.value).lower()

    def test_select_columns_error_duplicates(self, tmp_path):
        """ValueError raised if selection contains duplicate column names."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email'])
            writer.writerow(['1', 'Alice', 'alice@example.com'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(ValueError) as exc_info:
            frame.select_columns(['id', 'name', 'id'])
        assert "duplicate" in str(exc_info.value).lower()

    def test_select_columns_error_unknown_column(self, tmp_path):
        """ValueError raised if selection includes unknown columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(ValueError) as exc_info:
            frame.select_columns(['id', 'nonexistent'])
        assert "Unknown" in str(exc_info.value) or "nonexistent" in str(exc_info.value)

    def test_select_columns_error_string_argument(self, tmp_path):
        """TypeError raised if columns argument is a string."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            writer.writerow(['1', 'Alice'])

        frame = ar.read_csv(csv_path)

        with pytest.raises(TypeError) as exc_info:
            frame.select_columns('id')
        assert "sequence" in str(exc_info.value)


class TestIntegration:
    """Integration tests combining drop_columns and select_columns with pipelines."""

    def test_pipeline_with_drop_and_cleanup(self, tmp_path):
        """Integration: drop_columns combined with cleaning pipeline."""
        csv_path = tmp_path / "messy.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'debug_col', 'email'])
            writer.writerow(['1', '  Alice  ', 'temp_data', '  alice@example.com  '])
            writer.writerow(['2', '  Bob  ', 'temp_data', '  bob@example.com  '])

        # Workflow: clean and drop columns
        frame = ar.read_csv(csv_path)
        
        # Clean first, then drop columns
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        result = clean.drop_columns(['debug_col', 'email'])

        assert result.columns == ['id', 'name']
        df = ar.to_pandas(result)
        assert df['name'].iloc[0] == 'Alice'
        assert len(df) == 2

        # Write final result
        out_path = tmp_path / "final.csv"
        ar.write_csv(result, out_path)
        final_df = pd.read_csv(out_path)
        assert list(final_df.columns) == ['id', 'name']

    def test_drop_then_select_composition(self, tmp_path):
        """Compose drop_columns and select_columns operations."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['a', 'b', 'c', 'd', 'e', 'f'])
            writer.writerow(['1', '2', '3', '4', '5', '6'])

        frame = ar.read_csv(csv_path)

        # Drop some columns
        after_drop = frame.drop_columns(['c', 'e'])
        assert after_drop.columns == ['a', 'b', 'd', 'f']

        # Select from result
        final = after_drop.select_columns(['d', 'b'])
        assert final.columns == ['d', 'b']

        df = ar.to_pandas(final)
        assert df['d'].iloc[0] == '4'
        assert df['b'].iloc[0] == '2'

    def test_profiling_after_column_selection(self, tmp_path):
        """Profile data quality after selecting columns."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'email', 'age'])
            writer.writerow(['1', 'Alice', 'alice@example.com', '30'])
            writer.writerow(['2', 'Bob', 'bob@example.com', ''])  # missing age

        frame = ar.read_csv(csv_path)
        result = frame.select_columns(['id', 'name', 'age'])

        # Generate profile report
        report = ar.profile(result)
        summary = report.summary()

        assert summary['row_count'] == 2
        assert summary['column_count'] == 3
        # age column has a null
        age_profile = report.columns['age']
        assert age_profile['null_count'] > 0
