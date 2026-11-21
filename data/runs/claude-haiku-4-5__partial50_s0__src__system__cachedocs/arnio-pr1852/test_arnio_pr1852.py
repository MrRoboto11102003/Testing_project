import pytest
import pandas as pd
import numpy as np
import arnio as ar
import csv
from pathlib import Path


def test_coalesce_columns_basic(tmp_path):
    """Test coalesce_columns: select first non-null value from multiple columns."""
    csv_path = tmp_path / "coalesce.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'phone', 'mobile', 'email'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'phone': '555-1234', 'mobile': '', 'email': 'alice@example.com'},
            {'id': '2', 'phone': '', 'mobile': '555-5678', 'email': 'bob@example.com'},
            {'id': '3', 'phone': '', 'mobile': '', 'email': 'charlie@example.com'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.coalesce_columns(frame, subset=['phone', 'mobile', 'email'], output_column='contact')
    
    assert 'contact' in result.columns
    assert result.shape[0] == 3
    
    df = ar.to_pandas(result)
    assert df.loc[0, 'contact'] == '555-1234'
    assert df.loc[1, 'contact'] == '555-5678'
    assert pd.isna(df.loc[2, 'contact'])


def test_combine_columns_concatenate_values(tmp_path):
    """Test combine_columns: concatenate multiple columns with separator."""
    csv_path = tmp_path / "combine.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['first_name', 'last_name', 'city'])
        writer.writeheader()
        writer.writerows([
            {'first_name': 'Alice', 'last_name': 'Smith', 'city': 'NYC'},
            {'first_name': 'Bob', 'last_name': 'Jones', 'city': 'LA'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.combine_columns(frame, subset=['first_name', 'last_name'], 
                                separator=' ', output_column='full_name')
    
    assert 'full_name' in result.columns
    df = ar.to_pandas(result)
    assert df.loc[0, 'full_name'] == 'Alice Smith'
    assert df.loc[1, 'full_name'] == 'Bob Jones'
    
    # Write result back to CSV
    output_path = tmp_path / "combined.csv"
    ar.write_csv(result, str(output_path))
    assert output_path.exists()


def test_drop_columns_matching_by_pattern(tmp_path):
    """Test drop_columns_matching: remove columns by regex pattern."""
    csv_path = tmp_path / "drop_match.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'temp_col', 'temp_data', 'value'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'name': 'Alice', 'temp_col': 'x', 'temp_data': 'y', 'value': '100'},
            {'id': '2', 'name': 'Bob', 'temp_col': 'a', 'temp_data': 'b', 'value': '200'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.drop_columns_matching(frame, pattern='^temp_')
    
    assert result.shape[1] == 3
    assert 'temp_col' not in result.columns
    assert 'temp_data' not in result.columns
    assert 'id' in result.columns
    assert 'name' in result.columns
    
    df = ar.to_pandas(result)
    assert list(df.columns) == ['id', 'name', 'value']


def test_drop_constant_columns_remove_single_value_cols(tmp_path):
    """Test drop_constant_columns: remove columns with single unique value."""
    csv_path = tmp_path / "constant.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'region', 'status', 'value'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'region': 'US', 'status': 'active', 'value': '100'},
            {'id': '2', 'region': 'US', 'status': 'active', 'value': '200'},
            {'id': '3', 'region': 'US', 'status': 'active', 'value': '150'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.drop_constant_columns(frame)
    
    assert 'region' not in result.columns
    assert 'status' not in result.columns
    assert 'id' in result.columns
    assert 'value' in result.columns
    assert result.shape[1] == 2


def test_drop_nulls_removes_incomplete_rows(tmp_path):
    """Test drop_nulls: remove rows with null/empty values."""
    csv_path = tmp_path / "nulls.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'email'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'name': 'Alice', 'email': 'alice@example.com'},
            {'id': '2', 'name': '', 'email': 'bob@example.com'},
            {'id': '3', 'name': 'Charlie', 'email': ''},
            {'id': '4', 'name': 'David', 'email': 'david@example.com'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.drop_nulls(frame)
    
    assert result.shape[0] == 2
    df = ar.to_pandas(result)
    assert df['id'].tolist() == ['1', '4']
    
    # Test with subset
    result_subset = ar.drop_nulls(frame, subset=['name'])
    assert result_subset.shape[0] == 3


def test_filter_rows_comparison_operators(tmp_path):
    """Test filter_rows: filter rows by comparison operator."""
    csv_path = tmp_path / "filter.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'age', 'name'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'age': '25', 'name': 'Alice'},
            {'id': '2', 'age': '17', 'name': 'Bob'},
            {'id': '3', 'age': '30', 'name': 'Charlie'},
            {'id': '4', 'age': '16', 'name': 'David'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    
    # Test >= operator
    result_gte = ar.filter_rows(frame, column='age', op='>=', value=18)
    assert result_gte.shape[0] == 2
    
    # Test < operator
    result_lt = ar.filter_rows(frame, column='age', op='<', value=25)
    assert result_lt.shape[0] == 2
    
    # Test == operator
    result_eq = ar.filter_rows(frame, column='name', op='==', value='Alice')
    assert result_eq.shape[0] == 1


def test_keep_rows_with_nulls_isolates_incomplete(tmp_path):
    """Test keep_rows_with_nulls: keep only rows with at least one null."""
    csv_path = tmp_path / "incomplete.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'email'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'name': 'Alice', 'email': 'alice@example.com'},
            {'id': '2', 'name': '', 'email': 'bob@example.com'},
            {'id': '3', 'name': 'Charlie', 'email': ''},
            {'id': '4', 'name': 'David', 'email': 'david@example.com'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.keep_rows_with_nulls(frame)
    
    assert result.shape[0] == 2
    df = ar.to_pandas(result)
    assert df['id'].tolist() == ['2', '3']
    
    # Test with subset
    result_subset = ar.keep_rows_with_nulls(frame, subset=['email'])
    assert result_subset.shape[0] == 1


def test_replace_values_mapping_substitution(tmp_path):
    """Test replace_values: replace values using a mapping dictionary."""
    csv_path = tmp_path / "replace.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'status'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'status': 'active'},
            {'id': '2', 'status': 'inactive'},
            {'id': '3', 'status': 'active'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.replace_values(frame, {'active': 'A', 'inactive': 'I'}, column='status')
    
    df = ar.to_pandas(result)
    assert df.loc[0, 'status'] == 'A'
    assert df.loc[1, 'status'] == 'I'
    assert df.loc[2, 'status'] == 'A'


def test_safe_divide_columns_zero_handling(tmp_path):
    """Test safe_divide_columns: divide columns with zero/null denominator handling."""
    csv_path = tmp_path / "divide.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'revenue', 'cost'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'revenue': '1000', 'cost': '500'},
            {'id': '2', 'revenue': '2000', 'cost': '0'},
            {'id': '3', 'revenue': '500', 'cost': '250'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.safe_divide_columns(frame, numerator='revenue', denominator='cost', 
                                     output_column='ratio', fill_value=0.0)
    
    assert 'ratio' in result.columns
    df = ar.to_pandas(result)
    assert df.loc[0, 'ratio'] == 2.0
    assert df.loc[1, 'ratio'] == 0.0  # zero denominator replaced with fill_value
    assert df.loc[2, 'ratio'] == 2.0


def test_standardize_missing_tokens_normalize_na(tmp_path):
    """Test standardize_missing_tokens: normalize missing value strings to NaN."""
    csv_path = tmp_path / "missing.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'value'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'value': '100'},
            {'id': '2', 'value': 'N/A'},
            {'id': '3', 'value': 'NA'},
            {'id': '4', 'value': 'null'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    result = ar.standardize_missing_tokens(frame)
    
    df = ar.to_pandas(result)
    assert pd.notna(df.loc[0, 'value'])
    assert pd.isna(df.loc[1, 'value'])
    assert pd.isna(df.loc[2, 'value'])
    assert pd.isna(df.loc[3, 'value'])


def test_pandas_accessor_workflow(tmp_path):
    """Test using the pandas .arnio accessor for integrated workflow."""
    csv_path = tmp_path / "accessor.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'value'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'name': '  Alice  ', 'value': '100'},
            {'id': '2', 'name': '  Bob  ', 'value': '200'},
            {'id': '1', 'name': '  Alice  ', 'value': '100'},
        ])
    
    # Load as pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Use arnio accessor for cleaning
    result_df = df.arnio.clean(strip_whitespace=True, drop_duplicates=True)
    
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == 2
    assert result_df.loc[0, 'name'] == 'Alice'
    
    # Profile for quality assessment
    quality = result_df.arnio.profile()
    assert hasattr(quality, 'summary')
    summary = quality.summary()
    assert summary['row_count'] == 2


def test_pipeline_multiple_steps_combined(tmp_path):
    """Test applying multiple cleaning steps in a single pipeline."""
    csv_path = tmp_path / "pipeline.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'first', 'last', 'status', 'debug'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'first': 'Alice', 'last': 'Smith', 'status': 'active', 'debug': 'x'},
            {'id': '2', 'first': 'Bob', 'last': 'Jones', 'status': 'inactive', 'debug': 'x'},
            {'id': '1', 'first': 'Alice', 'last': 'Smith', 'status': 'active', 'debug': 'x'},
        ])
    
    frame = ar.read_csv(str(csv_path))
    
    # Apply multi-step pipeline
    clean = ar.pipeline(frame, [
        ('combine_columns', {'subset': ['first', 'last'], 'output_column': 'full_name', 'separator': ' '}),
        ('drop_columns', {'columns': ['debug']}),
        ('drop_duplicates', {'keep': 'first'}),
    ])
    
    df = ar.to_pandas(clean)
    assert 'full_name' in df.columns
    assert 'debug' not in df.columns
    assert df.shape[0] == 2
    assert df.loc[0, 'full_name'] == 'Alice Smith'


def test_error_handling_invalid_inputs():
    """Test error handling for invalid column names and types."""
    df_pd = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    frame = ar.from_pandas(df_pd)
    
    # Invalid column should raise KeyError
    with pytest.raises(KeyError):
        ar.combine_columns(frame, subset=['nonexistent_col'])
    
    # Invalid separator type should raise TypeError
    with pytest.raises(TypeError):
        ar.combine_columns(frame, subset=['a'], separator=123)
    
    # Empty output_column should raise ValueError
    with pytest.raises(ValueError):
        ar.combine_columns(frame, subset=['a'], output_column='')
    
    # Invalid filter operator should raise ValueError
    with pytest.raises(ValueError):
        ar.filter_rows(frame, column='a', op='<<<', value=5)


def test_csv_roundtrip_read_process_write(tmp_path):
    """Test complete workflow: read CSV, apply cleaning, write CSV."""
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    
    # Create input file with multiple issues
    with open(input_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'value', 'status', 'constant'])
        writer.writeheader()
        writer.writerows([
            {'id': '1', 'value': '100', 'status': 'yes', 'constant': 'X'},
            {'id': '2', 'value': '50', 'status': 'no', 'constant': 'X'},
            {'id': '3', 'value': '200', 'status': 'yes', 'constant': 'X'},
        ])
    
    # Read, process with multiple steps, write
    frame = ar.read_csv(str(input_csv))
    
    # Apply cleaning pipeline
    clean = ar.pipeline(frame, [
        ('replace_values', {'mapping': {'yes': '1', 'no': '0'}, 'column': 'status'}),
        ('drop_constant_columns',),
        ('filter_rows', {'column': 'value', 'op': '>=', 'value': 100}),
    ])
    
    ar.write_csv(clean, str(output_csv))
    
    # Verify output file exists and has correct content
    assert output_csv.exists()
    result_df = pd.read_csv(output_csv)
    assert result_df.shape[0] == 2
    assert 'constant' not in result_df.columns
    assert all(int(v) >= 100 for v in result_df['value'])


def test_complex_real_world_data_cleaning(tmp_path):
    """Test realistic data cleaning scenario combining all primitives."""
    csv_path = tmp_path / "messy_data.csv"
    output_path = tmp_path / "clean_data.csv"
    
    # Create realistic messy dataset
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'first_name', 'last_name', 'email', 'phone', 'revenue', 'cost', 'region', 'debug_col'])
        writer.writeheader()
        writer.writerows([
            {'user_id': '1', 'first_name': 'Alice', 'last_name': 'Smith', 'email': 'alice@example.com', 'phone': '555-1234', 'revenue': '5000', 'cost': '2000', 'region': 'US', 'debug_col': 'tmp'},
            {'user_id': '2', 'first_name': '', 'last_name': 'Jones', 'email': '', 'phone': '555-5678', 'revenue': '3000', 'cost': '1500', 'region': 'US', 'debug_col': 'tmp'},
            {'user_id': '3', 'first_name': 'Charlie', 'last_name': 'Brown', 'email': 'charlie@example.com', 'phone': '', 'revenue': '4000', 'cost': '2000', 'region': 'US', 'debug_col': 'tmp'},
            {'user_id': '1', 'first_name': 'Alice', 'last_name': 'Smith', 'email': 'alice@example.com', 'phone': '555-1234', 'revenue': '5000', 'cost': '2000', 'region': 'US', 'debug_col': 'tmp'},
        ])
    
    # Load and clean with comprehensive pipeline
    frame = ar.read_csv(str(csv_path))
    
    clean = ar.pipeline(frame, [
        ('combine_columns', {'subset': ['first_name', 'last_name'], 'output_column': 'full_name', 'separator': ' '}),
        ('coalesce_columns', {'subset': ['email', 'phone'], 'output_column': 'contact_info'}),
        ('drop_columns', {'columns': ['debug_col']}),
        ('drop_nulls', {'subset': ['first_name']}),
        ('safe_divide_columns', {'numerator': 'revenue', 'denominator': 'cost', 'output_column': 'margin_ratio'}),
        ('drop_duplicates', {'keep': 'first'}),
    ])
    
    # Write cleaned output
    ar.write_csv(clean, str(output_path))
    
    # Verify results
    assert output_path.exists()
    result_df = pd.read_csv(output_path)
    
    # Verify expected outcomes
    assert result_df.shape[0] == 2  # duplicates and nulls removed
    assert 'full_name' in result_df.columns
    assert 'contact_info' in result_df.columns
    assert 'margin_ratio' in result_df.columns
    assert 'debug_col' not in result_df.columns
    assert all(pd.notna(result_df['first_name']))
    assert result_df.loc[0, 'full_name'] == 'Alice Smith'
    assert result_df.loc[1, 'full_name'] == 'Charlie Brown'
