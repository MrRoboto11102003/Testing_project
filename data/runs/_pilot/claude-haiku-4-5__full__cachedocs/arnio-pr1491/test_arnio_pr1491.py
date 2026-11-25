import pytest
import arnio as ar
import pandas as pd
import re


def test_drop_columns_matching_basic_prefix():
    """Test dropping columns with prefix pattern"""
    df = pd.DataFrame({
        'temp_col1': [1, 2, 3],
        'temp_col2': [4, 5, 6],
        'data_col': [7, 8, 9]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['data_col']


def test_drop_columns_matching_suffix():
    """Test dropping columns with suffix pattern"""
    df = pd.DataFrame({
        'col_temp': [1, 2],
        'data_col_temp': [3, 4],
        'keep_col': [5, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"_temp$")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['keep_col']


def test_drop_columns_matching_multiple():
    """Test dropping multiple columns matching pattern"""
    df = pd.DataFrame({
        'debug_1': [1, 2],
        'debug_2': [3, 4],
        'debug_3': [5, 6],
        'main': [7, 8]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^debug_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['main']


def test_drop_columns_matching_no_matches():
    """Test when pattern matches no columns"""
    df = pd.DataFrame({
        'col1': [1, 2],
        'col2': [3, 4],
        'col3': [5, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^nomatch_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['col1', 'col2', 'col3']


def test_drop_columns_matching_all_columns_raises_error():
    """Test that matching all columns raises ValueError"""
    df = pd.DataFrame({
        'col1': [1, 2],
        'col2': [3, 4]
    })
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError):
        ar.drop_columns_matching(frame, r".*")


def test_drop_columns_matching_invalid_regex_raises_error():
    """Test that invalid regex pattern raises re.error"""
    df = pd.DataFrame({
        'col1': [1, 2],
        'col2': [3, 4]
    })
    frame = ar.from_pandas(df)
    with pytest.raises(re.error):
        ar.drop_columns_matching(frame, r"[invalid(")


def test_drop_columns_matching_non_string_pattern_raises_error():
    """Test that non-string pattern raises TypeError"""
    df = pd.DataFrame({
        'col1': [1, 2],
        'col2': [3, 4]
    })
    frame = ar.from_pandas(df)
    with pytest.raises(TypeError):
        ar.drop_columns_matching(frame, 123)


def test_drop_columns_matching_preserves_data():
    """Test that remaining columns preserve their data integrity"""
    df = pd.DataFrame({
        'temp_col': [1, 2, 3],
        'keep_col': [10, 20, 30]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert result_df['keep_col'].tolist() == [10, 20, 30]


def test_drop_columns_matching_preserves_column_order():
    """Test that column order is preserved for remaining columns"""
    df = pd.DataFrame({
        'a': [1],
        'temp_b': [2],
        'c': [3],
        'd': [4]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['a', 'c', 'd']


def test_drop_columns_matching_case_sensitive():
    """Test that pattern matching is case-sensitive by default"""
    df = pd.DataFrame({
        'Temp_col': [1, 2],
        'temp_col': [3, 4],
        'keep_col': [5, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert set(result_df.columns) == {'Temp_col', 'keep_col'}


def test_drop_columns_matching_complex_pattern():
    """Test with complex regex pattern matching digits"""
    df = pd.DataFrame({
        'col_001': [1, 2],
        'col_002': [3, 4],
        'col_100': [5, 6],
        'data_col': [7, 8]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"col_0\d{2}")
    result_df = ar.to_pandas(result)
    assert set(result_df.columns) == {'col_100', 'data_col'}


def test_drop_columns_matching_or_pattern():
    """Test with OR pattern to match multiple naming conventions"""
    df = pd.DataFrame({
        'temp_col': [1, 2],
        'debug_col': [3, 4],
        'keep_col': [5, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^(temp_|debug_)")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['keep_col']


def test_drop_columns_matching_works_with_pandas_df():
    """Test that function works with pandas DataFrame input"""
    df = pd.DataFrame({
        'temp_col': [1, 2],
        'keep_col': [3, 4]
    })
    result = ar.drop_columns_matching(df, r"^temp_")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['keep_col']


def test_drop_columns_matching_single_column_no_match():
    """Test with single column DataFrame that doesn't match"""
    df = pd.DataFrame({
        'data_col': [1, 2, 3]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ['data_col']


def test_drop_columns_matching_maintains_dtypes():
    """Test that remaining columns maintain their original dtypes"""
    df = pd.DataFrame({
        'temp_int': [1, 2, 3],
        'keep_float': [1.1, 2.2, 3.3],
        'keep_str': ['a', 'b', 'c']
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, r"^temp_")
    result_df = ar.to_pandas(result)
    assert result_df['keep_float'].dtype == 'float64'
    assert result_df['keep_str'].dtype == 'object'
