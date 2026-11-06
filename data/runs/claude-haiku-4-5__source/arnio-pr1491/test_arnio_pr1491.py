import pytest
import pandas as pd
import re
import arnio as ar
from arnio.cleaning import drop_columns_matching


class TestDropColumnsMatching:
    """Tests for arnio.cleaning.drop_columns_matching function."""

    def test_drop_columns_matching_prefix_pattern(self):
        """Test dropping columns with a prefix pattern."""
        df = pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'keep_col': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'^temp_')
        assert list(result.columns) == ['keep_col']
        assert len(result) == 3

    def test_drop_columns_matching_suffix_pattern(self):
        """Test dropping columns with a suffix pattern."""
        df = pd.DataFrame({
            'col_temp': [1, 2, 3],
            'data_temp': [4, 5, 6],
            'value': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'_temp$')
        assert list(result.columns) == ['value']

    def test_drop_columns_matching_no_matches(self):
        """Test when no columns match the pattern."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'^temp_')
        pd.testing.assert_frame_equal(df, result)

    def test_drop_columns_matching_multiple_columns(self):
        """Test dropping multiple columns matching the pattern."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'test_a': [4, 5, 6],
            'test_b': [7, 8, 9],
            'test_c': [10, 11, 12],
            'name': ['x', 'y', 'z']
        })
        result = drop_columns_matching(df, r'^test_')
        assert list(result.columns) == ['id', 'name']

    def test_drop_columns_matching_complex_regex_pattern(self):
        """Test with complex regex pattern using digit matching."""
        df = pd.DataFrame({
            'col_123': [1, 2, 3],
            'col_456': [4, 5, 6],
            'other': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'col_\d+')
        assert list(result.columns) == ['other']

    def test_drop_columns_matching_case_sensitive(self):
        """Test that pattern matching is case-sensitive."""
        df = pd.DataFrame({
            'Temp_col': [1, 2, 3],
            'temp_col': [4, 5, 6],
            'TEMP_col': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'^temp_')
        assert list(result.columns) == ['Temp_col', 'TEMP_col']

    def test_drop_columns_matching_preserves_data_integrity(self):
        """Test that data in remaining columns is preserved correctly."""
        df = pd.DataFrame({
            'temp_col': [1, 2, 3],
            'keep_col': [10, 20, 30]
        })
        result = drop_columns_matching(df, r'^temp_')
        pd.testing.assert_series_equal(result['keep_col'], df['keep_col'], check_names=True)

    def test_drop_columns_matching_pattern_not_string_raises_typeerror(self):
        """Test that non-string pattern raises TypeError."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(TypeError, match="pattern must be a string"):
            drop_columns_matching(df, 123)

    def test_drop_columns_matching_pattern_none_raises_typeerror(self):
        """Test that None pattern raises TypeError."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(TypeError, match="pattern must be a string"):
            drop_columns_matching(df, None)

    def test_drop_columns_matching_invalid_regex_raises_error(self):
        """Test that invalid regex pattern raises re.error."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(re.error):
            drop_columns_matching(df, r'[invalid(')

    def test_drop_columns_matching_all_columns_raises_valueerror(self):
        """Test that matching all columns raises ValueError."""
        df = pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6]
        })
        with pytest.raises(ValueError, match="Pattern matches all columns"):
            drop_columns_matching(df, r'.*')

    def test_drop_columns_matching_with_special_characters(self):
        """Test dropping columns with special characters in names."""
        df = pd.DataFrame({
            'col-1': [1, 2, 3],
            'col-2': [4, 5, 6],
            'col_3': [7, 8, 9]
        })
        result = drop_columns_matching(df, r'col-\d')
        assert list(result.columns) == ['col_3']

    def test_drop_columns_matching_with_arframe(self):
        """Test function works with ArFrame input."""
        data = {'keep': [1, 2, 3], 'temp_col': [4, 5, 6]}
        frame = ar.from_dict(data)
        result = drop_columns_matching(frame, r'^temp_')
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ['keep']

    def test_drop_columns_matching_alternation_pattern(self):
        """Test pattern with alternation matching multiple column name variations."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': [4, 5, 6],
            'age': [7, 8, 9],
            'email': [10, 11, 12]
        })
        result = drop_columns_matching(df, r'(name|email)')
        assert list(result.columns) == ['id', 'age']

    def test_drop_columns_matching_return_type_pandas(self):
        """Test that pandas DataFrame input returns pandas DataFrame."""
        df = pd.DataFrame({'temp_col': [1, 2], 'keep': [3, 4]})
        result = drop_columns_matching(df, r'^temp_')
        assert isinstance(result, pd.DataFrame)

    def test_drop_columns_matching_return_type_arframe(self):
        """Test that ArFrame input returns ArFrame."""
        data = {'keep': [1, 2], 'temp_col': [3, 4]}
        frame = ar.from_dict(data)
        result = drop_columns_matching(frame, r'^temp_')
        result_df = ar.to_pandas(result)
        assert isinstance(result_df, pd.DataFrame)
