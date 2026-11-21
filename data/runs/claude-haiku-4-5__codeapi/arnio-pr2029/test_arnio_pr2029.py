import pytest
import pandas as pd
import arnio as ar


class TestCleanColumnNames:
    """Tests for arnio.cleaning.clean_column_names function."""

    def test_basic_lowercase_normalization(self):
        """Test basic cleaning with default lowercase case_type."""
        df = pd.DataFrame({'ColName': [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'colname' in result.columns
        assert result.shape == (2, 1)

    def test_case_type_upper(self):
        """Test case_type='upper' converts to uppercase."""
        df = pd.DataFrame({'col_name': [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type='upper')
        assert 'COL_NAME' in result.columns

    def test_case_type_title(self):
        """Test case_type='title' titlecases each underscore-separated part."""
        df = pd.DataFrame({'my_col_name': [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type='title')
        assert 'My_Col_Name' in result.columns

    def test_case_type_camel(self):
        """Test case_type='camel' creates camelCase format."""
        df = pd.DataFrame({'my_col_name': [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type='camel')
        assert 'myColName' in result.columns

    def test_case_type_none_preserves_case(self):
        """Test case_type='none' preserves existing case after normalization."""
        df = pd.DataFrame({'MyColName': [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type='none')
        assert 'MyColName' in result.columns

    def test_special_characters_replaced_with_underscores(self):
        """Test that special characters are replaced with underscores."""
        df = pd.DataFrame({'col-name': [1], 'col.name': [2], 'col@name': [3]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert result.shape[1] == 3
        assert all('col_name' in col for col in result.columns)

    def test_spaces_converted_to_underscores(self):
        """Test that spaces are converted to underscores."""
        df = pd.DataFrame({'col name': [1], 'another col': [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'col_name' in result.columns
        assert 'another_col' in result.columns

    def test_consecutive_underscores_collapsed(self):
        """Test that consecutive underscores are collapsed to single underscore."""
        df = pd.DataFrame({'col__name': [1], 'col___test': [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'col_name' in result.columns
        assert 'col_test' in result.columns

    def test_leading_and_trailing_underscores_trimmed(self):
        """Test that leading and trailing underscores are removed."""
        df = pd.DataFrame({'_col_name_': [1], '__name__': [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'col_name' in result.columns
        assert 'name' in result.columns

    def test_numeric_characters_preserved(self):
        """Test that numeric characters in column names are preserved."""
        df = pd.DataFrame({'col_1_name': [1], 'col2name': [2], 'name_3': [3]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'col_1_name' in result.columns
        assert 'col2name' in result.columns
        assert 'name_3' in result.columns

    def test_empty_column_becomes_column_keyword(self):
        """Test that column names becoming empty after cleaning become 'column'."""
        df = pd.DataFrame({'___': [1]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert 'column' in result.columns

    def test_duplicate_column_names_raises_error(self):
        """Test that cleaning creating duplicate column names raises ValueError."""
        df = pd.DataFrame({'col-name': [1], 'col_name': [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="duplicates"):
            ar.clean_column_names(frame)

    def test_invalid_case_type_raises_value_error(self):
        """Test that invalid case_type value raises ValueError."""
        df = pd.DataFrame({'col': [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type='invalid')

    def test_non_string_case_type_raises_type_error(self):
        """Test that non-string case_type raises TypeError."""
        df = pd.DataFrame({'col': [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=123)

    def test_data_preservation_during_cleaning(self):
        """Test that underlying data is preserved when cleaning column names."""
        df = pd.DataFrame({'Col-Name': [1, 2, 3], 'Another.Col': [4.5, 5.5, 6.5]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert result.shape[0] == 3
        assert result.shape[1] == 2
