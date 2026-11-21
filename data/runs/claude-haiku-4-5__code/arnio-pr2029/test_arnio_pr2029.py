import pytest
import pandas as pd
import arnio as ar


class TestCleanColumnNames:
    """Tests for arnio.cleaning.clean_column_names function."""

    def test_basic_cleaning_special_characters(self):
        """Test cleaning of special characters in column names."""
        df = pd.DataFrame({
            'Column-Name': [1, 2, 3],
            'Another@Column': [4, 5, 6],
            'Third$Column': [7, 8, 9]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['column_name', 'another_column', 'third_column']
        assert list(result.columns) == expected_cols

    def test_consecutive_underscores_collapsed(self):
        """Test that consecutive underscores are collapsed into one."""
        df = pd.DataFrame({
            'Column___Name': [1, 2],
            'Another__Column': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['column_name', 'another_column']
        assert list(result.columns) == expected_cols

    def test_leading_trailing_underscores_trimmed(self):
        """Test that leading and trailing underscores are trimmed."""
        df = pd.DataFrame({
            '_column_name_': [1, 2],
            '__another__': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['column_name', 'another']
        assert list(result.columns) == expected_cols

    def test_case_type_lower_default(self):
        """Test case_type='lower' (default)."""
        df = pd.DataFrame({
            'ColumnName': [1, 2],
            'ANOTHER_COLUMN': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['columnname', 'another_column']
        assert list(result.columns) == expected_cols

    def test_case_type_upper(self):
        """Test case_type='upper'."""
        df = pd.DataFrame({
            'column_name': [1, 2],
            'another_column': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame, case_type='upper')
        
        expected_cols = ['COLUMN_NAME', 'ANOTHER_COLUMN']
        assert list(result.columns) == expected_cols

    def test_case_type_title(self):
        """Test case_type='title'."""
        df = pd.DataFrame({
            'column_name': [1, 2],
            'another_column': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame, case_type='title')
        
        expected_cols = ['Column_Name', 'Another_Column']
        assert list(result.columns) == expected_cols

    def test_case_type_camel(self):
        """Test case_type='camel'."""
        df = pd.DataFrame({
            'column_name': [1, 2],
            'another_column': [3, 4],
            'simple': [5, 6]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame, case_type='camel')
        
        expected_cols = ['columnName', 'anotherColumn', 'simple']
        assert list(result.columns) == expected_cols

    def test_case_type_none(self):
        """Test case_type='none' preserves case after cleaning."""
        df = pd.DataFrame({
            'Column-Name': [1, 2],
            'ANOTHER_Column': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame, case_type='none')
        
        expected_cols = ['Column_Name', 'ANOTHER_Column']
        assert list(result.columns) == expected_cols

    def test_empty_column_name_becomes_column(self):
        """Test that columns that become empty after cleaning become 'column'."""
        df = pd.DataFrame({
            '---': [1, 2],
            '___': [3, 4],
            'valid_name': [5, 6]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['column', 'column', 'valid_name']
        assert list(result.columns) == expected_cols

    def test_duplicate_column_names_raises_error(self):
        """Test that cleaning resulting in duplicate names raises ValueError."""
        df = pd.DataFrame({
            'column_name': [1, 2],
            'Column Name': [3, 4]
        })
        frame = ar.ArFrame(df)
        
        with pytest.raises(ValueError, match="Cleaning column names would create duplicates"):
            ar.cleaning.clean_column_names(frame)

    def test_invalid_case_type_raises_error(self):
        """Test that invalid case_type raises ValueError."""
        df = pd.DataFrame({'column': [1, 2]})
        frame = ar.ArFrame(df)
        
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.cleaning.clean_column_names(frame, case_type='invalid')

    def test_non_string_case_type_raises_error(self):
        """Test that non-string case_type raises TypeError."""
        df = pd.DataFrame({'column': [1, 2]})
        frame = ar.ArFrame(df)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.cleaning.clean_column_names(frame, case_type=123)

    def test_mixed_special_characters_and_numbers(self):
        """Test cleaning with mixed special characters and numbers."""
        df = pd.DataFrame({
            'Col123-Name@456!': [1, 2],
            'Data#2024$Value': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['col123_name_456', 'data_2024_value']
        assert list(result.columns) == expected_cols

    def test_spaces_converted_to_underscores(self):
        """Test that spaces are converted to underscores."""
        df = pd.DataFrame({
            'Column Name': [1, 2],
            'Another   Column': [3, 4]
        })
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        expected_cols = ['column_name', 'another_column']
        assert list(result.columns) == expected_cols

    def test_returns_arframe(self):
        """Test that the function returns an ArFrame instance."""
        df = pd.DataFrame({'Column': [1, 2]})
        frame = ar.ArFrame(df)
        result = ar.cleaning.clean_column_names(frame)
        
        assert isinstance(result, ar.ArFrame)
