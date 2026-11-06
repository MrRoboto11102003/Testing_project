"""
Pytest tests for arnio.cleaning.clean_column_names function.

Tests cover documented behavior including:
- Non-alphanumeric character replacement
- Consecutive underscore collapsing
- Leading/trailing underscore trimming
- Case normalization (lower, upper, title, camel, none)
- Error handling for invalid inputs
- Edge cases and special characters
"""

import pytest
import pandas as pd
import arnio as ar


class TestCleanColumnNames:
    """Test suite for clean_column_names function."""

    def test_basic_cleaning_default_lowercase(self):
        """Test basic column name cleaning with default (lower) case."""
        df = pd.DataFrame({
            "First Name": [1, 2],
            "Last-Name": [3, 4],
            "Age!": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        assert set(result_df.columns) == {"first_name", "last_name", "age"}

    def test_case_type_upper(self):
        """Test cleaning with uppercase case type."""
        df = pd.DataFrame({"hello world": [1, 2], "test_case": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        result_df = ar.to_pandas(result)

        assert "HELLO_WORLD" in result_df.columns
        assert "TEST_CASE" in result_df.columns

    def test_case_type_title(self):
        """Test cleaning with title case type."""
        df = pd.DataFrame({"hello_world": [1, 2], "test value": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        result_df = ar.to_pandas(result)

        assert len(result_df.columns) == 2
        assert all(col[0].isupper() if col else True for col in result_df.columns)

    def test_case_type_camel(self):
        """Test cleaning with camelCase type."""
        df = pd.DataFrame({"hello_world": [1, 2], "test value": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        result_df = ar.to_pandas(result)

        assert len(result_df.columns) == 2

    def test_case_type_none(self):
        """Test cleaning without case normalization."""
        df = pd.DataFrame({"Hello World": [1, 2], "TEST-Case": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        result_df = ar.to_pandas(result)

        # Should clean special chars but preserve original case
        assert len(result_df.columns) == 2
        assert all(isinstance(col, str) for col in result_df.columns)

    def test_consecutive_underscores_collapsed(self):
        """Test that consecutive underscores/separators collapse to single underscore."""
        df = pd.DataFrame({"col__with---multiple____separators": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        col_name = list(result_df.columns)[0]
        assert "__" not in col_name
        assert "---" not in col_name
        assert "____" not in col_name

    def test_leading_trailing_underscores_removed(self):
        """Test that leading and trailing underscores are trimmed."""
        df = pd.DataFrame({"___col_name___": [1, 2], "---test--": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        for col in result_df.columns:
            assert not col.startswith("_")
            assert not col.endswith("_")
            assert col  # not empty

    def test_special_characters_replaced(self):
        """Test that special characters are replaced with underscores."""
        df = pd.DataFrame({
            "col@name": [1],
            "test#value": [2],
            "amount$": [3],
            "rate%change": [4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        for col in result_df.columns:
            assert "@" not in col
            assert "#" not in col
            assert "$" not in col
            assert "%" not in col

    def test_spaces_replaced_with_underscores(self):
        """Test that spaces are replaced with underscores."""
        df = pd.DataFrame({
            "hello world": [1],
            "multi word column": [2],
            "a b c d": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        for col in result_df.columns:
            assert " " not in col

    def test_already_clean_columns_unchanged(self):
        """Test that already clean column names remain unchanged."""
        df = pd.DataFrame({"name": [1], "age": [2], "email": [3]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        assert set(result_df.columns) == {"name", "age", "email"}

    def test_numeric_characters_preserved(self):
        """Test that numeric characters in names are preserved."""
        df = pd.DataFrame({
            "col1": [1],
            "test_2_value": [2],
            "abc123xyz": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        assert "col1" in result_df.columns
        assert "test_2_value" in result_df.columns
        assert "abc123xyz" in result_df.columns

    def test_invalid_case_type_raises_value_error(self):
        """Test that invalid case_type raises ValueError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.clean_column_names(frame, case_type="invalid_case_type")

    def test_non_string_case_type_raises_type_error(self):
        """Test that non-string case_type raises TypeError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError):
            ar.clean_column_names(frame, case_type=123)

    def test_duplicate_column_names_after_cleaning_raises_error(self):
        """Test that cleaning creating duplicate names raises ValueError."""
        df = pd.DataFrame({
            "column_name": [1],
            "column-name": [2],  # Would become "column_name" after cleaning
        })
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.clean_column_names(frame)

    def test_empty_dataframe_column_names_cleaned(self):
        """Test cleaning on empty dataframe."""
        df = pd.DataFrame({"Col A": [], "Col B": []})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)

        assert set(result_df.columns) == {"col_a", "col_b"}
        assert len(result_df) == 0
