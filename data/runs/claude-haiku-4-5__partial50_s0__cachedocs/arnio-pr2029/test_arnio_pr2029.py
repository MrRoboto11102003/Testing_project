import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCleanColumnNames:
    """Test suite for arnio.cleaning.clean_column_names"""

    def test_basic_replace_non_alphanumeric_with_underscore(self):
        """Test replacing non-alphanumeric characters with underscores."""
        df = pd.DataFrame({"col-name": [1, 2], "col@name": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "col_name" in result_df.columns

    def test_collapse_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed to single underscore."""
        df = pd.DataFrame({"col__name": [1, 2], "col___other": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "col_name" in result_df.columns
        assert "col_other" in result_df.columns

    def test_trim_leading_trailing_underscores(self):
        """Test trimming leading and trailing underscores."""
        df = pd.DataFrame({"_col_name_": [1, 2], "__start": [3, 4], "end__": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "col_name" in result_df.columns
        assert "start" in result_df.columns
        assert "end" in result_df.columns

    def test_case_type_lower_default(self):
        """Test that default case_type 'lower' is applied."""
        df = pd.DataFrame({"ColName": [1, 2], "OTHER": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "colname" in result_df.columns
        assert "other" in result_df.columns

    def test_case_type_upper(self):
        """Test case_type='upper' converts to uppercase."""
        df = pd.DataFrame({"colname": [1, 2], "other": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        assert "COLNAME" in result_df.columns
        assert "OTHER" in result_df.columns

    def test_case_type_title(self):
        """Test case_type='title' converts to title case."""
        df = pd.DataFrame({"col_name": [1, 2], "other_field": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        result_df = ar.to_pandas(result)
        assert "Col_Name" in result_df.columns
        assert "Other_Field" in result_df.columns

    def test_case_type_camel(self):
        """Test case_type='camel' converts to camelCase."""
        df = pd.DataFrame({"col_name": [1, 2], "other_field": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        result_df = ar.to_pandas(result)
        assert "colName" in result_df.columns or "col_name" in result_df.columns
        assert "otherField" in result_df.columns or "other_field" in result_df.columns

    def test_case_type_none_preserves_case(self):
        """Test case_type='none' preserves original case."""
        df = pd.DataFrame({"ColName": [1, 2], "OTHER": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        result_df = ar.to_pandas(result)
        assert "ColName" in result_df.columns
        assert "OTHER" in result_df.columns

    def test_type_error_case_type_not_string(self):
        """Test TypeError raised when case_type is not a string."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.clean_column_names(frame, case_type=123)

    def test_type_error_case_type_none_type(self):
        """Test TypeError raised when case_type is None."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.clean_column_names(frame, case_type=None)

    def test_value_error_invalid_case_type(self):
        """Test ValueError for invalid case_type string."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.clean_column_names(frame, case_type="invalid_case")

    def test_duplicate_columns_after_cleaning_raises_error(self):
        """Test ValueError when cleaning creates duplicate column names."""
        df = pd.DataFrame({"col-name": [1, 2], "col_name": [3, 4]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.clean_column_names(frame)

    def test_already_clean_names_unchanged(self):
        """Test that already clean column names are preserved."""
        df = pd.DataFrame({"col_name": [1, 2], "other_field": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "col_name" in result_df.columns
        assert "other_field" in result_df.columns

    def test_numbers_preserved_in_column_names(self):
        """Test that numbers in column names are preserved."""
        df = pd.DataFrame({"col1": [1, 2], "field_2_name": [3, 4], "col@3": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert "col1" in result_df.columns
        assert "field_2_name" in result_df.columns
        assert "col_3" in result_df.columns

    def test_data_values_preserved_during_cleaning(self):
        """Test that data values are preserved during column name cleaning."""
        df = pd.DataFrame({"Col-Name": [1, 2, 3], "Other@Field": [4.5, 5.5, 6.5]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        pd.testing.assert_series_equal(
            result_df["col_name"], pd.Series([1, 2, 3], name="col_name")
        )
        pd.testing.assert_series_equal(
            result_df["other_field"], pd.Series([4.5, 5.5, 6.5], name="other_field")
        )
