import pytest
import pandas as pd
import arnio as ar


class TestCleanColumnNames:
    """Tests for arnio.cleaning.clean_column_names function."""

    def test_default_lowercase_normalization(self):
        """Test default case_type='lower' converts to lowercase."""
        df = pd.DataFrame({
            "FirstName": [1, 2],
            "LastName": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["firstname", "lastname"]

    def test_upper_case_normalization(self):
        """Test case_type='upper' converts names to uppercase."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["FIRST_NAME", "LAST_NAME"]

    def test_title_case_normalization(self):
        """Test case_type='title' converts to Title_Case with underscores."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["First_Name", "Last_Name"]

    def test_camel_case_normalization(self):
        """Test case_type='camel' converts to camelCase."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["firstName", "lastName"]

    def test_case_type_none_preserves_case(self):
        """Test case_type='none' preserves original case after cleaning special chars."""
        df = pd.DataFrame({
            "First-Name": [1, 2],
            "LAST_NAME": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        result_df = ar.to_pandas(result)
        cols = list(result_df.columns)
        assert "First_Name" in cols
        assert "LAST_NAME" in cols

    def test_special_characters_replaced_with_underscore(self):
        """Test non-alphanumeric characters are replaced with underscores."""
        df = pd.DataFrame({
            "first-name": [1, 2],
            "last@name": [3, 4],
            "age#value": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["first_name", "last_name", "age_value"]

    def test_consecutive_underscores_collapsed(self):
        """Test consecutive underscores are collapsed into single underscore."""
        df = pd.DataFrame({
            "first__name": [1, 2],
            "last___name": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["first_name", "last_name"]

    def test_boundary_underscores_trimmed(self):
        """Test leading and trailing underscores are removed."""
        df = pd.DataFrame({
            "_firstname_": [1, 2],
            "__lastname__": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["firstname", "lastname"]

    def test_empty_name_becomes_column(self):
        """Test that column names becoming empty after cleaning are renamed to 'column'."""
        df = pd.DataFrame({
            "@@@": [1, 2],
            "valid": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        cols = list(result_df.columns)
        assert "column" in cols
        assert "valid" in cols

    def test_duplicate_names_raises_error(self):
        """Test ValueError is raised when cleaning creates duplicate column names."""
        df = pd.DataFrame({
            "first-name": [1, 2],
            "first_name": [3, 4],
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="duplicate"):
            ar.clean_column_names(frame)

    def test_invalid_case_type_raises_error(self):
        """Test ValueError is raised for invalid case_type value."""
        df = pd.DataFrame({"name": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="invalid")

    def test_non_string_case_type_raises_error(self):
        """Test TypeError is raised when case_type is not a string."""
        df = pd.DataFrame({"name": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=123)

    def test_no_changes_returns_copy(self):
        """Test that already-clean names return a new frame (deepcopy)."""
        df = pd.DataFrame({
            "first_name": [1, 2],
            "last_name": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["first_name", "last_name"]
        assert result is not frame

    def test_mixed_special_chars_and_numbers(self):
        """Test cleaning columns with mixed special characters and numbers."""
        df = pd.DataFrame({
            "col-1-name": [1, 2],
            "col_2@data": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col_1_name", "col_2_data"]
