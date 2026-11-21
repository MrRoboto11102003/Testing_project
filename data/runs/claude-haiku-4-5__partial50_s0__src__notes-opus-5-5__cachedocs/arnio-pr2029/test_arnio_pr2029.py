import pytest
import pandas as pd
import arnio as ar
import re


class TestCleanColumnNames:
    """Test suite for arnio.cleaning.clean_column_names"""

    def test_lower_case(self):
        """Test basic lowercase normalization."""
        df = pd.DataFrame({
            "First Name": [1, 2],
            "E-Mail": [3, 4],
            "AGE": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        assert result.columns == ["first_name", "e_mail", "age"]
        assert ar.to_pandas(result)["first_name"].tolist() == [1, 2]

    def test_upper_case(self):
        """Test uppercase normalization."""
        df = pd.DataFrame({
            "first-name": [1, 2],
            "email": [3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        
        assert result.columns == ["FIRST_NAME", "EMAIL"]

    def test_title_case(self):
        """Test title case normalization."""
        df = pd.DataFrame({
            "first name": [1, 2],
            "id": [3, 4],
            "abc1def": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        
        assert result.columns == ["First_Name", "Id", "Abc1Def"]

    def test_camel_case(self):
        """Test camel case normalization."""
        df = pd.DataFrame({
            "First Name": [1, 2],
            "user_ID_2": [3, 4],
            "a_b_c": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        
        assert result.columns == ["firstName", "userId2", "aBC"]

    def test_none_case(self):
        """Test no case change (only cleaning of non-alphanumerics)."""
        df = pd.DataFrame({
            "First Name!": [1, 2],
            "__X__": [3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        
        assert result.columns == ["First_Name", "X"]

    def test_character_and_underscore_handling(self):
        """Test non-ASCII replacement, underscore collapsing, and trimming."""
        df = pd.DataFrame({
            "naïve": [1, 2],
            "café": [3, 4],
            "a___b": [5, 6],
            "___x___": [7, 8]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        # Non-ASCII chars replaced with _, consecutive underscores collapsed, 
        # leading/trailing underscores trimmed
        assert result.columns == ["na_ve", "caf_", "a_b", "x"]

    def test_empty_name_becomes_column(self):
        """Test that names that become empty after cleaning are set to 'column'."""
        df = pd.DataFrame({
            "___": [1, 2],
            "!!": [3, 4],
            "é": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        # All three become "column" after cleaning (all non-alphanumeric chars)
        assert result.columns == ["column", "column", "column"]

    def test_duplicate_error(self):
        """Test ValueError when cleaning creates duplicate column names."""
        df = pd.DataFrame({
            "a b": [1, 2],
            "a_b": [3, 4]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match=re.escape("Cleaning column names would create duplicates: ['a_b', 'a_b']")):
            ar.clean_column_names(frame, case_type="lower")

    def test_duplicate_error_case_sensitive(self):
        """Test ValueError for duplicates created by case normalization."""
        df = pd.DataFrame({
            "First": [1, 2],
            "first": [3, 4]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match=re.escape("Cleaning column names would create duplicates:")):
            ar.clean_column_names(frame, case_type="lower")

    def test_invalid_case_type_string(self):
        """Test ValueError for invalid case_type string."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="case_type must be one of 'lower', 'upper', 'title', 'camel' or 'none'"):
            ar.clean_column_names(frame, case_type="snake")

    def test_case_type_not_string(self):
        """Test TypeError when case_type is not a string."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=1)
        
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=None)

    def test_no_changes_needed(self):
        """Test that a new frame is returned even when no name changes occur."""
        df = pd.DataFrame({"a": [1, 2], "b_c": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        assert result.columns == ["a", "b_c"]
        assert result is not frame  # Always returns new object
        assert ar.to_pandas(result)["a"].tolist() == [1, 2]

    def test_zero_row_frame(self):
        """Test with a frame that has columns but zero rows."""
        df = pd.DataFrame({"First Name": [], "E-Mail": []})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        assert result.columns == ["first_name", "e_mail"]
        assert result.shape == (0, 2)

    def test_zero_column_frame(self):
        """Test with a frame that has no columns."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        assert result.columns == []
        assert result.shape == (0, 0)

    def test_data_integrity(self):
        """Test that data values, types, and row count are preserved."""
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "E-Mail": ["a", "b", "c"],
            "Score": [1.5, 2.5, 3.5]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="lower")
        
        result_df = ar.to_pandas(result)
        assert result_df["first_name"].tolist() == [1, 2, 3]
        assert result_df["e_mail"].tolist() == ["a", "b", "c"]
        assert result_df["score"].tolist() == [1.5, 2.5, 3.5]
        assert result.shape == frame.shape

    def test_default_case_type_is_lower(self):
        """Test that the default case_type is 'lower'."""
        df = pd.DataFrame({"First-Name": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)  # No case_type specified
        
        assert result.columns == ["first_name"]
