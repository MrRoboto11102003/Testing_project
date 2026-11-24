import pytest
import pandas as pd
import arnio as ar
import re


class TestCleanColumnNames:
    
    def test_basic_cleaning_lower(self):
        """Test basic cleaning with default lower case."""
        df = pd.DataFrame({"First Name": [1, 2], "e-mail": ["x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == ["first_name", "e_mail"]
        assert ar.to_pandas(result)["first_name"].tolist() == [1, 2]
    
    def test_case_type_upper(self):
        """Test upper case normalization."""
        df = pd.DataFrame({"first-name": [1], "age": [25]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="upper")
        assert list(result.columns) == ["FIRST_NAME", "AGE"]
    
    def test_case_type_title(self):
        """Test title case normalization."""
        df = pd.DataFrame({"first name": [1], "abc1def": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="title")
        assert list(result.columns) == ["First_Name", "Abc1Def"]
    
    def test_case_type_camel(self):
        """Test camel case normalization."""
        df = pd.DataFrame({"First Name": [1], "user_ID_2": [2], "a_b_c": [3]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="camel")
        assert list(result.columns) == ["firstName", "userId2", "aBC"]
    
    def test_case_type_none(self):
        """Test no case change."""
        df = pd.DataFrame({"First Name!": [1], "__X__": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame, case_type="none")
        assert list(result.columns) == ["First_Name", "X"]
    
    def test_non_ascii_replaced(self):
        """Test that non-ASCII characters are replaced with underscores."""
        df = pd.DataFrame({"naïve": [1], "café": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == ["na_ve", "caf_"]
    
    def test_underscores_collapsed_and_trimmed(self):
        """Test that consecutive underscores are collapsed and boundary underscores trimmed."""
        df = pd.DataFrame({"__a__b__": [1], "___": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == ["a_b", "column"]
    
    def test_empty_name_becomes_column(self):
        """Test that names that become empty are replaced with 'column'."""
        df = pd.DataFrame({"!!!": [1], "??": [2]})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == ["column", "column"]
    
    def test_duplicate_symbols_raises_error(self):
        """Test that duplicate column names after cleaning raise ValueError."""
        df = pd.DataFrame({"a b": [1], "a_b": [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("Cleaning column names would create duplicates: ['a_b', 'a_b']")):
            ar.clean_column_names(frame)
    
    def test_duplicate_case_normalization_raises_error(self):
        """Test duplicates from case normalization."""
        df = pd.DataFrame({"First": [1], "first": [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("Cleaning column names would create duplicates: ['first', 'first']")):
            ar.clean_column_names(frame)
    
    def test_case_type_validation(self):
        """Test that invalid case_type values raise appropriate errors."""
        df = pd.DataFrame({"Name": [1]})
        frame = ar.from_pandas(df)
        
        # Non-string case_type
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=1)
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.clean_column_names(frame, case_type=None)
        
        # Invalid string case_type
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="snake")
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.clean_column_names(frame, case_type="Lower")
    
    def test_data_shape_preserved(self):
        """Test that data, dtypes, and shape are preserved."""
        df = pd.DataFrame({
            "First Name": [1, 2, 3],
            "Score!": [1.5, 2.5, 3.5],
            "Active?": [True, False, True]
        })
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        result_pd = ar.to_pandas(result)
        
        assert result.shape == frame.shape
        assert result_pd["first_name"].tolist() == [1, 2, 3]
        assert result_pd["score"].tolist() == [1.5, 2.5, 3.5]
    
    def test_zero_column_frame(self):
        """Test with zero-column frame."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == []
    
    def test_zero_rows_with_columns(self):
        """Test with zero rows but with columns."""
        df = pd.DataFrame({"A B": []})
        frame = ar.from_pandas(df)
        result = ar.clean_column_names(frame)
        assert list(result.columns) == ["a_b"]
        assert result.shape == (0, 1)
    
    def test_non_arframe_raises_error(self):
        """Test that non-ArFrame input raises an error."""
        df = pd.DataFrame({"Name": [1]})
        with pytest.raises(Exception):
            ar.clean_column_names(df)
