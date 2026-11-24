import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


class TestParseBoolStrings:
    """Test suite for arnio.cleaning.parse_bool_strings"""
    
    def test_default_values_parsing(self):
        """Test that default true/false values are parsed correctly"""
        df = pd.DataFrame({
            "col": ["yes", "no", "true", "false", "y", "n", "1", "0", "YES", "NO"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        expected = [True, False, True, False, True, False, True, False, True, False]
        assert result_df["col"].tolist() == expected
    
    def test_whitespace_and_case_insensitivity(self):
        """Test that matching strips whitespace and is case-insensitive"""
        df = pd.DataFrame({
            "col": [" YES ", "  no  ", "\ttrue\n", "FALSE"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["col"].tolist() == [True, False, True, False]
    
    def test_unsupported_values_preserved(self):
        """Test that non-matching strings and None values are preserved"""
        df = pd.DataFrame({
            "col": ["yes", "maybe", "no", None, "true", "unknown"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        values = result_df["col"].tolist()
        assert values[0] == True
        assert values[1] == "maybe"
        assert values[2] == False
        assert pd.isna(values[3])
        assert values[4] == True
        assert values[5] == "unknown"
    
    def test_custom_true_values_replace_defaults(self):
        """Test that custom true_values completely replace defaults"""
        df = pd.DataFrame({
            "col": ["yes", "si", "yep", "no"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, true_values={"si", "yep"})
        result_df = ar.to_pandas(result)
        
        values = result_df["col"].tolist()
        assert values[0] == "yes"  # Not in custom true_values
        assert values[1] == True
        assert values[2] == True
        assert values[3] == False  # Still uses default false_values
    
    def test_custom_false_values_replace_defaults(self):
        """Test that custom false_values completely replace defaults"""
        df = pd.DataFrame({
            "col": ["yes", "nope", "nada", "true"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, false_values={"nope", "nada"})
        result_df = ar.to_pandas(result)
        
        values = result_df["col"].tolist()
        assert values[0] == True  # Still uses default true_values
        assert values[1] == False
        assert values[2] == False
        assert values[3] == True
    
    def test_subset_limits_columns(self):
        """Test that subset parameter applies conversion only to specified columns"""
        df = pd.DataFrame({
            "a": ["yes", "no"],
            "b": ["true", "false"],
            "c": [1, 2]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["a"])
        result_df = ar.to_pandas(result)
        
        assert result_df["a"].tolist() == [True, False]
        assert result_df["b"].tolist() == ["true", "false"]  # unchanged
        assert result_df["c"].tolist() == [1, 2]  # unchanged
    
    def test_subset_skips_boolean_columns(self):
        """Test that boolean columns named in subset are skipped"""
        df = pd.DataFrame({
            "bool_col": [True, False],
            "str_col": ["yes", "no"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["bool_col", "str_col"])
        result_df = ar.to_pandas(result)
        
        assert result_df["bool_col"].dtype == "bool"
        assert result_df["bool_col"].tolist() == [True, False]
        assert result_df["str_col"].tolist() == [True, False]
    
    def test_empty_value_collections(self):
        """Test that empty true/false value collections work correctly"""
        df = pd.DataFrame({
            "col": ["yes", "no", "true", "false"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, true_values=set(), false_values={"no"})
        result_df = ar.to_pandas(result)
        
        values = result_df["col"].tolist()
        assert values[0] == "yes"  # Not converted
        assert values[1] == False
        assert values[2] == "true"  # Not converted
        assert values[3] == "false"  # Not in false_values
    
    def test_error_bare_string_for_values(self):
        """Test that bare strings raise TypeError for true_values and false_values"""
        df = pd.DataFrame({"col": ["yes"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match=re.escape("true_values must be a set/list/tuple of strings, not a bare string")):
            ar.parse_bool_strings(frame, true_values="yes")
        
        with pytest.raises(TypeError, match=re.escape("false_values must be a set/list/tuple of strings, not a bare string")):
            ar.parse_bool_strings(frame, false_values="no")
    
    def test_error_non_string_elements_in_values(self):
        """Test that non-string elements in value collections raise TypeError"""
        df = pd.DataFrame({"col": ["yes"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="true_values must contain only strings, got int"):
            ar.parse_bool_strings(frame, true_values=["yes", 1])
        
        with pytest.raises(TypeError, match="false_values must contain only strings, got float"):
            ar.parse_bool_strings(frame, false_values=["no", 1.5])
    
    def test_error_overlapping_values_after_normalization(self):
        """Test that overlap between true and false values raises ValueError"""
        df = pd.DataFrame({"col": ["yes"]})
        frame = ar.from_pandas(df)
        
        # Direct overlap
        with pytest.raises(ValueError, match=re.escape("true_values and false_values overlap after normalization")):
            ar.parse_bool_strings(frame, true_values={"yes"}, false_values={" YES "})
        
        # Overlap with defaults
        with pytest.raises(ValueError, match=re.escape("true_values and false_values overlap after normalization")):
            ar.parse_bool_strings(frame, true_values={"no"})  # "no" is default false
    
    def test_error_invalid_subset_parameter(self):
        """Test that invalid subset parameters raise appropriate errors"""
        df = pd.DataFrame({"col": ["yes"]})
        frame = ar.from_pandas(df)
        
        # Nonexistent column
        with pytest.raises(ValueError, match=re.escape("Columns not found in frame")):
            ar.parse_bool_strings(frame, subset=["nonexistent"])
    
    def test_input_frame_not_mutated(self):
        """Test that the input frame is not mutated by the function"""
        df = pd.DataFrame({"col": ["yes", "no"]})
        frame = ar.from_pandas(df)
        original_values = ar.to_pandas(frame)["col"].tolist()
        
        result = ar.parse_bool_strings(frame)
        
        # Input should be unchanged
        assert ar.to_pandas(frame)["col"].tolist() == original_values
        # Result should be different
        assert ar.to_pandas(result)["col"].tolist() == [True, False]
    
    def test_zero_row_frame_handling(self):
        """Test that zero-row frames are handled correctly"""
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ["col"]
    
    def test_subset_none_applies_to_object_string_columns_only(self):
        """Test that subset=None only converts object/string columns, not numeric ones"""
        df = pd.DataFrame({
            "strings": ["yes", "no"],
            "integers": [1, 2],
            "floats": [1.5, 2.5]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["strings"].tolist() == [True, False]
        assert result_df["integers"].tolist() == [1, 2]
        assert result_df["floats"].tolist() == [1.5, 2.5]
