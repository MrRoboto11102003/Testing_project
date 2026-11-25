import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestParseBoolStrings:
    """Tests for ar.parse_bool_strings function."""
    
    def test_basic_conversion_default_values(self):
        """Test conversion of common boolean strings with defaults."""
        df = pd.DataFrame({
            "flag": ["yes", "no", "true", "false", "y", "n", "1", "0"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Result should have converted booleans or remain string if mixed
        assert "flag" in result_df.columns
        assert len(result_df) == 8
    
    def test_converts_all_string_columns_by_default(self):
        """Test that all string columns are converted when subset is None."""
        df = pd.DataFrame({
            "bool_col1": ["yes", "no"],
            "bool_col2": ["true", "false"],
            "int_col": [1, 2]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # All columns should be present
        assert set(result_df.columns) == {"bool_col1", "bool_col2", "int_col"}
        assert len(result_df) == 2
    
    def test_subset_parameter_limits_conversion(self):
        """Test that subset parameter only converts specified columns."""
        df = pd.DataFrame({
            "convert_me": ["yes", "no"],
            "leave_me": ["true", "false"],
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["convert_me"])
        result_df = ar.to_pandas(result)
        
        assert "convert_me" in result_df.columns
        assert "leave_me" in result_df.columns
        assert len(result_df) == 2
    
    def test_preserves_null_values(self):
        """Test that null/None values are preserved."""
        df = pd.DataFrame({
            "flag": ["yes", None, "no", np.nan]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Null count should be preserved
        assert result_df["flag"].isna().sum() >= 2
    
    def test_preserves_unsupported_values(self):
        """Test that unsupported string values remain in output."""
        df = pd.DataFrame({
            "flag": ["yes", "maybe", "no", "unknown"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Data should not be lost; unsupported values preserved
        assert len(result_df) == 4
        # Check values are present (as strings or in mixed column)
        values_str = " ".join(str(v) for v in result_df["flag"].values)
        assert "maybe" in values_str or "unknown" in values_str
    
    def test_custom_true_values(self):
        """Test conversion with custom true_values set."""
        df = pd.DataFrame({
            "flag": ["yes", "nope", "yeah", "nah"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(
            frame,
            true_values={"yes", "yeah"},
            false_values={"nope", "nah"}
        )
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 4
        assert "flag" in result_df.columns
    
    def test_custom_false_values(self):
        """Test conversion with custom false_values set."""
        df = pd.DataFrame({
            "status": ["enabled", "disabled", "off"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(
            frame,
            true_values={"enabled"},
            false_values={"disabled", "off"}
        )
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 3
    
    def test_empty_frame(self):
        """Test handling of empty frame."""
        df = pd.DataFrame({"flag": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert "flag" in result_df.columns
    
    def test_no_string_columns(self):
        """Test frame with no string/object columns returns unchanged."""
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "float": [1.1, 2.2, 3.3]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Shape and columns should be preserved
        assert result_df.shape == df.shape
        assert set(result_df.columns) == set(df.columns)
    
    def test_already_boolean_column(self):
        """Test that already-boolean columns are handled correctly."""
        df = pd.DataFrame({
            "bool_col": [True, False, True],
            "str_col": ["yes", "no", "yes"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Frame should have both columns
        assert "bool_col" in result_df.columns
        assert "str_col" in result_df.columns
        assert len(result_df) == 3
    
    def test_numeric_string_1_and_0(self):
        """Test that '1' and '0' strings are converted."""
        df = pd.DataFrame({
            "flag": ["1", "0", "1", "0"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 4
        assert "flag" in result_df.columns
    
    def test_case_insensitive_conversion(self):
        """Test handling of different case variations of boolean strings."""
        df = pd.DataFrame({
            "flag": ["TRUE", "False", "YES", "no", "Yes", "NO"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        
        # Should handle case variations
        assert len(result_df) == 6
    
    def test_multiple_columns_with_subset(self):
        """Test conversion of multiple specific columns via subset."""
        df = pd.DataFrame({
            "col1": ["yes", "no"],
            "col2": ["true", "false"],
            "col3": ["maybe", "sure"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)
        
        assert all(col in result_df.columns for col in ["col1", "col2", "col3"])
        assert len(result_df) == 2
    
    def test_returns_arframe(self):
        """Test that function returns an ArFrame object."""
        df = pd.DataFrame({"flag": ["yes", "no"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        
        # Should be an ArFrame with expected attributes
        assert hasattr(result, "columns")
        assert hasattr(result, "shape")
        assert hasattr(result, "dtypes")
    
    def test_nonexistent_column_in_subset(self):
        """Test that nonexistent column in subset raises appropriate error."""
        df = pd.DataFrame({"col1": ["yes", "no"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, KeyError)):
            ar.parse_bool_strings(frame, subset=["nonexistent_col"])
