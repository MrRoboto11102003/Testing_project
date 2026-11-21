import re
import pytest
import pandas as pd
import arnio as ar


class TestDropColumnsMatching:
    """Tests for arnio.cleaning.drop_columns_matching"""

    def test_partial_match_with_arframe(self):
        """Test dropping columns that match a pattern with ArFrame input."""
        df = pd.DataFrame({
            "temp_a": [1, 2],
            "temp_b": [3, 4],
            "keep_me": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        
        assert isinstance(result, ar.ArFrame)
        assert result.columns == ["keep_me"]
        assert result.shape == (2, 1)
        
    def test_partial_match_with_dataframe(self):
        """Test dropping columns that match a pattern with DataFrame input."""
        df = pd.DataFrame({
            "temp_a": [1, 2],
            "temp_b": [3, 4],
            "keep_me": [5, 6],
        })
        result = ar.drop_columns_matching(df, "^temp_")
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["keep_me"]
        assert result.shape == (2, 1)

    def test_no_columns_match(self):
        """Test when pattern matches no columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^zzz")
        
        assert isinstance(result, ar.ArFrame)
        assert result.columns == ["a", "b"]
        assert result.shape == (2, 2)

    def test_all_columns_match_raises_error(self):
        """Test that matching all columns raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match=re.escape("Pattern matches all columns. At least one column must remain.")):
            ar.drop_columns_matching(frame, "")

    def test_pattern_type_error_int(self):
        """Test that non-string pattern raises TypeError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="pattern must be a string, got int"):
            ar.drop_columns_matching(frame, 5)

    def test_pattern_type_error_none(self):
        """Test that None pattern raises TypeError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="pattern must be a string, got NoneType"):
            ar.drop_columns_matching(frame, None)

    def test_invalid_regex_pattern(self):
        """Test that invalid regex pattern raises re.error."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(re.error, match=re.escape("Invalid regex pattern: '['")):
            ar.drop_columns_matching(frame, "[")

    def test_frame_type_error(self):
        """Test that invalid frame type raises TypeError."""
        with pytest.raises(TypeError, match="frame must be an ArFrame or a pandas DataFrame"):
            ar.drop_columns_matching([1, 2], "pattern")

    def test_zero_column_frame(self):
        """Test zero-column frame returns input object itself."""
        df = pd.DataFrame(index=range(3))
        result = ar.drop_columns_matching(df, "")
        
        # For zero-column frame, should return the same object
        assert result is df

    def test_zero_rows_some_columns(self):
        """Test zero-row frame with some columns works normally."""
        df = pd.DataFrame({"a": [], "b": [], "c": []})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^a$|^c$")
        
        assert isinstance(result, ar.ArFrame)
        assert result.columns == ["b"]
        assert result.shape == (0, 1)

    def test_case_sensitivity(self):
        """Test that pattern matching is case-sensitive."""
        df = pd.DataFrame({
            "TEMP_a": [1],
            "temp_b": [2],
            "keep": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp")
        
        # Should only match lowercase "temp_b"
        assert result.columns == ["TEMP_a", "keep"]

    def test_substring_matching_not_anchored(self):
        """Test that matching is substring-based (uses re.search, not re.match)."""
        df = pd.DataFrame({
            "my_temp": [1],
            "temp_x": [2],
            "keep": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "temp")
        
        # Both "my_temp" and "temp_x" contain "temp"
        assert result.columns == ["keep"]

    def test_non_string_column_labels(self):
        """Test that non-string column labels are converted to strings."""
        df = pd.DataFrame({0: [1], 1: [2], "x": [3]})
        result = ar.drop_columns_matching(df, r"^\d$")
        
        # Pattern ^\d$ should match numeric labels 0 and 1
        assert list(result.columns) == ["x"]

    def test_column_order_preservation(self):
        """Test that column order is preserved after dropping."""
        df = pd.DataFrame({"a": [1], "mid": [2], "z": [3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^mid$")
        
        assert result.columns == ["a", "z"]

    def test_dataframe_index_preservation(self):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame(
            {"temp_a": [1, 2], "keep": [3, 4]},
            index=["row1", "row2"]
        )
        result = ar.drop_columns_matching(df, "^temp_")
        
        assert list(result.index) == ["row1", "row2"]
