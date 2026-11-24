import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumns:
    """Test suite for arnio.cleaning.drop_columns"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.5, 2.5, None]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def two_column_frame(self):
        """Create a two-column ArFrame"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        return ar.from_pandas(df)
    
    # Basic functionality tests
    def test_drop_single_column(self, sample_frame):
        """Drop a single column"""
        result = ar.drop_columns(sample_frame, ["b"])
        assert list(result.columns) == ["a", "c"]
        assert result.shape == (3, 2)
    
    def test_drop_multiple_columns(self, sample_frame):
        """Drop multiple columns"""
        result = ar.drop_columns(sample_frame, ["b", "c"])
        assert list(result.columns) == ["a"]
    
    def test_column_order_preserved(self, sample_frame):
        """Remaining columns stay in original frame order"""
        result = ar.drop_columns(sample_frame, ["c", "a"])
        assert list(result.columns) == ["b"]
    
    def test_empty_column_list(self, sample_frame):
        """Empty column list returns same columns"""
        result = ar.drop_columns(sample_frame, [])
        assert list(result.columns) == ["a", "b", "c"]
    
    def test_values_preserved(self, sample_frame):
        """Values are preserved after drop"""
        result = ar.drop_columns(sample_frame, ["b"])
        result_df = ar.to_pandas(result)
        assert result_df["a"].tolist() == [1, 2, 3]
        assert pd.isna(result_df["c"].iloc[2])
    
    def test_input_frame_not_mutated(self, sample_frame):
        """Input frame is not mutated"""
        original_columns = list(sample_frame.columns)
        ar.drop_columns(sample_frame, ["b"])
        assert list(sample_frame.columns) == original_columns
    
    # Error case: invalid frame type
    def test_error_frame_not_arframe(self):
        """TypeError when frame is not an ArFrame"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(TypeError):
            ar.drop_columns(df, ["a"])
    
    # Error cases: invalid columns type
    def test_error_columns_is_string(self, sample_frame):
        """TypeError when columns is a string"""
        with pytest.raises(TypeError, match=re.escape("columns must be a sequence of column names, not a string")):
            ar.drop_columns(sample_frame, "b")
    
    def test_error_columns_is_set(self, sample_frame):
        """TypeError when columns is a set"""
        with pytest.raises(TypeError, match=re.escape("columns must be a sequence of column names")):
            ar.drop_columns(sample_frame, {"b"})
    
    def test_error_non_string_item(self, sample_frame):
        """TypeError when columns contains non-string items"""
        with pytest.raises(TypeError, match=re.escape("columns must contain only string column names")):
            ar.drop_columns(sample_frame, ["a", 1])
    
    # Error cases: invalid column names
    def test_error_missing_column(self, sample_frame):
        """ValueError when a requested column is missing"""
        with pytest.raises(ValueError, match=re.escape("Columns not found in frame: ['z']")):
            ar.drop_columns(sample_frame, ["z"])
    
    def test_error_drop_all_columns(self, sample_frame):
        """ValueError when trying to drop all columns"""
        with pytest.raises(ValueError, match=re.escape("drop_columns cannot remove all columns from the frame")):
            ar.drop_columns(sample_frame, ["a", "b", "c"])
    
    def test_error_duplicates_all_columns(self, two_column_frame):
        """ValueError when duplicates cause all-columns removal"""
        with pytest.raises(ValueError, match=re.escape("drop_columns cannot remove all columns from the frame")):
            ar.drop_columns(two_column_frame, ["a", "a"])
    
    # Test check ordering and edge cases
    def test_missing_checked_before_all_columns(self, two_column_frame):
        """Missing columns checked before all-columns check"""
        with pytest.raises(ValueError, match=re.escape("Columns not found in frame: ['z']")):
            ar.drop_columns(two_column_frame, ["a", "z"])
    
    def test_duplicates_harmless(self, sample_frame):
        """Duplicates are harmless when not dropping all columns"""
        result = ar.drop_columns(sample_frame, ["a", "a"])
        assert list(result.columns) == ["b", "c"]
