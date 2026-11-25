import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumns:
    """Tests for arnio.cleaning.drop_columns"""
    
    def test_drop_single_column(self):
        """Drop a single column from the frame"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["b"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["a", "c"]
        assert result_df.shape == (3, 2)
    
    def test_drop_multiple_columns(self):
        """Drop multiple columns from the frame"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["age", "city"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["id", "name"]
        assert result_df.shape == (3, 2)
    
    def test_preserves_column_order(self):
        """Remaining columns preserve their original order"""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "col3": [5, 6],
            "col4": [7, 8],
            "col5": [9, 10]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["col2", "col4"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["col1", "col3", "col5"]
    
    def test_empty_columns_list(self):
        """Dropping no columns returns all columns unchanged"""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, [])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["a", "b"]
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), df)
    
    def test_nonexistent_column_raises_valueerror(self):
        """Raises ValueError when trying to drop a column that doesn't exist"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.drop_columns(frame, ["nonexistent"])
    
    def test_one_invalid_column_in_list_raises_valueerror(self):
        """Raises ValueError when one of multiple columns doesn't exist"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.drop_columns(frame, ["a", "nonexistent"])
    
    def test_invalid_columns_type_raises_typeerror(self):
        """Raises TypeError when columns is not a sequence of strings"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.drop_columns(frame, "a")
    
    def test_returns_new_frame_does_not_modify_original(self):
        """drop_columns returns a new ArFrame without modifying the original"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        original_cols = frame.columns.copy()
        
        result = ar.drop_columns(frame, ["a"])
        
        assert frame.columns == original_cols
        assert result.columns == ["b"]
    
    def test_data_integrity_after_drop(self):
        """Data values are preserved correctly after dropping columns"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "drop_me": [100, 200, 300],
            "value": [10.5, 20.5, 30.5]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["drop_me"])
        result_df = ar.to_pandas(result)
        
        expected = df[["id", "value"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(result_df, expected)
    
    def test_preserves_dtypes_of_remaining_columns(self):
        """Data types of remaining columns are preserved"""
        df = pd.DataFrame({
            "int_col": np.array([1, 2, 3], dtype=np.int64),
            "str_col": ["a", "b", "c"],
            "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64)
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["int_col"])
        result_df = ar.to_pandas(result)
        
        assert result_df["float_col"].dtype == np.float64
    
    def test_drop_all_columns_raises_error(self):
        """Attempting to drop all columns should raise an error"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, RuntimeError)):
            ar.drop_columns(frame, ["a", "b"])
    
    def test_case_sensitive_column_names(self):
        """Column names are case-sensitive"""
        df = pd.DataFrame({"Name": [1, 2], "name": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["Name"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name"]
    
    def test_drop_columns_with_nulls(self):
        """Drop columns containing null values"""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, 5, None],
            "c": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["a"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["b", "c"]
        assert result_df.shape == (3, 2)
    
    def test_drop_columns_mixed_types(self):
        """Drop columns from mixed numeric and string data"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["name"])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["id", "score"]
        assert len(result_df) == 3
