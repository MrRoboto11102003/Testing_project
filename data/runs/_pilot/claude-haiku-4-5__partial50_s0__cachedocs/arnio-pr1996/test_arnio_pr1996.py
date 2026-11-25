import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Test suite for ar.drop_empty_columns()"""
    
    def test_drop_single_empty_column(self):
        """Remove a single column with all null values"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "empty_col": [None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name", "age"]
        assert len(result_df) == 2
    
    def test_drop_multiple_empty_columns(self):
        """Remove multiple columns with all null values"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "empty1": [None, None],
            "age": [25, 30],
            "empty2": [np.nan, np.nan]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name", "age"]
        assert len(result_df) == 2
    
    def test_no_empty_columns(self):
        """No columns removed when all have at least one non-null value"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "city": ["Paris", None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name", "age", "city"]
    
    def test_all_columns_empty(self):
        """All columns are empty - returns frame with no columns but same rows"""
        df = pd.DataFrame({
            "empty1": [None, None, None],
            "empty2": [np.nan, np.nan, np.nan]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3
    
    def test_single_row_with_empty_columns(self):
        """Single row with empty columns"""
        df = pd.DataFrame({
            "name": ["Alice"],
            "empty": [None],
            "age": [25]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name", "age"]
        assert len(result_df) == 1
    
    def test_numeric_columns_with_all_nan(self):
        """Numeric columns with all NaN values are removed"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "empty_numeric": [np.nan, np.nan, np.nan],
            "score": [90.5, 85.0, 92.3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["id", "score"]
        assert len(result_df) == 3
    
    def test_string_columns_with_all_none(self):
        """String columns with all None values are removed"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "empty_str": [None, None, None],
            "city": ["NYC", "LA", "SF"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["name", "city"]
        assert len(result_df) == 3
    
    def test_preserves_column_order(self):
        """Remaining columns maintain their original order"""
        df = pd.DataFrame({
            "first": [1, 2],
            "empty1": [None, None],
            "second": ["a", "b"],
            "empty2": [np.nan, np.nan],
            "third": [10.5, 20.5]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["first", "second", "third"]
    
    def test_preserves_data_values(self):
        """Remaining columns preserve their data values"""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "empty": [None, None],
            "age": [25, 30]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Alice", "Bob"]
        assert result_df["age"].tolist() == [25, 30]
    
    def test_column_with_single_value_not_empty(self):
        """Column with a single non-null value among nulls is not considered empty"""
        df = pd.DataFrame({
            "data": ["a", None, None],
            "mostly_empty": [None, "b", None],
            "all_empty": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "data" in result_df.columns
        assert "mostly_empty" in result_df.columns
        assert "all_empty" not in result_df.columns
    
    def test_returns_arframe(self):
        """Result is an ArFrame, not a pandas DataFrame"""
        df = pd.DataFrame({
            "col": [1, 2],
            "empty": [None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert isinstance(result, ar.ArFrame)
    
    def test_does_not_modify_input_frame(self):
        """Original frame is not modified"""
        df = pd.DataFrame({
            "name": ["Alice"],
            "empty": [None]
        })
        frame = ar.from_pandas(df)
        original_cols = frame.columns.copy()
        
        _ = ar.drop_empty_columns(frame)
        
        assert frame.columns == original_cols
    
    def test_empty_dataframe(self):
        """DataFrame with zero rows"""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype='object'),
            "col2": pd.Series([], dtype='object')
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
    
    def test_single_column_empty(self):
        """DataFrame with single column that is empty"""
        df = pd.DataFrame({
            "empty": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3
    
    def test_boolean_nulls_with_mixed_values(self):
        """Boolean column with mixed nulls and values is preserved"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "flags": [True, None, False],
            "all_null_bool": [np.nan, np.nan, np.nan]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "id" in result_df.columns
        assert "flags" in result_df.columns
        assert "all_null_bool" not in result_df.columns
