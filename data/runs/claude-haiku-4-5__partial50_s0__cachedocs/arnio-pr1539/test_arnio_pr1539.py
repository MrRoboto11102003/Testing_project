import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns()"""

    def setup_method(self):
        """Create test data for each test"""
        self.df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"]
        })
        self.frame = ar.from_pandas(self.df)

    def test_drop_single_column_with_list(self):
        """Test dropping a single column with list input"""
        result = self.frame.drop_columns(["name"])
        result_df = ar.to_pandas(result)
        
        assert "name" not in result_df.columns
        assert list(result_df.columns) == ["id", "age", "city"]
        assert len(result_df) == 3

    def test_drop_single_column_with_tuple(self):
        """Test dropping a single column with tuple input"""
        result = self.frame.drop_columns(("name",))
        result_df = ar.to_pandas(result)
        
        assert "name" not in result_df.columns
        assert list(result_df.columns) == ["id", "age", "city"]

    def test_drop_multiple_columns_with_list(self):
        """Test dropping multiple columns"""
        result = self.frame.drop_columns(["name", "city"])
        result_df = ar.to_pandas(result)
        
        assert "name" not in result_df.columns
        assert "city" not in result_df.columns
        assert list(result_df.columns) == ["id", "age"]

    def test_drop_multiple_columns_with_tuple(self):
        """Test dropping multiple columns with tuple input"""
        result = self.frame.drop_columns(("name", "city"))
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["id", "age"]

    def test_drop_preserves_column_order(self):
        """Test that remaining columns preserve their original order"""
        result = self.frame.drop_columns(["age"])
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["id", "name", "city"]

    def test_drop_empty_list_returns_copy(self):
        """Test that empty list returns a copy unchanged"""
        result = self.frame.drop_columns([])
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == list(self.frame.columns)
        assert result is not self.frame

    def test_drop_silently_ignores_duplicate_names(self):
        """Test that duplicate column names in cols are silently ignored"""
        result = self.frame.drop_columns(["name", "name", "city", "city"])
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["id", "age"]

    def test_drop_does_not_modify_original_frame(self):
        """Test that original frame is not mutated"""
        original_columns = list(self.frame.columns)
        self.frame.drop_columns(["name"])
        
        assert list(self.frame.columns) == original_columns

    def test_drop_all_columns(self):
        """Test dropping all columns results in empty frame"""
        result = self.frame.drop_columns(["id", "name", "age", "city"])
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3

    def test_drop_raises_type_error_for_non_sequence(self):
        """Test TypeError when cols is not a list or tuple"""
        with pytest.raises(TypeError):
            self.frame.drop_columns("name")
        
        with pytest.raises(TypeError):
            self.frame.drop_columns({"name"})

    def test_drop_raises_type_error_for_non_string_elements(self):
        """Test TypeError when cols contains non-string elements"""
        with pytest.raises(TypeError):
            self.frame.drop_columns([1, 2])
        
        with pytest.raises(TypeError):
            self.frame.drop_columns(["name", 123])
        
        with pytest.raises(TypeError):
            self.frame.drop_columns([None])

    def test_drop_raises_value_error_for_missing_column(self):
        """Test ValueError when a column doesn't exist"""
        with pytest.raises(ValueError):
            self.frame.drop_columns(["nonexistent"])
        
        with pytest.raises(ValueError):
            self.frame.drop_columns(["name", "missing_col"])

    def test_drop_data_values_preserved(self):
        """Test that data integrity is maintained after dropping columns"""
        result = self.frame.drop_columns(["name", "city"])
        result_df = ar.to_pandas(result)
        
        assert result_df["id"].tolist() == [1, 2, 3]
        assert result_df["age"].tolist() == [25, 30, 35]

    def test_drop_single_column_frame(self):
        """Test dropping the only column from a single-column frame"""
        single_frame = ar.from_pandas(pd.DataFrame({"only": [1, 2, 3]}))
        result = single_frame.drop_columns(["only"])
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3
