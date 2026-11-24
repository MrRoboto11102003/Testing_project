import pytest
import pandas as pd
import arnio as ar


class TestDropColumns:
    """Test ArFrame.drop_columns method."""
    
    def setup_method(self):
        """Create a test frame with columns a, b, c."""
        self.df = ar.from_pandas(pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", None, "z"],
            "c": [1.5, 2.5, 3.5]
        }))
    
    def test_drop_basic_operations(self):
        """Test basic drop operations: single column, empty, tuple."""
        result = self.df.drop_columns(["b"])
        assert result.columns == ["a", "c"]
        
        result = self.df.drop_columns([])
        assert result.columns == ["a", "b", "c"]
        assert result is not self.df
        
        result = self.df.drop_columns(("a", "c"))
        assert result.columns == ["b"]
    
    def test_drop_duplicates_and_order(self):
        """Test that duplicates are ignored and column order is preserved."""
        result = self.df.drop_columns(["b", "b"])
        assert result.columns == ["a", "c"]
        
        result = self.df.drop_columns(["c", "a"])
        assert result.columns == ["b"]
    
    def test_drop_all_columns_error(self):
        """Test that dropping all columns raises ValueError."""
        with pytest.raises(ValueError, match="drop_columns cannot remove all columns from the frame"):
            self.df.drop_columns(["a", "b", "c"])
    
    def test_drop_type_errors(self):
        """Test TypeError for invalid input types and elements."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            self.df.drop_columns("a")
        
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            self.df.drop_columns(["a", 1])
    
    def test_drop_unknown_column_error(self):
        """Test ValueError for unknown columns (checked before all-columns error)."""
        with pytest.raises(ValueError, match="Unknown column"):
            self.df.drop_columns(["a", "b", "c", "zz"])
    
    def test_drop_attrs_copy_and_no_mutation(self):
        """Test that attrs are deep copied and original is unchanged."""
        frame = ar.ArFrame(self.df._frame, attrs={"meta": {"k": 1}})
        result = frame.drop_columns(["b"])
        assert result._attrs == {"meta": {"k": 1}}
        assert result._attrs is not frame._attrs
        assert result._attrs["meta"] is not frame._attrs["meta"]
        
        original_cols = frame.columns.copy()
        frame.drop_columns(["b"])
        assert frame.columns == original_cols


class TestSelectColumns:
    """Test ArFrame.select_columns method."""
    
    def setup_method(self):
        """Create a test frame with columns a, b, c."""
        self.df = ar.from_pandas(pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.5, 2.5, 3.5],
            "c": ["x", "y", "z"]
        }))
    
    def test_select_basic_operations(self):
        """Test basic select operations: single/multiple columns, tuple."""
        result = self.df.select_columns(["a"])
        assert result.columns == ["a"]
        
        result = self.df.select_columns(["a", "c"])
        assert set(result.columns) == {"a", "c"}
        
        result = self.df.select_columns(("b",))
        assert result.columns == ["b"]
    
    def test_select_type_errors(self):
        """Test TypeError for invalid input types."""
        with pytest.raises(TypeError, match="columns must be a sequence of column names, not a string"):
            self.df.select_columns("a")
        
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            self.df.select_columns({"a"})
    
    def test_select_empty_error(self):
        """Test that empty selection raises ValueError."""
        with pytest.raises(ValueError, match="Column selection cannot be empty"):
            self.df.select_columns([])
    
    def test_select_value_errors(self):
        """Test ValueError for non-string elements, duplicates, unknown columns."""
        with pytest.raises(TypeError, match="All column names must be strings"):
            self.df.select_columns([1, 1])
        
        with pytest.raises(ValueError, match="Duplicate column names are not allowed"):
            self.df.select_columns(["a", "a"])
        
        # Duplicates error comes before unknown error
        with pytest.raises(ValueError, match="Duplicate column names are not allowed"):
            self.df.select_columns(["zz", "zz"])
        
        with pytest.raises(ValueError, match="Unknown columns"):
            self.df.select_columns(["x", "a", "y"])
    
    def test_select_attrs_copy_and_no_mutation(self):
        """Test that attrs are deep copied and original is unchanged."""
        frame = ar.ArFrame(self.df._frame, attrs={"meta": {"k": [1]}})
        result = frame.select_columns(["a"])
        assert result._attrs == {"meta": {"k": [1]}}
        assert result._attrs is not frame._attrs
        assert result._attrs["meta"] is not frame._attrs["meta"]
        
        original_cols = frame.columns.copy()
        frame.select_columns(["a", "c"])
        assert frame.columns == original_cols
