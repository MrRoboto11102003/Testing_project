import pytest
import pandas as pd
from pathlib import Path

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestDropColumns:
    """End-to-end tests for ArFrame.drop_columns()"""
    
    def test_drop_single_column(self, tmp_path):
        """Read CSV, drop one column, verify result and write back."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.drop_columns(["col1"])
        
        assert list(result.columns) == ["col2", "col3"]
        assert result.shape[0] == 2
        
        output_file = tmp_path / "output.csv"
        result.to_csv(str(output_file))
        df = pd.read_csv(output_file)
        assert list(df.columns) == ["col2", "col3"]
    
    def test_drop_multiple_columns_preserves_order(self, tmp_path):
        """Drop multiple columns while preserving order of remaining."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d,e\n1,2,3,4,5\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.drop_columns(["b", "d"])
        
        assert list(result.columns) == ["a", "c", "e"]
    
    def test_drop_columns_with_duplicates_silently_ignored(self, tmp_path):
        """Duplicates in cols list should be silently ignored."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.drop_columns(["x", "x", "y"])
        
        assert list(result.columns) == ["z"]
    
    def test_drop_columns_empty_list_returns_copy(self, tmp_path):
        """Empty list should return unchanged copy."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.drop_columns([])
        
        assert list(result.columns) == ["col1", "col2"]
        assert result.shape == frame.shape
        assert result is not frame
    
    def test_drop_columns_tuple_input(self, tmp_path):
        """Tuple should work in addition to list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.drop_columns(("a", "c"))
        
        assert list(result.columns) == ["b"]
    
    def test_drop_columns_type_error_invalid_container(self, tmp_path):
        """TypeError when cols is not list or tuple."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            frame.drop_columns("col1")
    
    def test_drop_columns_type_error_non_string_elements(self, tmp_path):
        """TypeError when cols contains non-string elements."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            frame.drop_columns(["col1", 2])
    
    def test_drop_columns_value_error_unknown_column(self, tmp_path):
        """ValueError when column doesn't exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["nonexistent"])
    
    def test_drop_columns_value_error_drop_all(self, tmp_path):
        """ValueError when trying to drop all columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1\n1\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(["col1"])


class TestSelectColumns:
    """End-to-end tests for ArFrame.select_columns()"""
    
    def test_select_single_column(self, tmp_path):
        """Read CSV, select one column, verify result and write back."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_columns(["col2"])
        
        assert list(result.columns) == ["col2"]
        assert result.shape[0] == 2
        
        output_file = tmp_path / "output.csv"
        result.to_csv(str(output_file))
        df = pd.read_csv(output_file)
        assert list(df.columns) == ["col2"]
    
    def test_select_multiple_columns_preserves_order(self, tmp_path):
        """Selection should preserve the order specified, not original."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_columns(["d", "b", "a"])
        
        assert list(result.columns) == ["d", "b", "a"]
    
    def test_select_columns_type_error_string(self, tmp_path):
        """TypeError when columns is a string instead of sequence."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a sequence"):
            frame.select_columns("col1")
    
    def test_select_columns_type_error_invalid_container(self, tmp_path):
        """TypeError when columns is not list or tuple."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a list or tuple"):
            frame.select_columns({"col1"})
    
    def test_select_columns_value_error_empty(self, tmp_path):
        """ValueError when selection is empty."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="cannot be empty"):
            frame.select_columns([])
    
    def test_select_columns_value_error_duplicates(self, tmp_path):
        """ValueError when columns contains duplicates."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Duplicate"):
            frame.select_columns(["col1", "col2", "col1"])
    
    def test_select_columns_value_error_unknown_columns(self, tmp_path):
        """ValueError when columns include unknown names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unknown columns"):
            frame.select_columns(["col1", "unknown"])
