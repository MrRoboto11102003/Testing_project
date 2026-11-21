import pytest
import arnio as ar
import pandas as pd
import tempfile
import os


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "sample.csv")
        df = pd.DataFrame({
            "col_int": [1, 2, 3],
            "col_float": [1.5, 2.5, 3.5],
            "col_str": ["a", "b", "c"],
        })
        df.to_csv(csv_path, index=False)
        frame = ar.read_csv(csv_path)
        yield frame


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""
    
    def test_drop_single_column_list(self, sample_frame):
        """Drop single column with list input."""
        result = sample_frame.drop_columns(["col_int"])
        assert "col_int" not in result.columns
        assert "col_float" in result.columns
    
    def test_drop_multiple_columns_tuple(self, sample_frame):
        """Drop multiple columns with tuple input."""
        result = sample_frame.drop_columns(("col_int", "col_float"))
        assert len(result.columns) == 1
        assert "col_str" in result.columns
    
    def test_drop_duplicates_silently_ignored(self, sample_frame):
        """Duplicates are silently ignored."""
        result = sample_frame.drop_columns(["col_int", "col_int"])
        assert len(result.columns) == 2
    
    def test_drop_empty_returns_unchanged_copy(self, sample_frame):
        """Empty list returns unchanged copy."""
        result = sample_frame.drop_columns([])
        assert list(result.columns) == list(sample_frame.columns)
    
    def test_drop_all_columns_raises_valueerror(self, sample_frame):
        """Attempting to drop all columns raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(list(sample_frame.columns))
    
    def test_drop_nonexistent_column_raises_valueerror(self, sample_frame):
        """Nonexistent column raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["nonexistent"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""
    
    def test_select_preserves_specified_order(self, sample_frame):
        """Selected columns are in specified order."""
        result = sample_frame.select_columns(["col_str", "col_int"])
        assert list(result.columns) == ["col_str", "col_int"]
    
    def test_select_empty_list_raises_valueerror(self, sample_frame):
        """Empty selection raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns([])
    
    def test_select_duplicate_columns_raise_valueerror(self, sample_frame):
        """Duplicate column names raise ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["col_int", "col_int"])
    
    def test_select_nonexistent_column_raises_valueerror(self, sample_frame):
        """Nonexistent column raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["nonexistent"])
    
    def test_select_string_input_raises_typeerror(self, sample_frame):
        """String input raises TypeError."""
        with pytest.raises(TypeError):
            sample_frame.select_columns("col_int")


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes()"""
    
    def test_select_dtypes_include_single(self, sample_frame):
        """Include single dtype as string."""
        result = sample_frame.select_dtypes(include="int64")
        assert "col_int" in result.columns
    
    def test_select_dtypes_exclude_single(self, sample_frame):
        """Exclude single dtype as string."""
        result = sample_frame.select_dtypes(exclude="string")
        assert "col_str" not in result.columns
    
    def test_select_dtypes_neither_raises_valueerror(self, sample_frame):
        """Neither include nor exclude raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes()
    
    def test_select_dtypes_overlapping_raises_valueerror(self, sample_frame):
        """Overlapping include and exclude raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes(include="int64", exclude="int64")
