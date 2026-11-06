import pytest
import arnio as ar
import pandas as pd


class TestArFrameSelectDtypes:
    """Test suite for ArFrame.select_dtypes()"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with various dtypes."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [1.5, 2.5, 3.5],
            'name': ['apple', 'banana', 'cherry'],
            'active': [True, False, True],
        })
        return ar.from_pandas(df)
    
    # Basic include tests
    
    def test_include_int64(self, sample_frame):
        """Include a single int64 column."""
        result = sample_frame.select_dtypes(include="int64")
        assert result.columns == ['id']
    
    def test_include_float64(self, sample_frame):
        """Include a single float64 column."""
        result = sample_frame.select_dtypes(include="float64")
        assert result.columns == ['price']
    
    def test_include_string(self, sample_frame):
        """Include a single string column."""
        result = sample_frame.select_dtypes(include="string")
        assert result.columns == ['name']
    
    def test_include_multiple_dtypes_list(self, sample_frame):
        """Include multiple dtypes as a list."""
        result = sample_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {'id', 'price'}
    
    def test_include_multiple_dtypes_tuple(self, sample_frame):
        """Include multiple dtypes as a tuple."""
        result = sample_frame.select_dtypes(include=("int64", "bool"))
        assert set(result.columns) == {'id', 'active'}
    
    def test_exclude_string(self, sample_frame):
        """Exclude string columns."""
        result = sample_frame.select_dtypes(exclude="string")
        assert 'name' not in result.columns
        assert len(result.columns) == 3
    
    def test_exclude_multiple_dtypes(self, sample_frame):
        """Exclude multiple dtypes."""
        result = sample_frame.select_dtypes(exclude=["string", "bool"])
        assert set(result.columns) == {'id', 'price'}
    
    def test_include_and_exclude_combined(self, sample_frame):
        """Combine include and exclude filters."""
        result = sample_frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude=["string"]
        )
        assert set(result.columns) == {'id', 'price'}
    
    def test_column_order_preserved(self, sample_frame):
        """Column order from original frame is preserved."""
        result = sample_frame.select_dtypes(include=["int64", "float64", "bool"])
        assert result.columns == ['id', 'price', 'active']
    
    def test_returns_arframe_type(self, sample_frame):
        """Result is an ArFrame."""
        result = sample_frame.select_dtypes(include="int64")
        assert isinstance(result, ar.ArFrame)
    
    # Error cases
    
    def test_neither_include_nor_exclude_raises_valueerror(self, sample_frame):
        """ValueError when neither include nor exclude provided."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes()
    
    def test_overlapping_include_exclude_raises_valueerror(self, sample_frame):
        """ValueError when include and exclude dtypes overlap."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes(include="int64", exclude="int64")
    
    def test_unknown_dtype_raises_valueerror(self, sample_frame):
        """ValueError for unrecognized dtype string."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes(include="unknown_dtype")
    
    def test_invalid_include_type_raises_typeerror(self, sample_frame):
        """TypeError when include is not string, list, or tuple."""
        with pytest.raises(TypeError):
            sample_frame.select_dtypes(include=123)
    
    def test_invalid_exclude_type_raises_typeerror(self, sample_frame):
        """TypeError when exclude is not string, list, or tuple."""
        with pytest.raises(TypeError):
            sample_frame.select_dtypes(exclude={})
