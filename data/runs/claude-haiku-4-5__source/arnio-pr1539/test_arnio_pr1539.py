import pytest
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns method"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        data = {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.1, 2.2, 3.3],
            "col4": [True, False, True],
        }
        return ar.from_dict(data)
    
    def test_drop_single_column_with_list(self, sample_frame):
        """Test dropping a single column with a list"""
        result = sample_frame.drop_columns(["col1"])
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col2", "col3", "col4"]
    
    def test_drop_single_column_with_tuple(self, sample_frame):
        """Test dropping a single column with a tuple"""
        result = sample_frame.drop_columns(("col1",))
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col2", "col3", "col4"]
    
    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns"""
        result = sample_frame.drop_columns(["col1", "col3"])
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col2", "col4"]
    
    def test_column_order_preserved(self, sample_frame):
        """Test that original column order is preserved"""
        result = sample_frame.drop_columns(["col2"])
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col1", "col3", "col4"]
    
    def test_duplicate_columns_silently_ignored(self, sample_frame):
        """Test that duplicates in input are silently ignored"""
        result = sample_frame.drop_columns(["col1", "col1", "col2"])
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col3", "col4"]
    
    def test_empty_list_returns_unchanged_copy(self, sample_frame):
        """Test that empty list returns a copy with all columns"""
        result = sample_frame.drop_columns([])
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(sample_frame)
        assert list(result_df.columns) == list(original_df.columns)
        assert result_df.shape == original_df.shape
    
    def test_returns_new_instance(self, sample_frame):
        """Test that drop_columns returns a new ArFrame instance"""
        result = sample_frame.drop_columns(["col1"])
        assert result is not sample_frame
        # Verify original frame is unchanged
        original_df = ar.to_pandas(sample_frame)
        assert list(original_df.columns) == ["col1", "col2", "col3", "col4"]
    
    def test_data_integrity(self, sample_frame):
        """Test that remaining data is intact after dropping columns"""
        result = sample_frame.drop_columns(["col2", "col4"])
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(sample_frame)
        
        assert result_df["col1"].tolist() == original_df["col1"].tolist()
        assert result_df["col3"].tolist() == original_df["col3"].tolist()
    
    def test_invalid_column_name_raises_valueerror(self, sample_frame):
        """Test that invalid column name raises ValueError"""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["nonexistent"])
    
    def test_mixed_valid_invalid_columns_raises_valueerror(self, sample_frame):
        """Test that mix of valid and invalid columns raises ValueError"""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["col1", "nonexistent"])
    
    def test_non_list_tuple_raises_typeerror(self, sample_frame):
        """Test that non-list/tuple input raises TypeError"""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns("col1")
        
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns({"col1"})
    
    def test_non_string_elements_raises_typeerror(self, sample_frame):
        """Test that non-string elements in cols raises TypeError"""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns([1, 2])
        
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(["col1", 2])
    
    def test_dropping_all_columns_raises_valueerror(self, sample_frame):
        """Test that attempting to drop all columns raises ValueError"""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(["col1", "col2", "col3", "col4"])
    
    def test_drop_all_but_one_column(self, sample_frame):
        """Test dropping all but one column succeeds"""
        result = sample_frame.drop_columns(["col1", "col2", "col3"])
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col4"]
        assert len(result_df) == 3
