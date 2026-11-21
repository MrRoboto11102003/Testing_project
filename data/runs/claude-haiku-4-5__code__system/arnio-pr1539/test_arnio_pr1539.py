import pytest
import csv
import arnio as ar


class TestDropColumns:
    """End-to-end tests for arnio.frame.ArFrame.drop_columns"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing"""
        csv_file = tmp_path / "sample.csv"
        data = [
            ["name", "age", "city", "country"],
            ["Alice", "25", "NYC", "USA"],
            ["Bob", "30", "London", "UK"],
            ["Charlie", "35", "Paris", "France"],
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return csv_file
    
    def test_drop_single_column_returns_arframe(self, sample_csv):
        """Test that drop_columns returns an ArFrame with column removed"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns(["age"])
        
        assert isinstance(result, ar.ArFrame)
        assert "age" not in result.columns
        assert len(result.columns) == 3
        assert list(result.columns) == ["name", "city", "country"]
    
    def test_drop_multiple_columns(self, sample_csv):
        """Test dropping multiple columns preserves order"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns(["age", "country"])
        
        assert list(result.columns) == ["name", "city"]
        assert len(result) == 3
    
    def test_drop_columns_with_tuple(self, sample_csv):
        """Test that tuple argument works"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns(("age", "country"))
        
        assert len(result.columns) == 2
        assert "age" not in result.columns
        assert "country" not in result.columns
    
    def test_drop_columns_duplicates_ignored(self, sample_csv):
        """Test that duplicates in drop list are silently ignored"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns(["age", "age", "city", "age"])
        
        assert len(result.columns) == 2
        assert list(result.columns) == ["name", "country"]
    
    def test_drop_columns_empty_list_returns_copy(self, sample_csv):
        """Test that empty list returns a copy with all columns unchanged"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns([])
        
        assert list(result.columns) == list(frame.columns)
        assert result is not frame
        assert len(result) == len(frame)
    
    def test_drop_columns_nonexistent_raises_valueerror(self, sample_csv):
        """Test ValueError for non-existent column"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(ValueError) as exc_info:
            frame.drop_columns(["nonexistent"])
        
        assert "Unknown column" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
    
    def test_drop_columns_mixed_valid_invalid_raises_valueerror(self, sample_csv):
        """Test ValueError when mix of valid and invalid columns"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(ValueError) as exc_info:
            frame.drop_columns(["age", "badcol"])
        
        assert "Unknown column" in str(exc_info.value)
    
    def test_drop_columns_string_input_raises_typeerror(self, sample_csv):
        """Test TypeError when cols is string instead of list/tuple"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(TypeError) as exc_info:
            frame.drop_columns("age")
        
        assert "must be a list or tuple" in str(exc_info.value)
    
    def test_drop_columns_non_string_elements_raises_typeerror(self, sample_csv):
        """Test TypeError when column names include non-string"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(TypeError) as exc_info:
            frame.drop_columns(["age", 123])
        
        assert "must be strings" in str(exc_info.value)
    
    def test_drop_all_columns_raises_valueerror(self, sample_csv):
        """Test ValueError when attempting to drop all columns"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(ValueError) as exc_info:
            frame.drop_columns(["name", "age", "city", "country"])
        
        assert "cannot remove all columns" in str(exc_info.value)
    
    def test_drop_columns_dict_input_raises_typeerror(self, sample_csv):
        """Test TypeError when cols is a dict"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(TypeError):
            frame.drop_columns({"age": None})
    
    def test_drop_columns_set_input_raises_typeerror(self, sample_csv):
        """Test TypeError when cols is a set"""
        frame = ar.read_csv(str(sample_csv))
        
        with pytest.raises(TypeError):
            frame.drop_columns({"age", "city"})
    
    def test_drop_columns_first_last_middle(self, sample_csv):
        """Test dropping first, last, and middle columns preserves order"""
        frame = ar.read_csv(str(sample_csv))
        result = frame.drop_columns(["name", "age", "country"])
        
        assert list(result.columns) == ["city"]
        assert len(result) == 3
    
    def test_drop_columns_original_frame_unmodified(self, sample_csv):
        """Test that original frame is not modified by drop_columns"""
        frame = ar.read_csv(str(sample_csv))
        original_cols = list(frame.columns)
        
        result = frame.drop_columns(["age", "country"])
        
        assert list(frame.columns) == original_cols
        assert "age" in frame.columns
        assert "country" in frame.columns
        assert len(result) != len(frame.columns)
