import pytest
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    data = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "SF"]
    }
    return ar.from_dict(data)


class TestDropColumns:
    """Tests for ArFrame.drop_columns"""
    
    def test_drop_single_and_multiple(self, sample_frame):
        """Test dropping single and multiple columns."""
        result = sample_frame.drop_columns(["id"])
        assert result.columns == ["name", "age", "city"]
        assert result.shape == (3, 3)
        
        result = sample_frame.drop_columns(["id", "age"])
        assert result.columns == ["name", "city"]
        assert result.shape == (3, 2)
    
    def test_drop_tuple_input_and_empty_list(self, sample_frame):
        """Test tuple input works and empty list returns unchanged copy."""
        result = sample_frame.drop_columns(("id", "name"))
        assert result.columns == ["age", "city"]
        
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape
    
    def test_drop_duplicates_silently_ignored(self, sample_frame):
        """Test that duplicate column names in drop list are ignored."""
        result = sample_frame.drop_columns(["id", "id", "name", "name"])
        assert result.columns == ["age", "city"]
    
    def test_drop_invalid_type_raises_typeerror(self, sample_frame):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns("id")
        with pytest.raises(TypeError):
            sample_frame.drop_columns({"id"})
        with pytest.raises(TypeError):
            sample_frame.drop_columns(["id", 123, "name"])
    
    def test_drop_unknown_column_raises_valueerror(self, sample_frame):
        """Test that unknown column names raise ValueError."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["unknown"])
    
    def test_drop_all_columns_raises_valueerror(self, sample_frame):
        """Test that dropping all columns raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["id", "name", "age", "city"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns"""
    
    def test_select_single_and_multiple(self, sample_frame):
        """Test selecting single and multiple columns."""
        result = sample_frame.select_columns(["id"])
        assert result.columns == ["id"]
        assert result.shape == (3, 1)
        
        result = sample_frame.select_columns(["name", "age"])
        assert result.columns == ["name", "age"]
        assert result.shape == (3, 2)
    
    def test_select_preserves_and_reorders_columns(self, sample_frame):
        """Test that column order matches the selection order."""
        result = sample_frame.select_columns(["city", "id", "name"])
        assert result.columns == ["city", "id", "name"]
    
    def test_select_string_input_raises_typeerror(self, sample_frame):
        """Test that string input raises TypeError."""
        with pytest.raises(TypeError):
            sample_frame.select_columns("id")
        with pytest.raises(TypeError):
            sample_frame.select_columns(["id", 123, "name"])
    
    def test_select_unknown_column_raises_valueerror(self, sample_frame):
        """Test that unknown column raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["id", "unknown"])
    
    def test_select_empty_raises_valueerror(self, sample_frame):
        """Test that empty selection raises ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns([])
    
    def test_select_duplicates_raise_valueerror(self, sample_frame):
        """Test that duplicate column names raise ValueError."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["id", "id", "name"])
