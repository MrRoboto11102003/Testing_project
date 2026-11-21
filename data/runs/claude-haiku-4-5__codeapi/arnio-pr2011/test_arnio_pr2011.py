import pytest
import arnio as ar


class TestDropColumns:
    """Tests for ArFrame.drop_columns method."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with 4 columns."""
        return ar.from_dict({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })
    
    def test_drop_single_column(self, sample_frame):
        """Test dropping a single column."""
        result = sample_frame.drop_columns(['age'])
        assert result.columns == ['id', 'name', 'city']
    
    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns."""
        result = sample_frame.drop_columns(['age', 'city'])
        assert result.columns == ['id', 'name']
    
    def test_drop_preserves_order(self, sample_frame):
        """Test that remaining columns preserve original order."""
        result = sample_frame.drop_columns(['name', 'city'])
        assert result.columns == ['id', 'age']
    
    def test_drop_accepts_tuple(self, sample_frame):
        """Test that drop_columns accepts tuple input."""
        result = sample_frame.drop_columns(('id', 'age'))
        assert result.columns == ['name', 'city']
    
    def test_drop_empty_list(self, sample_frame):
        """Test that empty list returns unchanged copy."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
    
    def test_drop_silently_ignores_duplicates(self, sample_frame):
        """Test that duplicate column names are silently ignored."""
        result = sample_frame.drop_columns(['id', 'id', 'name'])
        assert result.columns == ['age', 'city']
    
    def test_drop_all_columns_raises(self, sample_frame):
        """Test that dropping all columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(['id', 'name', 'age', 'city'])
    
    def test_drop_unknown_column_raises(self, sample_frame):
        """Test that unknown column name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(['nonexistent'])
    
    def test_drop_non_string_raises(self, sample_frame):
        """Test that non-string column names raise TypeError."""
        with pytest.raises(TypeError, match="strings"):
            sample_frame.drop_columns(['id', 123])


class TestSelectColumns:
    """Tests for ArFrame.select_columns method."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with 4 columns."""
        return ar.from_dict({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })
    
    def test_select_single_column(self, sample_frame):
        """Test selecting a single column."""
        result = sample_frame.select_columns(['name'])
        assert result.columns == ['name']
    
    def test_select_preserves_order(self, sample_frame):
        """Test that selected columns appear in specified order."""
        result = sample_frame.select_columns(['city', 'id', 'name'])
        assert result.columns == ['city', 'id', 'name']
    
    def test_select_empty_raises(self, sample_frame):
        """Test that empty selection raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            sample_frame.select_columns([])
    
    def test_select_duplicates_raise(self, sample_frame):
        """Test that duplicate column names raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate"):
            sample_frame.select_columns(['id', 'name', 'id'])
    
    def test_select_unknown_column_raises(self, sample_frame):
        """Test that unknown column name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            sample_frame.select_columns(['id', 'nonexistent'])
    
    def test_select_string_input_raises(self, sample_frame):
        """Test that string input raises TypeError."""
        with pytest.raises(TypeError, match="not a string"):
            sample_frame.select_columns('id')
