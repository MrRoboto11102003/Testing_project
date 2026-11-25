import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame with mixed types."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'score': [95.5, 87.3, 92.1],
        'active': [True, False, True],
    })
    return ar.from_pandas(df)


class TestDropColumns:
    def test_drop_single_and_multiple_columns(self, sample_frame):
        """Drop single column and multiple columns."""
        result = sample_frame.drop_columns(['name'])
        assert 'name' not in result.columns
        assert len(result.columns) == 3
        
        result = sample_frame.drop_columns(['name', 'active'])
        assert result.columns == ['id', 'score']

    def test_drop_with_empty_list_and_duplicates(self, sample_frame):
        """Empty list returns unchanged copy; duplicates ignored."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        
        result = sample_frame.drop_columns(['name', 'name'])
        assert result.columns == ['id', 'score', 'active']

    def test_drop_tuple_input_and_error_cases(self, sample_frame):
        """Tuple input accepted; TypeError and ValueError raised appropriately."""
        result = sample_frame.drop_columns(('name',))
        assert 'name' not in result.columns
        
        with pytest.raises(ValueError):
            sample_frame.drop_columns(['nonexistent'])
        
        with pytest.raises(TypeError):
            sample_frame.drop_columns('name')
        
        with pytest.raises(TypeError):
            sample_frame.drop_columns(['name', 1])


class TestSelectColumns:
    def test_select_single_and_multiple_columns(self, sample_frame):
        """Select single and multiple columns."""
        result = sample_frame.select_columns(['id'])
        assert result.columns == ['id']
        
        result = sample_frame.select_columns(['score', 'id', 'name'])
        assert result.columns == ['score', 'id', 'name']

    def test_select_order_preservation_and_error_cases(self, sample_frame):
        """Column order is preserved in selection; errors on empty/duplicates/unknown."""
        result = sample_frame.select_columns(['active', 'id', 'score'])
        assert result.columns == ['active', 'id', 'score']
        
        with pytest.raises(ValueError):
            sample_frame.select_columns([])
        
        with pytest.raises(ValueError):
            sample_frame.select_columns(['id', 'id'])
        
        with pytest.raises(ValueError):
            sample_frame.select_columns(['nonexistent'])
        
        with pytest.raises(TypeError):
            sample_frame.select_columns('id')
        
        with pytest.raises(TypeError):
            sample_frame.select_columns(['id', 123])


class TestSelectDtypes:
    def test_include_single_and_multiple_dtypes(self, sample_frame):
        """Include single and multiple dtypes."""
        result = sample_frame.select_dtypes(include='int64')
        assert result.columns == ['id']
        
        result = sample_frame.select_dtypes(include=['int64', 'float64'])
        assert set(result.columns) == {'id', 'score'}

    def test_exclude_single_and_multiple_dtypes(self, sample_frame):
        """Exclude single and multiple dtypes."""
        result = sample_frame.select_dtypes(exclude='bool')
        assert 'active' not in result.columns
        
        result = sample_frame.select_dtypes(exclude=['string', 'bool'])
        assert set(result.columns) == {'id', 'score'}

    def test_include_and_exclude_together(self, sample_frame):
        """Include and exclude work together when non-overlapping."""
        result = sample_frame.select_dtypes(
            include=['int64', 'float64'],
            exclude='bool'
        )
        assert set(result.columns) == {'id', 'score'}

    def test_select_dtypes_error_cases(self, sample_frame):
        """Error cases: missing params, overlap, invalid dtype, non-string elements."""
        with pytest.raises(ValueError):
            sample_frame.select_dtypes()
        
        with pytest.raises(ValueError):
            sample_frame.select_dtypes(include='int64', exclude='int64')
        
        with pytest.raises(ValueError):
            sample_frame.select_dtypes(include='invalid_dtype')
        
        with pytest.raises(TypeError):
            sample_frame.select_dtypes(include=['int64', 123])
