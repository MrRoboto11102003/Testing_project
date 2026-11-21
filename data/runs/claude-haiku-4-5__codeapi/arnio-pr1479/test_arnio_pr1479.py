import pytest
import json
import pandas as pd
import arnio as ar


class TestDataQualityReportToDict:
    """Test suite for DataQualityReport.to_dict method."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a simple ArFrame for testing."""
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 92.0, 78.5, 88.0, 95.5],
            'active': [True, False, True, True, False]
        }
        return ar.from_dict(data)
    
    @pytest.fixture
    def sample_report(self, sample_frame):
        """Create a data quality report from sample frame."""
        return ar.profile(sample_frame)
    
    def test_to_dict_basic_call_no_args(self, sample_report):
        """Test basic to_dict() call with no arguments."""
        result = sample_report.to_dict()
        assert isinstance(result, dict)
        assert 'row_count' in result
        assert 'column_count' in result
        assert 'memory_usage' in result
        assert 'duplicate_rows' in result
        assert 'duplicate_ratio' in result
        assert 'quality_score' in result
        assert 'score_components' in result
        assert 'columns' in result
        assert 'suggestions' in result
    
    def test_to_dict_preserves_report_metrics(self, sample_report):
        """Test that returned dict preserves all top-level report metrics."""
        result = sample_report.to_dict()
        assert result['row_count'] == sample_report.row_count
        assert result['column_count'] == sample_report.column_count
        assert result['memory_usage'] == sample_report.memory_usage
        assert result['duplicate_rows'] == sample_report.duplicate_rows
        assert result['duplicate_ratio'] == sample_report.duplicate_ratio
        assert result['quality_score'] == sample_report.quality_score
    
    def test_to_dict_all_columns_included_by_default(self, sample_report):
        """Test that all columns are included when exclude_columns is not specified."""
        result = sample_report.to_dict()
        assert len(result['columns']) == sample_report.column_count
        for col_name in sample_report.columns:
            assert col_name in result['columns']
    
    def test_to_dict_exclude_columns_as_list(self, sample_report):
        """Test exclude_columns parameter with list type."""
        result = sample_report.to_dict(exclude_columns=['id', 'name'])
        assert 'id' not in result['columns']
        assert 'name' not in result['columns']
        assert len(result['columns']) == sample_report.column_count - 2
    
    def test_to_dict_exclude_columns_as_set(self, sample_report):
        """Test exclude_columns parameter with set type."""
        result = sample_report.to_dict(exclude_columns={'id', 'score'})
        assert 'id' not in result['columns']
        assert 'score' not in result['columns']
        assert len(result['columns']) == sample_report.column_count - 2
    
    def test_to_dict_exclude_columns_as_tuple(self, sample_report):
        """Test exclude_columns parameter with tuple type."""
        result = sample_report.to_dict(exclude_columns=('name', 'active'))
        assert 'name' not in result['columns']
        assert 'active' not in result['columns']
        assert len(result['columns']) == sample_report.column_count - 2
    
    def test_to_dict_exclude_columns_empty_list(self, sample_report):
        """Test exclude_columns with empty list includes all columns."""
        result = sample_report.to_dict(exclude_columns=[])
        assert len(result['columns']) == sample_report.column_count
    
    def test_to_dict_exclude_nonexistent_column_raises_keyerror(self, sample_report):
        """Test that excluding non-existent column raises KeyError."""
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            sample_report.to_dict(exclude_columns=['nonexistent_column'])
    
    def test_to_dict_exclude_columns_wrong_type_raises_typeerror(self, sample_report):
        """Test that invalid exclude_columns type raises TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must be"):
            sample_report.to_dict(exclude_columns="id")
    
    def test_to_dict_exclude_columns_with_non_string_raises_typeerror(self, sample_report):
        """Test that non-string items in exclude_columns raise TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must contain only string"):
            sample_report.to_dict(exclude_columns=['id', 123, 'name'])
    
    def test_to_dict_redact_sample_values_true(self, sample_report):
        """Test redact_sample_values=True parameter."""
        result = sample_report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        assert len(result['columns']) > 0
    
    def test_to_dict_redact_sample_values_false(self, sample_report):
        """Test redact_sample_values=False parameter."""
        result = sample_report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
        assert len(result['columns']) > 0
    
    def test_to_dict_columns_sorted_alphabetically(self, sample_report):
        """Test that column names in output are sorted."""
        result = sample_report.to_dict()
        column_names = list(result['columns'].keys())
        assert column_names == sorted(column_names)
    
    def test_to_dict_combined_exclude_and_redact(self, sample_report):
        """Test using both exclude_columns and redact_sample_values together."""
        result = sample_report.to_dict(
            exclude_columns=['id', 'name'],
            redact_sample_values=True
        )
        assert 'id' not in result['columns']
        assert 'name' not in result['columns']
        assert 'score' in result['columns']
        assert 'active' in result['columns']
        assert len(result['columns']) == sample_report.column_count - 2
    
    def test_to_dict_exclude_all_columns(self, sample_report):
        """Test excluding all columns."""
        all_columns = list(sample_report.columns.keys())
        result = sample_report.to_dict(exclude_columns=all_columns)
        assert len(result['columns']) == 0
        # Top-level metrics should still be present
        assert result['row_count'] == sample_report.row_count
        assert result['column_count'] == sample_report.column_count
    
    def test_to_dict_returns_json_serializable(self, sample_report):
        """Test that returned dict is JSON-serializable."""
        result = sample_report.to_dict()
        # Should not raise an exception
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_to_dict_suggestions_structure(self, sample_report):
        """Test that suggestions have proper structure."""
        result = sample_report.to_dict()
        suggestions = result['suggestions']
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert 'step' in suggestion
            assert 'kwargs' in suggestion
            assert 'confidence_score' in suggestion
            assert 'confidence_reason' in suggestion
