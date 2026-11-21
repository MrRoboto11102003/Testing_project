import pytest
import pandas as pd
import arnio as ar


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict method"""
    
    @pytest.fixture
    def report(self):
        """Create a DataQualityReport from sample data"""
        df = pd.DataFrame({
            'col_a': [1, 2, 3, 1],
            'col_b': ['x', 'y', 'z', 'x'],
            'col_c': [1.1, 2.2, 3.3, 1.1]
        })
        return ar.quality.DataQualityReport.from_dataframe(df)
    
    def test_to_dict_basic_structure(self, report):
        """Test that to_dict returns a dict with required keys"""
        result = report.to_dict()
        assert isinstance(result, dict)
        required_keys = {
            'row_count', 'column_count', 'memory_usage',
            'duplicate_rows', 'duplicate_ratio', 'quality_score',
            'score_components', 'columns', 'suggestions'
        }
        assert required_keys.issubset(set(result.keys()))
    
    def test_to_dict_redact_sample_values_false(self, report):
        """Test redact_sample_values=False is accepted"""
        result = report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values_true(self, report):
        """Test redact_sample_values=True is accepted"""
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values_default(self, report):
        """Test redact_sample_values defaults correctly"""
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_none(self, report):
        """Test exclude_columns=None includes all columns"""
        result = report.to_dict(exclude_columns=None)
        assert len(result['columns']) == 3
        assert all(col in result['columns'] for col in ['col_a', 'col_b', 'col_c'])
    
    def test_to_dict_exclude_columns_list(self, report):
        """Test exclude_columns as list"""
        result = report.to_dict(exclude_columns=['col_a'])
        assert 'col_a' not in result['columns']
        assert 'col_b' in result['columns']
        assert 'col_c' in result['columns']
    
    def test_to_dict_exclude_columns_set(self, report):
        """Test exclude_columns as set"""
        result = report.to_dict(exclude_columns={'col_b'})
        assert 'col_b' not in result['columns']
        assert 'col_a' in result['columns']
        assert 'col_c' in result['columns']
    
    def test_to_dict_exclude_columns_tuple(self, report):
        """Test exclude_columns as tuple"""
        result = report.to_dict(exclude_columns=('col_c',))
        assert 'col_c' not in result['columns']
        assert 'col_a' in result['columns']
        assert 'col_b' in result['columns']
    
    def test_to_dict_exclude_columns_multiple(self, report):
        """Test excluding multiple columns"""
        result = report.to_dict(exclude_columns=['col_a', 'col_b'])
        assert 'col_a' not in result['columns']
        assert 'col_b' not in result['columns']
        assert 'col_c' in result['columns']
        assert len(result['columns']) == 1
    
    def test_to_dict_exclude_columns_empty(self, report):
        """Test empty exclude_columns includes all columns"""
        result = report.to_dict(exclude_columns=[])
        assert len(result['columns']) == 3
    
    def test_to_dict_exclude_columns_invalid_type_string(self, report):
        """Test exclude_columns as string raises TypeError"""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns='col_a')
    
    def test_to_dict_exclude_columns_invalid_type_dict(self, report):
        """Test exclude_columns as dict raises TypeError"""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns={'col_a': 'value'})
    
    def test_to_dict_exclude_columns_non_string_element(self, report):
        """Test exclude_columns with non-string element raises TypeError"""
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            report.to_dict(exclude_columns=['col_a', 42])
    
    def test_to_dict_exclude_columns_unknown_column(self, report):
        """Test unknown column name in exclude_columns raises KeyError"""
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=['unknown_column'])
    
    def test_to_dict_exclude_columns_partial_unknown(self, report):
        """Test mix of known and unknown columns raises KeyError"""
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=['col_a', 'nonexistent'])
    
    def test_to_dict_columns_sorted_output(self, report):
        """Test that columns in output are sorted alphabetically"""
        result = report.to_dict()
        column_names = list(result['columns'].keys())
        assert column_names == sorted(column_names)
    
    def test_to_dict_suggestions_structure(self, report):
        """Test suggestions have required structure"""
        result = report.to_dict()
        assert isinstance(result['suggestions'], list)
        for suggestion in result['suggestions']:
            assert isinstance(suggestion, dict)
            assert 'step' in suggestion
            assert 'kwargs' in suggestion
            assert 'confidence_score' in suggestion
            assert 'confidence_reason' in suggestion
    
    def test_to_dict_with_both_parameters(self, report):
        """Test using both parameters together"""
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=['col_a']
        )
        assert 'col_a' not in result['columns']
        assert 'col_b' in result['columns']
        assert 'col_c' in result['columns']
        assert isinstance(result, dict)
