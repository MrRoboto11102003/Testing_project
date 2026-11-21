import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file with various data quality issues."""
    data = {
        'id': [1, 2, 3, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David'],
        'age': [25, 30, None, None, 35],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'charlie@example.com', 'david@example.com'],
        'salary': [50000, 60000, 55000, 55000, 65000]
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "sample.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def quality_report(sample_csv):
    """Generate a quality report from sample CSV."""
    df = pd.read_csv(sample_csv)
    return ar.quality(df)


class TestDataQualityReportToDict:
    """Test suite for DataQualityReport.to_dict method."""

    def test_to_dict_basic_returns_dict(self, quality_report):
        """Test that to_dict returns a dictionary with expected top-level keys."""
        result = quality_report.to_dict()
        
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

    def test_to_dict_is_json_serializable(self, quality_report):
        """Test that to_dict output is fully JSON-serializable."""
        result = quality_report.to_dict()
        
        # Should not raise exception when converting to JSON
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        
        # Should deserialize back to dict
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)

    def test_to_dict_redact_sample_values_false(self, quality_report):
        """Test to_dict with redact_sample_values=False (default behavior)."""
        result = quality_report.to_dict(redact_sample_values=False)
        
        assert isinstance(result, dict)
        assert 'columns' in result
        assert isinstance(result['columns'], dict)

    def test_to_dict_redact_sample_values_true(self, quality_report):
        """Test to_dict with redact_sample_values=True."""
        result = quality_report.to_dict(redact_sample_values=True)
        
        assert isinstance(result, dict)
        assert 'columns' in result
        # Should still be valid structure even with redaction
        assert isinstance(result['columns'], dict)

    def test_exclude_columns_as_list(self, quality_report):
        """Test exclude_columns parameter as a list."""
        result = quality_report.to_dict(exclude_columns=['age', 'email'])
        
        assert 'age' not in result['columns']
        assert 'email' not in result['columns']
        assert 'id' in result['columns']
        assert 'name' in result['columns']
        assert 'salary' in result['columns']

    def test_exclude_columns_as_tuple(self, quality_report):
        """Test exclude_columns parameter as a tuple."""
        result = quality_report.to_dict(exclude_columns=('age', 'email'))
        
        assert 'age' not in result['columns']
        assert 'email' not in result['columns']
        assert 'name' in result['columns']

    def test_exclude_columns_as_set(self, quality_report):
        """Test exclude_columns parameter as a set."""
        result = quality_report.to_dict(exclude_columns={'age', 'email'})
        
        assert 'age' not in result['columns']
        assert 'email' not in result['columns']
        assert 'id' in result['columns']

    def test_exclude_columns_empty_collection(self, quality_report):
        """Test exclude_columns with empty list includes all columns."""
        result = quality_report.to_dict(exclude_columns=[])
        
        assert len(result['columns']) == 5

    def test_exclude_columns_invalid_type_string_raises_typeerror(self, quality_report):
        """Test that string exclude_columns raises TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            quality_report.to_dict(exclude_columns="age")

    def test_exclude_columns_invalid_type_dict_raises_typeerror(self, quality_report):
        """Test that dict exclude_columns raises TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            quality_report.to_dict(exclude_columns={'age': 1})

    def test_exclude_columns_non_string_element_raises_typeerror(self, quality_report):
        """Test that non-string column names raise TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            quality_report.to_dict(exclude_columns=['age', 123])

    def test_exclude_columns_unknown_column_raises_keyerror(self, quality_report):
        """Test that unknown exclude_columns raise KeyError with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            quality_report.to_dict(exclude_columns=['nonexistent'])
        
        error_msg = str(exc_info.value)
        assert 'Unknown exclude_columns' in error_msg
        assert 'nonexistent' in error_msg

    def test_to_dict_data_values_present(self, quality_report):
        """Test that to_dict includes expected data quality metrics."""
        result = quality_report.to_dict()
        
        assert isinstance(result['row_count'], int)
        assert result['row_count'] == 5
        assert isinstance(result['column_count'], int)
        assert result['column_count'] == 5
        assert isinstance(result['memory_usage'], (int, float))
        assert isinstance(result['duplicate_rows'], int)
        assert isinstance(result['duplicate_ratio'], (int, float))

    def test_to_dict_quality_metrics_present(self, quality_report):
        """Test that to_dict includes quality score and components."""
        result = quality_report.to_dict()
        
        assert 'quality_score' in result
        assert 'score_components' in result
        assert isinstance(result['quality_score'], (int, float))
        assert isinstance(result['score_components'], dict)

    def test_to_dict_suggestions_format(self, quality_report):
        """Test that suggestions have correct format with all required fields."""
        result = quality_report.to_dict()
        
        assert isinstance(result['suggestions'], list)
        for suggestion in result['suggestions']:
            assert isinstance(suggestion, dict)
            assert 'step' in suggestion
            assert 'kwargs' in suggestion
            assert 'confidence_score' in suggestion
            assert 'confidence_reason' in suggestion

    def test_to_dict_with_multiple_exclude_and_redact(self, quality_report):
        """Test combining exclude_columns and redact_sample_values parameters."""
        result = quality_report.to_dict(
            exclude_columns=['age', 'email'],
            redact_sample_values=True
        )
        
        assert 'age' not in result['columns']
        assert 'email' not in result['columns']
        assert 'name' in result['columns']
        assert isinstance(result, dict)
