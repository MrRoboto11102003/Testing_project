import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'score': [85.5, 90.0, 88.5, None, 92.0],
        'email': ['alice@example.com', 'bob@test.org', 'charlie@example.com', None, 'eve@test.org'],
    })
    csv_file = tmp_path / "sample.csv"
    df.to_csv(csv_file, index=False)
    return csv_file, df


class TestColumnProfileToDict:
    def test_basic_column_profile_to_dict(self, sample_csv):
        """Test basic ColumnProfile.to_dict() with standard data."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        profile = ar.profile(df)
        
        col_dict = profile.columns['name'].to_dict()
        
        assert isinstance(col_dict, dict)
        assert col_dict['name'] == 'name'
        assert isinstance(col_dict['null_count'], int)
        assert isinstance(col_dict['row_count'], int)
        assert 'sample_values' in col_dict
        assert isinstance(col_dict['sample_values'], list)
        assert 'warnings' in col_dict
        assert 'top_values' in col_dict
    
    def test_column_profile_to_dict_with_redaction(self, sample_csv):
        """Test ColumnProfile.to_dict() with redact_sample_values=True."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        profile = ar.profile(df)
        
        col_dict = profile.columns['email'].to_dict(redact_sample_values=True)
        
        for value in col_dict['sample_values']:
            if value is not None:
                assert value == '[REDACTED]'
    
    def test_numeric_column_includes_quantiles(self, sample_csv):
        """Test that numeric columns include quantile statistics in to_dict()."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        profile = ar.profile(df)
        
        col_dict = profile.columns['score'].to_dict()
        
        assert 'q25' in col_dict
        assert 'q50' in col_dict
        assert 'q75' in col_dict
        assert 'q95' in col_dict
        assert 'min' in col_dict
        assert 'max' in col_dict
        assert 'iqr' in col_dict
        assert 'outlier_count' in col_dict
        assert 'outlier_ratio' in col_dict


class TestDataQualityReportScoreBreakdown:
    def test_score_breakdown_returns_all_penalties(self, sample_csv):
        """Test that score_breakdown() returns all expected penalty keys."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        assert 'null_penalty' in breakdown
        assert 'duplicate_penalty' in breakdown
        assert 'type_mismatch_penalty' in breakdown
        assert 'final_score' in breakdown
        
        for key, value in breakdown.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric"
    
    def test_score_breakdown_final_score_matches_report(self, sample_csv):
        """Test that final_score from breakdown matches report.quality_score."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        breakdown = report.score_breakdown()
        assert breakdown['final_score'] == report.quality_score


class TestDataQualityReportToDict:
    def test_to_dict_basic_structure(self, sample_csv):
        """Test basic DataQualityReport.to_dict() returns expected structure."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert 'row_count' in report_dict
        assert 'column_count' in report_dict
        assert 'quality_score' in report_dict
        assert 'columns' in report_dict
        assert 'suggestions' in report_dict
        assert report_dict['row_count'] == 5
        assert report_dict['column_count'] == 4
    
    def test_to_dict_with_redact_sample_values(self, sample_csv):
        """Test to_dict() with redact_sample_values=True."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        report_dict = report.to_dict(redact_sample_values=True)
        
        for col_name, col_data in report_dict['columns'].items():
            for value in col_data['sample_values']:
                if value is not None:
                    assert value == '[REDACTED]'
    
    def test_to_dict_exclude_columns_list(self, sample_csv):
        """Test to_dict() with exclude_columns as list."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        report_dict = report.to_dict(exclude_columns=['name', 'email'])
        
        assert 'id' in report_dict['columns']
        assert 'score' in report_dict['columns']
        assert 'name' not in report_dict['columns']
        assert 'email' not in report_dict['columns']
    
    def test_to_dict_exclude_columns_set(self, sample_csv):
        """Test to_dict() with exclude_columns as set."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        report_dict = report.to_dict(exclude_columns={'name'})
        
        assert 'name' not in report_dict['columns']
        assert 'id' in report_dict['columns']
    
    def test_to_dict_exclude_columns_tuple(self, sample_csv):
        """Test to_dict() with exclude_columns as tuple."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        report_dict = report.to_dict(exclude_columns=('score', 'email'))
        
        assert 'score' not in report_dict['columns']
        assert 'email' not in report_dict['columns']
        assert 'id' in report_dict['columns']
        assert 'name' in report_dict['columns']
    
    def test_to_dict_exclude_unknown_column_raises_keyerror(self, sample_csv):
        """Test that exclude_columns with unknown column raises KeyError."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=['nonexistent_column'])
    
    def test_to_dict_bad_exclude_columns_type_raises_typeerror(self, sample_csv):
        """Test that invalid exclude_columns type raises TypeError."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns={'not': 'list'})
    
    def test_to_dict_non_string_in_exclude_columns_raises_typeerror(self, sample_csv):
        """Test that non-string values in exclude_columns raises TypeError."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        with pytest.raises(TypeError, match="must contain only string"):
            report.to_dict(exclude_columns=['name', 123])


class TestDataQualityReportToJson:
    def test_to_json_returns_valid_json_string(self, sample_csv):
        """Test that to_json() returns valid JSON string."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        json_str = report.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert 'row_count' in parsed
        assert 'columns' in parsed
    
    def test_to_json_with_indent(self, sample_csv):
        """Test to_json() with indent parameter creates formatted output."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        json_str = report.to_json(indent=2)
        
        assert '\n' in json_str
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_to_json_write_to_file_returns_none(self, sample_csv, tmp_path):
        """Test to_json() with output parameter writes to file and returns None."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        output_file = tmp_path / "report.json"
        with open(output_file, 'w') as f:
            result = report.to_json(output=f, indent=2)
        
        assert result is None
        assert output_file.exists()
        
        with open(output_file) as f:
            parsed = json.load(f)
        assert parsed['row_count'] == 5
        assert parsed['column_count'] == 4
    
    def test_to_json_invalid_output_raises_typeerror(self, sample_csv):
        """Test that invalid output parameter raises TypeError."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        with pytest.raises(TypeError, match="writable text stream"):
            report.to_json(output="not_a_file")
    
    def test_to_json_with_redaction_and_exclusion(self, sample_csv, tmp_path):
        """Test to_json() with both redact_sample_values and exclude_columns."""
        csv_file, _ = sample_csv
        df = pd.read_csv(csv_file)
        report = ar.summarize(df)
        
        output_file = tmp_path / "redacted_report.json"
        with open(output_file, 'w') as f:
            report.to_json(
                output=f,
                indent=2,
                redact_sample_values=True,
                exclude_columns=['email'],
            )
        
        with open(output_file) as f:
            parsed = json.load(f)
        
        assert 'email' not in parsed['columns']
        for col_data in parsed['columns'].values():
            for val in col_data['sample_values']:
                if val is not None:
                    assert val == '[REDACTED]'


class TestEdgeCasesAndIntegration:
    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'col1': [], 'col2': []})
        csv_file = tmp_path / "empty.csv"
        df.to_csv(csv_file, index=False)
        df_loaded = pd.read_csv(csv_file)
        
        report = ar.summarize(df_loaded)
        report_dict = report.to_dict()
        
        assert report_dict['row_count'] == 0
        assert report_dict['column_count'] == 2
        assert isinstance(report_dict['columns'], dict)
    
    def test_all_null_column_profiling(self, tmp_path):
        """Test profiling of column with all null values."""
        df = pd.DataFrame({
            'all_null': [None, None, None],
            'has_data': [1, 2, 3],
        })
        csv_file = tmp_path / "null_column.csv"
        df.to_csv(csv_file, index=False)
        df_loaded = pd.read_csv(csv_file)
        
        profile = ar.profile(df_loaded)
        col_dict = profile.columns['all_null'].to_dict()
        
        assert col_dict['null_count'] == 3
        assert col_dict['null_ratio'] == 1.0
        
        json_output = json.dumps(col_dict)
        assert isinstance(json_output, str)
    
    def test_complete_workflow_with_file_output(self, tmp_path):
        """Test complete end-to-end workflow: load, profile, generate report, save to file."""
        df = pd.DataFrame({
            'value': range(1, 11),
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'amount': [10.5, 20.3, None, 15.7, 25.0, 30.2, None, 18.5, 22.1, 19.8],
        })
        
        input_csv = tmp_path / "input.csv"
        df.to_csv(input_csv, index=False)
        
        df_loaded = pd.read_csv(input_csv)
        report = ar.summarize(df_loaded)
        
        breakdown = report.score_breakdown()
        assert all(isinstance(v, (int, float)) for v in breakdown.values())
        
        report_json_file = tmp_path / "report.json"
        with open(report_json_file, 'w') as f:
            report.to_json(
                output=f,
                indent=2,
                redact_sample_values=False,
                exclude_columns=['category']
            )
        
        with open(report_json_file) as f:
            json_data = json.load(f)
        
        assert 'category' not in json_data['columns']
        assert 'value' in json_data['columns']
        assert 'amount' in json_data['columns']
        assert json_data['row_count'] == 10
