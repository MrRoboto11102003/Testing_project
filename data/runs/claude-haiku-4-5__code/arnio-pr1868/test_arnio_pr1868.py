import pytest
import json
import io
import pandas as pd
import numpy as np
import arnio as ar


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict"""

    def test_to_dict_returns_dict_with_expected_fields(self):
        """Test that to_dict returns a JSON-friendly dict with core fields"""
        df = pd.DataFrame({'col': [1, 2, 3, None]})
        report = ar.quality.DataQualityReport(df)
        profile = report.columns['col']
        result = profile.to_dict()

        assert isinstance(result, dict)
        expected_fields = {'name', 'dtype', 'row_count', 'null_count', 
                          'null_ratio', 'sample_values', 'warnings', 'top_values'}
        assert all(field in result for field in expected_fields)

    def test_to_dict_redact_sample_values_true(self):
        """Test redaction when redact_sample_values=True"""
        df = pd.DataFrame({'col': ['secret1', 'secret2', 'secret3']})
        report = ar.quality.DataQualityReport(df)
        profile = report.columns['col']
        result = profile.to_dict(redact_sample_values=True)

        if result['sample_values']:
            assert all(v == '[REDACTED]' for v in result['sample_values'])

    def test_to_dict_numeric_dtype_includes_quantiles(self):
        """Test numeric columns include quantile fields"""
        df = pd.DataFrame({'nums': [1.0, 2.0, 3.0, 4.0, 5.0]})
        report = ar.quality.DataQualityReport(df)
        profile = report.columns['nums']
        result = profile.to_dict()

        numeric_fields = {'q25', 'q50', 'q75', 'q95', 'iqr', 
                         'outlier_lower_bound', 'outlier_upper_bound'}
        assert any(field in result for field in numeric_fields)

    def test_to_dict_string_dtype_excludes_quantiles(self):
        """Test string columns exclude quantile fields"""
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        report = ar.quality.DataQualityReport(df)
        profile = report.columns['text']
        result = profile.to_dict()

        numeric_fields = {'q25', 'q50', 'q75', 'q95'}
        assert not any(field in result for field in numeric_fields)

    def test_to_dict_json_serializable(self):
        """Test that to_dict result is JSON serializable"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)
        profile = report.columns['col']
        result = profile.to_dict()

        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)


class TestDataQualityReportScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown"""

    def test_score_breakdown_returns_dict_with_required_keys(self):
        """Test score_breakdown returns dict with penalties and final_score"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)
        result = report.score_breakdown()

        assert isinstance(result, dict)
        assert 'final_score' in result
        assert 'null_penalty' in result
        assert 'duplicate_penalty' in result
        assert 'type_mismatch_penalty' in result

    def test_score_breakdown_values_are_floats(self):
        """Test all score breakdown values are numeric"""
        df = pd.DataFrame({'col': [1, None, 3, 3]})
        report = ar.quality.DataQualityReport(df)
        result = report.score_breakdown()

        for value in result.values():
            assert isinstance(value, (int, float))

    def test_score_breakdown_final_score_matches_quality_score(self):
        """Test that final_score equals report's quality_score"""
        df = pd.DataFrame({'col': [1, 2, 3, None]})
        report = ar.quality.DataQualityReport(df)
        result = report.score_breakdown()

        assert result['final_score'] == report.quality_score


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict"""

    def test_to_dict_returns_dict_with_expected_structure(self):
        """Test to_dict returns dict with required top-level keys"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict()

        assert isinstance(result, dict)
        expected_keys = {'row_count', 'column_count', 'quality_score',
                        'columns', 'suggestions', 'score_components'}
        assert all(key in result for key in expected_keys)

    def test_to_dict_exclude_columns_list(self):
        """Test exclude_columns parameter with list"""
        df = pd.DataFrame({'col1': [1], 'col2': [2], 'col3': [3]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict(exclude_columns=['col1', 'col2'])

        assert 'col1' not in result['columns']
        assert 'col2' not in result['columns']
        assert 'col3' in result['columns']

    def test_to_dict_exclude_columns_set(self):
        """Test exclude_columns parameter with set"""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict(exclude_columns={'col1'})

        assert 'col1' not in result['columns']
        assert 'col2' in result['columns']

    def test_to_dict_exclude_columns_tuple(self):
        """Test exclude_columns parameter with tuple"""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict(exclude_columns=('col1',))

        assert 'col1' not in result['columns']

    def test_to_dict_exclude_columns_invalid_type_raises_typeerror(self):
        """Test exclude_columns raises TypeError for invalid type"""
        df = pd.DataFrame({'col': [1]})
        report = ar.quality.DataQualityReport(df)

        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns="col")

    def test_to_dict_exclude_columns_non_string_items_raises_typeerror(self):
        """Test exclude_columns raises TypeError if items are not strings"""
        df = pd.DataFrame({'col': [1]})
        report = ar.quality.DataQualityReport(df)

        with pytest.raises(TypeError, match="exclude_columns must contain only string"):
            report.to_dict(exclude_columns=[1, 2])

    def test_to_dict_exclude_columns_unknown_column_raises_keyerror(self):
        """Test exclude_columns raises KeyError for unknown columns"""
        df = pd.DataFrame({'col1': [1]})
        report = ar.quality.DataQualityReport(df)

        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=['unknown_col'])

    def test_to_dict_redact_sample_values_parameter(self):
        """Test redact_sample_values parameter is applied"""
        df = pd.DataFrame({'col': ['secret']})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict(redact_sample_values=True)

        for col_profile in result['columns'].values():
            if col_profile.get('sample_values'):
                assert all(v == '[REDACTED]' for v in col_profile['sample_values'])

    def test_to_dict_json_serializable(self):
        """Test to_dict result is JSON serializable"""
        df = pd.DataFrame({'col': [1, 2]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict()

        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json"""

    def test_to_json_returns_string_by_default(self):
        """Test to_json returns JSON string when output is None"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_json()

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_to_json_indent_parameter_formats_output(self):
        """Test indent parameter produces indented output"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)

        result_no_indent = report.to_json(indent=None)
        result_with_indent = report.to_json(indent=2)

        assert '\n' in result_with_indent
        assert len(result_with_indent) > len(result_no_indent)

    def test_to_json_output_parameter_writes_to_file_like_object(self):
        """Test output parameter writes to file-like object and returns None"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)

        output = io.StringIO()
        result = report.to_json(output=output)

        assert result is None
        content = output.getvalue()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_to_json_output_parameter_invalid_raises_typeerror(self):
        """Test output parameter raises TypeError for non-writable object"""
        df = pd.DataFrame({'col': [1]})
        report = ar.quality.DataQualityReport(df)

        with pytest.raises(TypeError, match="output must be"):
            report.to_json(output="not_writable")

    def test_to_json_passes_redact_sample_values_to_to_dict(self):
        """Test redact_sample_values parameter is passed through"""
        df = pd.DataFrame({'col': ['secret']})
        report = ar.quality.DataQualityReport(df)

        result = report.to_json(redact_sample_values=True)
        data = json.loads(result)

        for col_profile in data['columns'].values():
            if col_profile.get('sample_values'):
                assert all(v == '[REDACTED]' for v in col_profile['sample_values'])

    def test_to_json_passes_exclude_columns_to_to_dict(self):
        """Test exclude_columns parameter is passed through"""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        report = ar.quality.DataQualityReport(df)

        result = report.to_json(exclude_columns=['col1'])
        data = json.loads(result)

        assert 'col1' not in data['columns']
        assert 'col2' in data['columns']

    def test_to_json_combined_parameters(self):
        """Test multiple parameters work together"""
        df = pd.DataFrame({'col1': [1], 'col2': ['secret']})
        report = ar.quality.DataQualityReport(df)

        output = io.StringIO()
        result = report.to_json(
            indent=2,
            redact_sample_values=True,
            exclude_columns=['col1'],
            output=output
        )

        assert result is None
        content = output.getvalue()
        data = json.loads(content)
        assert 'col1' not in data['columns']
        assert 'col2' in data['columns']
        for col_profile in data['columns'].values():
            if col_profile.get('sample_values'):
                assert all(v == '[REDACTED]' for v in col_profile['sample_values'])
