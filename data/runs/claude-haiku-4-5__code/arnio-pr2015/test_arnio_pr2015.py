import pytest
import pandas as pd
import arnio as ar


class TestParseBoolStrings:
    """Tests for arnio.cleaning.parse_bool_strings"""

    def test_default_true_values(self):
        """Test conversion with default true values"""
        df = pd.DataFrame({'col': ['true', 'yes', 'y', '1']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert all(result_df['col'] == True)

    def test_default_false_values(self):
        """Test conversion with default false values"""
        df = pd.DataFrame({'col': ['false', 'no', 'n', '0']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert all(result_df['col'] == False)

    def test_case_insensitivity(self):
        """Test that conversion is case-insensitive"""
        df = pd.DataFrame({'col': ['TRUE', 'False', 'YES', 'No']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df['col'].iloc[0] == True
        assert result_df['col'].iloc[1] == False
        assert result_df['col'].iloc[2] == True
        assert result_df['col'].iloc[3] == False

    def test_whitespace_handling(self):
        """Test that whitespace is stripped"""
        df = pd.DataFrame({'col': [' true ', '  false  ', '\ttrue\t']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df['col'].iloc[0] == True
        assert result_df['col'].iloc[1] == False
        assert result_df['col'].iloc[2] == True

    def test_custom_true_values(self):
        """Test conversion with custom true values"""
        df = pd.DataFrame({'col': ['on', 'off', 'active']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, true_values={'on', 'active'})
        result_df = ar.to_pandas(result)
        assert result_df['col'].iloc[0] == True
        assert result_df['col'].iloc[1] == 'off'
        assert result_df['col'].iloc[2] == True

    def test_custom_false_values(self):
        """Test conversion with custom false values"""
        df = pd.DataFrame({'col': ['on', 'off', 'inactive']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, false_values={'off', 'inactive'})
        result_df = ar.to_pandas(result)
        assert result_df['col'].iloc[0] == 'on'
        assert result_df['col'].iloc[1] == False
        assert result_df['col'].iloc[2] == False

    def test_subset_parameter(self):
        """Test that subset parameter limits conversion to specified columns"""
        df = pd.DataFrame({
            'col_a': ['true', 'false'],
            'col_b': ['true', 'false'],
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=['col_a'])
        result_df = ar.to_pandas(result)
        assert result_df['col_a'].iloc[0] == True
        assert result_df['col_a'].iloc[1] == False
        assert isinstance(result_df['col_b'].iloc[0], str)

    def test_ignores_boolean_columns(self):
        """Test that existing boolean columns are ignored"""
        df = pd.DataFrame({
            'bool_col': pd.array([True, False], dtype='bool'),
            'str_col': ['true', 'false'],
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert pd.api.types.is_bool_dtype(result_df['bool_col'])
        assert result_df['bool_col'].iloc[0] == True

    def test_preserves_unsupported_values(self):
        """Test that unsupported string values are preserved"""
        df = pd.DataFrame({'col': ['true', 'maybe', 'false', 'unknown']})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df['col'].iloc[0] == True
        assert result_df['col'].iloc[1] == 'maybe'
        assert result_df['col'].iloc[2] == False
        assert result_df['col'].iloc[3] == 'unknown'

    def test_true_values_bare_string_error(self):
        """Test that passing a bare string for true_values raises TypeError"""
        df = pd.DataFrame({'col': ['true']})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="true_values must be a set/list/tuple"):
            ar.parse_bool_strings(frame, true_values='true')

    def test_false_values_bare_string_error(self):
        """Test that passing a bare string for false_values raises TypeError"""
        df = pd.DataFrame({'col': ['false']})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="false_values must be a set/list/tuple"):
            ar.parse_bool_strings(frame, false_values='false')

    def test_true_values_non_string_error(self):
        """Test that non-string items in true_values raises TypeError"""
        df = pd.DataFrame({'col': ['true']})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values={'true', 1})

    def test_false_values_non_string_error(self):
        """Test that non-string items in false_values raises TypeError"""
        df = pd.DataFrame({'col': ['false']})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="false_values must contain only strings"):
            ar.parse_bool_strings(frame, false_values={'false', 1})

    def test_overlapping_values_error(self):
        """Test that overlapping true and false values raises ValueError"""
        df = pd.DataFrame({'col': ['true']})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="true_values and false_values overlap"):
            ar.parse_bool_strings(
                frame,
                true_values={'maybe'},
                false_values={'maybe'}
            )

    def test_subset_nonexistent_column_error(self):
        """Test that specifying non-existent columns in subset raises ValueError"""
        df = pd.DataFrame({'col_a': ['true']})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.parse_bool_strings(frame, subset=['nonexistent'])
