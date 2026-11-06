import pytest
import pandas as pd
import arnio as ar
import re


class TestDropColumnsMatching:
    """Test suite for ar.drop_columns_matching()"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with various column names"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "temp_col1": [10, 20, 30],
            "temp_col2": [40, 50, 60],
            "customer_name": ["alice", "bob", "charlie"],
            "temp_notes": ["a", "b", "c"],
            "revenue": [100, 200, 300],
        })
        return ar.from_pandas(df)

    def test_drop_columns_with_prefix_pattern(self, sample_frame):
        """Drop columns starting with 'temp_'"""
        result = ar.drop_columns_matching(sample_frame, pattern="^temp_")
        remaining = result.columns
        assert "id" in remaining
        assert "customer_name" in remaining
        assert "revenue" in remaining
        assert "temp_col1" not in remaining
        assert "temp_col2" not in remaining
        assert "temp_notes" not in remaining

    def test_drop_columns_with_suffix_pattern(self, sample_frame):
        """Drop columns ending with specific pattern"""
        result = ar.drop_columns_matching(sample_frame, pattern="col[12]$")
        remaining = result.columns
        assert "temp_col1" not in remaining
        assert "temp_col2" not in remaining
        assert "temp_notes" in remaining

    def test_no_columns_match_returns_all_columns(self, sample_frame):
        """When no columns match, return frame with all columns"""
        result = ar.drop_columns_matching(sample_frame, pattern="^nonexistent_")
        assert set(result.columns) == set(sample_frame.columns)

    def test_all_columns_match_returns_empty_frame(self, sample_frame):
        """When all columns match pattern, return frame with no columns"""
        result = ar.drop_columns_matching(sample_frame, pattern=".*")
        assert result.shape[1] == 0

    def test_complex_regex_pattern(self):
        """Test with complex regex pattern using character class"""
        df = pd.DataFrame({
            "col_1": [1, 2],
            "col_2": [3, 4],
            "col_10": [5, 6],
            "other": [7, 8],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^col_[1-5]$")
        remaining = result.columns
        assert "col_1" not in remaining
        assert "col_2" not in remaining
        assert "col_10" in remaining
        assert "other" in remaining

    def test_column_order_preserved_after_drop(self, sample_frame):
        """Remaining columns maintain original order"""
        result = ar.drop_columns_matching(sample_frame, pattern="^temp_")
        expected_order = ["id", "customer_name", "revenue"]
        assert result.columns == expected_order

    def test_empty_frame(self):
        """Test with empty frame (zero rows)"""
        df = pd.DataFrame({"keep_col": [], "temp_col": []})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        assert len(result.columns) == 1
        assert "keep_col" in result.columns

    def test_single_column_match(self):
        """Drop a single column that matches the pattern"""
        df = pd.DataFrame({"id": [1, 2], "temp_col": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        assert "id" in result.columns
        assert "temp_col" not in result.columns

    def test_special_characters_in_column_names(self):
        """Pattern matching works with special characters in column names"""
        df = pd.DataFrame({
            "col-with-dash": [1, 2],
            "col_with_underscore": [3, 4],
            "col.with.dot": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern=r"col[\-_]")
        remaining = result.columns
        assert "col-with-dash" not in remaining
        assert "col_with_underscore" not in remaining
        assert "col.with.dot" in remaining

    def test_numeric_data_preserved(self, sample_frame):
        """Data integrity: numeric values preserved after column drop"""
        result = ar.drop_columns_matching(sample_frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        assert result_df["id"].tolist() == [1, 2, 3]
        assert result_df["revenue"].tolist() == [100, 200, 300]

    def test_string_data_preserved(self, sample_frame):
        """Data integrity: string values preserved after column drop"""
        result = ar.drop_columns_matching(sample_frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        assert result_df["customer_name"].tolist() == ["alice", "bob", "charlie"]

    def test_case_sensitive_pattern_matching(self, sample_frame):
        """Pattern matching is case-sensitive by default"""
        result = ar.drop_columns_matching(sample_frame, pattern="^TEMP_")
        assert len(result.columns) == len(sample_frame.columns)

    def test_alternation_pattern(self):
        """Pattern with alternation drops multiple column types"""
        df = pd.DataFrame({
            "id": [1],
            "temp_col": [2],
            "debug_col": [3],
            "keep": [4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^(temp_|debug_)")
        remaining = result.columns
        assert "temp_col" not in remaining
        assert "debug_col" not in remaining
        assert "keep" in remaining

    def test_invalid_regex_pattern_raises_error(self, sample_frame):
        """Invalid regex pattern should raise error"""
        with pytest.raises((re.error, ValueError)):
            ar.drop_columns_matching(sample_frame, pattern="[invalid(")

    def test_returns_new_frame_not_modifying_input(self, sample_frame):
        """Function returns new frame without modifying input"""
        original_columns = set(sample_frame.columns)
        result = ar.drop_columns_matching(sample_frame, pattern="^temp_")
        assert set(sample_frame.columns) == original_columns
        assert len(result.columns) < len(sample_frame.columns)
