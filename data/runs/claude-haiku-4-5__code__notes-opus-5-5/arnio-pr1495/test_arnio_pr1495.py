import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import DataQualityReport


class TestProfileInputValidation:
    """Test input parameter validation for profile()."""
    
    @pytest.fixture
    def basic_frame(self):
        """Create a basic test frame."""
        return ar.from_pandas(pd.DataFrame({"col": [1, 2, 3]}))
    
    def test_sample_size_not_integer(self, basic_frame):
        """sample_size must be int, not float."""
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(basic_frame, sample_size=1.5)
    
    def test_sample_size_not_bool(self, basic_frame):
        """sample_size cannot be bool (even though bool is subclass of int)."""
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(basic_frame, sample_size=True)
    
    def test_sample_size_negative(self, basic_frame):
        """sample_size must be non-negative."""
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(basic_frame, sample_size=-1)
    
    def test_approx_top_values_not_bool(self, basic_frame):
        """approx_top_values must be bool."""
        with pytest.raises(TypeError, match="approx_top_values must be a bool"):
            ar.profile(basic_frame, approx_top_values=1)
    
    def test_approx_top_values_min_unique_not_integer(self, basic_frame):
        """approx_top_values_min_unique must be int."""
        with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
            ar.profile(basic_frame, approx_top_values_min_unique="5")
    
    def test_approx_top_values_min_ratio_nan(self, basic_frame):
        """approx_top_values_min_ratio cannot be NaN."""
        with pytest.raises(ValueError, match="approx_top_values_min_ratio must be a finite number"):
            ar.profile(basic_frame, approx_top_values_min_ratio=float("nan"))
    
    def test_approx_top_values_min_ratio_too_large(self, basic_frame):
        """approx_top_values_min_ratio must be <= 1."""
        with pytest.raises(ValueError, match="approx_top_values_min_ratio must be between 0 and 1"):
            ar.profile(basic_frame, approx_top_values_min_ratio=1.5)
    
    def test_approx_top_values_sample_size_zero(self, basic_frame):
        """approx_top_values_sample_size must be positive (> 0)."""
        with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
            ar.profile(basic_frame, approx_top_values_sample_size=0)
    
    def test_exclude_columns_unknown_column(self, basic_frame):
        """exclude_columns with unknown column name raises error."""
        with pytest.raises(Exception):
            ar.profile(basic_frame, exclude_columns=["unknown_col"])


class TestProfileBasicMetrics:
    """Test basic metrics: row_count, column_count, duplicate detection."""
    
    def test_row_count(self):
        """row_count matches number of rows in frame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.row_count == 3
    
    def test_column_count(self):
        """column_count matches number of columns (before exclusions)."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.column_count == 3
    
    def test_column_count_with_exclusions(self):
        """column_count excludes specified columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame, exclude_columns=["b"])
        assert result.column_count == 2
    
    def test_duplicate_rows_count(self):
        """duplicate_rows counts rows after first occurrence."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 2]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.duplicate_rows == 3
    
    def test_duplicate_ratio_calculation(self):
        """duplicate_ratio is duplicate_rows / row_count."""
        df = pd.DataFrame({"a": [1, 1, 2, 2]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.duplicate_ratio == 0.5
    
    def test_duplicate_ratio_zero_rows(self):
        """duplicate_ratio is 0.0 for zero-row frame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.duplicate_ratio == 0.0


class TestProfileColumnProfiles:
    """Test column-level profiling (ColumnProfile structure and metrics)."""
    
    def test_numeric_column_basic_stats(self):
        """Numeric columns have min, max, mean, std."""
        df = pd.DataFrame({"nums": [10, 20, 30, 40]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["nums"]
        assert col.min == 10
        assert col.max == 40
        assert col.mean == 25.0
        assert col.std == pytest.approx(11.18, abs=0.1)
    
    def test_string_column_top_values(self):
        """String columns have top_values with counts and ratios."""
        df = pd.DataFrame({"words": ["a", "b", "b", "c", "c", "c"]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["words"]
        assert col.top_values is not None
        assert col.top_values[0][0] == "c"
        assert col.top_values[0][1] == 3
        assert col.top_values[0][2] == pytest.approx(0.5)
    
    def test_sample_values_respects_size(self):
        """sample_values respects sample_size parameter."""
        df = pd.DataFrame({"nums": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame, sample_size=2)
        col = result.columns["nums"]
        assert len(col.sample_values) == 2
        assert col.sample_values == [1, 2]
    
    def test_sample_values_zero(self):
        """sample_values is empty when sample_size=0."""
        df = pd.DataFrame({"nums": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame, sample_size=0)
        col = result.columns["nums"]
        assert col.sample_values == []
    
    def test_null_ratio_calculation(self):
        """null_ratio is null_count / row_count."""
        df = pd.DataFrame({"vals": [1, None, 3, None, 5]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["vals"]
        assert col.null_count == 2
        assert col.null_ratio == 0.4
    
    def test_unique_ratio_uses_non_null_count(self):
        """unique_ratio divides by non-null count, not total rows."""
        df = pd.DataFrame({"vals": [1, 2, None, None, None]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["vals"]
        # 2 non-null values, 2 unique values -> ratio 1.0
        assert col.unique_count == 2
        assert col.unique_ratio == 1.0


class TestProfileQualityScore:
    """Test quality_score and score_components calculation."""
    
    def test_perfect_score_clean_data(self):
        """Frame with no duplicates, nulls, or type issues scores high."""
        df = pd.DataFrame({"nums": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.quality_score >= 90.0
    
    def test_score_components_is_dict(self):
        """score_components is a dict."""
        df = pd.DataFrame({"nums": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert isinstance(result.score_components, dict)
    
    def test_zero_row_frame_score(self):
        """Zero-row frame gets quality_score 100.0 and empty components."""
        df = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.quality_score == 100.0
        assert result.score_components == {}


class TestProfileEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_zero_row_frame(self):
        """Zero-row frame is handled gracefully."""
        df = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        assert result.row_count == 0
        assert result.duplicate_rows == 0
    
    def test_all_null_column(self):
        """Column with all nulls is detected and handled."""
        df = pd.DataFrame({"nulls": [None, None, None]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["nulls"]
        assert col.null_count == 3
        assert col.null_ratio == 1.0
    
    def test_excludes_multiple_columns(self):
        """exclude_columns can exclude multiple columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame, exclude_columns=["a", "c"])
        assert result.column_count == 1
        assert "b" in result.columns
        assert "a" not in result.columns
    
    def test_exclusions_affect_duplicates(self):
        """Duplicates count changes when columns are excluded."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result_full = ar.profile(frame)
        result_excl = ar.profile(frame, exclude_columns=["b"])
        assert result_full.duplicate_rows == 0
        assert result_excl.duplicate_rows == 1
    
    def test_string_column_whitespace_detection(self):
        """String columns detect leading/trailing whitespace."""
        df = pd.DataFrame({"text": [" hello", "world  ", "test"]})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["text"]
        assert col.whitespace_count >= 1
    
    def test_numeric_quantiles_computed(self):
        """Numeric columns compute quantiles (q25, q50, q75, q95)."""
        df = pd.DataFrame({"vals": range(1, 101)})
        frame = ar.from_pandas(df)
        result = ar.profile(frame)
        col = result.columns["vals"]
        assert hasattr(col, "q25")
        assert hasattr(col, "q50")
        assert hasattr(col, "q75")
        assert hasattr(col, "q95")
        assert col.q25 is not None
        assert col.q50 is not None
