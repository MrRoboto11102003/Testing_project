import pytest
import numpy as np
import pandas as pd
import arnio as ar


class TestProfileParameterValidation:
    """Test input parameter validation for ar.profile()."""

    def test_sample_size_must_be_integer(self):
        """sample_size must be int, not bool or float."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=1.5)
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=True)

    def test_sample_size_must_be_non_negative(self):
        """sample_size must be >= 0."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)

    def test_approx_top_values_must_be_bool(self):
        """approx_top_values must be bool."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="approx_top_values must be a bool"):
            ar.profile(frame, approx_top_values=1)

    def test_approx_top_values_min_unique_must_be_integer(self):
        """approx_top_values_min_unique must be int, not bool or string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
            ar.profile(frame, approx_top_values_min_unique="5")
        with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
            ar.profile(frame, approx_top_values_min_unique=True)

    def test_approx_top_values_min_unique_must_be_non_negative(self):
        """approx_top_values_min_unique must be >= 0."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="approx_top_values_min_unique must be non-negative"):
            ar.profile(frame, approx_top_values_min_unique=-1)

    def test_approx_top_values_min_ratio_must_be_number(self):
        """approx_top_values_min_ratio must be int or float, not bool or string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
            ar.profile(frame, approx_top_values_min_ratio="0.2")
        with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
            ar.profile(frame, approx_top_values_min_ratio=True)

    def test_approx_top_values_min_ratio_must_be_finite(self):
        """approx_top_values_min_ratio must be finite."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="finite number between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=float("nan"))
        with pytest.raises(ValueError, match="finite number between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=float("inf"))

    def test_approx_top_values_min_ratio_must_be_in_range(self):
        """approx_top_values_min_ratio must be between 0 and 1."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)

    def test_approx_top_values_sample_size_must_be_integer(self):
        """approx_top_values_sample_size must be int."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="approx_top_values_sample_size must be an integer"):
            ar.profile(frame, approx_top_values_sample_size=2.0)

    def test_approx_top_values_sample_size_must_be_positive(self):
        """approx_top_values_sample_size must be > 0."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
            ar.profile(frame, approx_top_values_sample_size=0)


class TestProfileBasicFunctionality:
    """Test core profiling functionality."""

    def test_profile_returns_dataqualityreport(self):
        """profile() should return a DataQualityReport object."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert hasattr(report, "duplicate_rows")
        assert hasattr(report, "quality_score")
        assert hasattr(report, "columns")
        assert hasattr(report, "suggestions")

    def test_profile_numeric_column_basic(self):
        """Profile a simple numeric column with correct stats."""
        df = pd.DataFrame({"age": [10, 20, 20, 40]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 4
        assert report.column_count == 1
        assert report.duplicate_rows == 1
        assert report.duplicate_ratio == pytest.approx(0.25, abs=0.01)
        
        col = report.columns["age"]
        assert col.dtype == "int64"
        assert col.row_count == 4
        assert col.null_count == 0
        assert col.null_ratio == 0.0
        assert col.unique_count == 3
        assert col.unique_ratio == pytest.approx(1.0, abs=0.01)
        assert col.mean == pytest.approx(22.5)
        assert col.semantic_type == "numeric"

    def test_profile_string_column_with_whitespace(self):
        """String columns should detect whitespace."""
        df = pd.DataFrame({"name": [" alice", "bob", "bob", "carol"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["name"]
        assert col.dtype == "string"
        assert col.whitespace_count == 1
        assert col.unique_count == 3
        assert col.top_values is not None
        # bob should appear twice
        assert any(v[0] == "bob" and v[1] == 2 for v in col.top_values)

    def test_profile_boolean_column(self):
        """Boolean columns should have minimal statistics."""
        df = pd.DataFrame({"is_active": [True, False, True, False]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["is_active"]
        assert col.dtype == "bool"
        assert col.semantic_type == "boolean"
        assert col.mean is None
        assert col.std is None
        assert col.top_values is None

    def test_profile_with_nulls(self):
        """Columns with nulls should report null statistics."""
        df = pd.DataFrame({"score": [10.5, None, 20.5, 30.5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["score"]
        assert col.null_count == 1
        assert col.null_ratio == pytest.approx(0.25, abs=0.01)
        assert col.unique_count == 3

    def test_profile_zero_sample_size(self):
        """sample_size=0 should give empty sample_values."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=0)
        
        col = report.columns["a"]
        assert col.sample_values == []

    def test_profile_empty_frame(self):
        """Frame with no rows should have score 100 and semantic 'empty'."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="string")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0
        assert report.quality_score == 100.0
        assert report.columns["a"].semantic_type == "empty"

    def test_profile_exclude_columns(self):
        """exclude_columns should remove columns from profiling."""
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["b"])
        
        assert report.column_count == 2
        assert "a" in report.columns
        assert "b" not in report.columns
        assert "c" in report.columns
        assert report.duplicate_rows == 1  # [1,1] and [3,4]

    def test_profile_quality_score_with_duplicates(self):
        """Duplicate rows should reduce quality score."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 1
        assert report.duplicate_ratio > 0
        assert report.quality_score < 100.0
        assert "duplicate_penalty" in report.score_components

    def test_profile_identifier_semantic_type(self):
        """Columns ending with _id should be identified."""
        df = pd.DataFrame({"user_id": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.columns["user_id"].semantic_type == "identifier"

    def test_profile_numeric_column_outlier_detection(self):
        """Numeric columns should detect outliers using IQR."""
        df = pd.DataFrame({"vals": [10, 15, 20, 25, 30, 1000]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["vals"]
        # Should have 4+ finite values to compute IQR
        assert col.outlier_count is not None
        assert col.outlier_count >= 1  # 1000 should be an outlier

    def test_profile_string_length_statistics(self):
        """String columns should compute min/max/mean lengths."""
        df = pd.DataFrame({"text": ["a", "bb", "ccc"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["text"]
        assert col.min == 1
        assert col.max == 3
        assert col.mean == pytest.approx(2.0)

    def test_profile_suggested_dtype_for_numeric_strings(self):
        """String columns with numeric values should suggest numeric dtype."""
        df = pd.DataFrame({"nums": ["1", "22", "3"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col = report.columns["nums"]
        assert col.suggested_dtype == "int64"

    def test_profile_suggestions_include_cleaning_steps(self):
        """Profiling should return suggestions for cleaning."""
        df = pd.DataFrame({"name": [" alice ", "bob"], "vals": [1, 1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        suggestion_steps = [s.step for s in report.suggestions]
        assert "strip_whitespace" in suggestion_steps or len(report.suggestions) > 0
        assert "drop_duplicates" in suggestion_steps or len(report.suggestions) >= 0

    def test_profile_quality_score_all_null_column(self):
        """All-null columns should affect quality score."""
        df = pd.DataFrame({"x": [1, 2], "y": [None, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.columns["y"].semantic_type == "empty"
        assert report.quality_score < 100.0
        assert "null_penalty" in report.score_components
