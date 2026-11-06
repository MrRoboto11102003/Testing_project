import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfileBasicFunctionality:
    """Test core profiling behavior."""

    def test_profile_returns_report_with_required_fields(self):
        """Profile should return DataQualityReport with all required fields."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert hasattr(report, "duplicate_rows")
        assert hasattr(report, "duplicate_ratio")
        assert hasattr(report, "quality_score")
        assert hasattr(report, "score_components")
        assert hasattr(report, "columns")
        assert hasattr(report, "suggestions")

    def test_profile_correct_row_and_column_counts(self):
        """Profile should report accurate row and column counts."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": ["x", "y", "z", "w"],
            "c": [1.5, 2.5, 3.5, 4.5]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 4
        assert report.column_count == 3

    def test_profile_detects_duplicate_rows(self):
        """Profile should detect and count exact duplicate rows."""
        df = pd.DataFrame({
            "col1": [1, 1, 2, 3, 3, 3],
            "col2": ["a", "a", "b", "c", "c", "c"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 3  # rows 1, 4, 5 are duplicates
        assert report.duplicate_ratio == pytest.approx(0.5)

    def test_profile_with_default_parameters(self):
        """Profile should work with default parameters."""
        df = pd.DataFrame({"col1": list(range(20)), "col2": list("a" * 20)})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 20
        assert report.column_count == 2


class TestProfileExcludeColumns:
    """Test exclude_columns parameter."""

    def test_exclude_single_column(self):
        """Profile should exclude specified column from analysis."""
        df = pd.DataFrame({
            "keep1": [1, 2, 3],
            "exclude_me": [100, 200, 300],
            "keep2": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["exclude_me"])
        
        assert report.column_count == 2
        assert "keep1" in report.columns
        assert "keep2" in report.columns
        assert "exclude_me" not in report.columns

    def test_exclude_multiple_columns(self):
        """Profile should exclude multiple columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "d": [7, 8]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["b", "d"])
        
        assert report.column_count == 2
        assert set(report.columns.keys()) == {"a", "c"}

    def test_exclude_nonexistent_column_raises_error(self):
        """Profile should raise ValueError when excluding non-existent column."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.profile(frame, exclude_columns=["does_not_exist"])


class TestProfileParameterValidation:
    """Test parameter validation and type checking."""

    def test_sample_size_must_be_nonnegative_int(self):
        """Profile should require sample_size >= 0 and type int."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)
        
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=1.5)

    def test_approx_top_values_ratio_must_be_valid_float(self):
        """Profile should validate ratio between 0 and 1."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)

    def test_approx_top_values_sample_size_must_be_positive(self):
        """Profile should require positive sample size."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="must be positive"):
            ar.profile(frame, approx_top_values_sample_size=0)

    def test_invalid_frame_type_raises_error(self):
        """Profile should raise error for non-ArFrame input."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        
        with pytest.raises((TypeError, AttributeError)):
            ar.profile(df)


class TestProfileEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_profile_empty_frame(self):
        """Profile should handle empty frame gracefully."""
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64"), 
                          "col2": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0

    def test_profile_single_row_frame(self):
        """Profile should handle single row frame."""
        df = pd.DataFrame({"col1": [1], "col2": ["a"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 1
        assert report.column_count == 2

    def test_profile_all_nulls_column(self):
        """Profile should handle column with all nulls."""
        df = pd.DataFrame({
            "normal": [1, 2, 3],
            "all_nulls": [None, None, None]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert "all_nulls" in report.columns
        col_profile = report.columns["all_nulls"]
        assert col_profile.null_count == 3

    def test_profile_with_zero_sample_size(self):
        """Profile should work with sample_size=0."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=0)
        
        assert report.column_count == 1
        # Should complete without error


class TestProfileQualityScore:
    """Test quality score computation."""

    def test_quality_score_is_numeric_in_valid_range(self):
        """Quality score should be between 0 and 100."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert isinstance(report.quality_score, (int, float))
        assert 0 <= report.quality_score <= 100

    def test_quality_score_has_components(self):
        """Quality score should include component breakdown."""
        df = pd.DataFrame({
            "col1": [1, 1, 2],  # has duplicate
            "col2": [None, "b", "c"]  # has null
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert isinstance(report.score_components, dict)


class TestProfileSuggestions:
    """Test cleaning suggestions."""

    def test_profile_provides_suggestions(self):
        """Profile should provide cleaning suggestions."""
        df = pd.DataFrame({
            "col1": [1, 1, 2],  # duplicates
            "col2": [" a ", " b ", " c "]  # whitespace
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert isinstance(report.suggestions, list)
