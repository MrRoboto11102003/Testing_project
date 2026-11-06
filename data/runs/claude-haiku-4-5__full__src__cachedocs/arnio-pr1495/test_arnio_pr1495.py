import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfile:
    """Tests for ar.profile() function."""

    def test_basic_profile_generation(self):
        """Test basic profile generation on a simple frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 2
        assert "id" in report.columns
        assert "name" in report.columns
        assert isinstance(report.quality_score, (int, float))

    def test_profile_with_null_values(self):
        """Test that nulls are counted correctly in column profiles."""
        df = pd.DataFrame({
            "value": [1.0, None, 3.0, None],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert report.columns["value"].null_count == 2

    def test_profile_detects_duplicate_rows(self):
        """Test that duplicate rows are detected."""
        df = pd.DataFrame({
            "id": [1, 2, 2],
            "name": ["alice", "bob", "bob"],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 1
        assert report.duplicate_ratio > 0

    def test_exclude_columns_parameter(self):
        """Test that exclude_columns removes columns from report."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "secret": ["x", "y", "z"],
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame, exclude_columns=["secret"])
        
        assert report.column_count == 2
        assert "secret" not in report.columns
        assert "id" in report.columns
        assert "name" in report.columns

    def test_sample_size_zero(self):
        """Test profile with sample_size=0 produces empty samples."""
        df = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame, sample_size=0)
        
        assert len(report.columns["name"].sample_values) == 0

    def test_sample_size_limits_samples(self):
        """Test that sample_size parameter limits sample values."""
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame, sample_size=2)
        
        assert len(report.columns["value"].sample_values) <= 2

    def test_approx_top_values_configuration(self):
        """Test that approx_top_values parameters work without error."""
        df = pd.DataFrame({
            "category": ["A"] * 100 + ["B"] * 50 + ["C"] * 10,
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=50,
            approx_top_values_min_ratio=0.1,
            approx_top_values_sample_size=1000,
        )
        
        assert "category" in report.columns

    def test_quality_score_in_bounds(self):
        """Test that quality_score is present and within valid range."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.0, 2.0, None],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert hasattr(report, "quality_score")
        assert 0 <= report.quality_score <= 100
        assert hasattr(report, "score_components")

    def test_suggestions_list_generated(self):
        """Test that suggestions list is generated."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "constant": ["same", "same", "same"],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert isinstance(report.suggestions, list)

    def test_memory_usage_recorded(self):
        """Test that memory_usage is recorded as positive value."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.0, 2.0, 3.0],
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert report.memory_usage > 0

    def test_invalid_sample_size_type_raises_typeerror(self):
        """Test that non-integer sample_size raises TypeError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size="5")

    def test_negative_sample_size_raises_valueerror(self):
        """Test that negative sample_size raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)

    def test_invalid_approx_ratio_bounds(self):
        """Test that out-of-bounds approx_top_values_min_ratio raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)

    def test_exclude_nonexistent_column_raises_error(self):
        """Test that excluding a non-existent column raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.profile(frame, exclude_columns=["nonexistent"])

    def test_empty_frame_profile(self):
        """Test profile on an empty frame."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object"),
        })
        frame = ar.from_pandas(df)
        
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0
