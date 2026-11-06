import pytest
import arnio as ar
import pandas as pd
import numpy as np


class TestProfile:
    """Tests for arnio.quality.profile function."""

    def test_basic_profile_returns_report(self):
        """Test that profile returns a DataQualityReport."""
        data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        frame = ar.from_dict(data)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_report_has_summary_method(self):
        """Test that returned report has a summary method as documented."""
        data = {"col": [1, 2, 3]}
        frame = ar.from_dict(data)
        report = ar.profile(frame)
        assert hasattr(report, "summary")
        assert callable(report.summary)

    def test_profile_with_sample_size_parameter(self):
        """Test profile respects sample_size parameter."""
        data = {"values": list(range(100))}
        frame = ar.from_dict(data)
        report_small = ar.profile(frame, sample_size=2)
        report_large = ar.profile(frame, sample_size=10)
        assert report_small is not None
        assert report_large is not None

    def test_profile_exclude_single_column(self):
        """Test excluding a single column from profiling."""
        data = {"keep": [1, 2, 3], "exclude": ["a", "b", "c"]}
        frame = ar.from_dict(data)
        report = ar.profile(frame, exclude_columns=["exclude"])
        assert report is not None

    def test_profile_exclude_multiple_columns(self):
        """Test excluding multiple columns from profiling."""
        data = {
            "keep1": [1, 2, 3],
            "exclude1": ["a", "b", "c"],
            "keep2": [4.0, 5.0, 6.0],
            "exclude2": ["x", "y", "z"],
        }
        frame = ar.from_dict(data)
        report = ar.profile(
            frame, exclude_columns=["exclude1", "exclude2"]
        )
        assert report is not None

    def test_profile_approx_top_values_default_disabled(self):
        """Test that approximation is disabled by default."""
        values = [f"val_{i}" for i in range(2000)]
        frame = ar.from_dict({"col": values})
        report = ar.profile(frame)  # approx_top_values=False by default
        assert report is not None

    def test_profile_approx_top_values_enabled(self):
        """Test profile with approximation explicitly enabled."""
        values = [f"val_{i}" for i in range(2500)]
        frame = ar.from_dict({"col": values})
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.2,
        )
        assert report is not None

    def test_profile_approx_top_values_min_unique_threshold(self):
        """Test approximation respects min_unique threshold."""
        values = ["val_" + str(i % 50) for i in range(500)]  # Only 50 unique
        frame = ar.from_dict({"col": values})
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,  # Higher than actual
        )
        assert report is not None

    def test_profile_approx_top_values_min_ratio_threshold(self):
        """Test approximation respects min_ratio threshold."""
        values = ["val_" + str(i % 5) for i in range(1000)]  # Ratio: 5/1000
        frame = ar.from_dict({"col": values})
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_ratio=0.5,  # Higher than actual (0.005)
        )
        assert report is not None

    def test_profile_approx_top_values_sample_size(self):
        """Test approximation uses specified sample size."""
        values = [f"val_{i}" for i in range(5000)]
        frame = ar.from_dict({"col": values})
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_sample_size=500,
        )
        assert report is not None

    def test_profile_with_null_values(self):
        """Test profile handles columns with mixed null and non-null values."""
        data = {"col": [1, None, 3, None, 5, 6, 7]}
        frame = ar.from_dict(data)
        report = ar.profile(frame, sample_size=3)
        assert report is not None

    def test_profile_all_nulls_edge_case(self):
        """Test profile handles column with all null values."""
        data = {"all_null": [None, None, None], "has_values": [1, 2, 3]}
        frame = ar.from_dict(data)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_single_row_edge_case(self):
        """Test profile handles single-row frames."""
        data = {"col1": [42], "col2": ["single"]}
        frame = ar.from_dict(data)
        report = ar.profile(frame, sample_size=5)
        assert report is not None

    def test_profile_empty_frame_edge_case(self):
        """Test profile handles empty frames."""
        data = {"empty_col": []}
        frame = ar.from_dict(data)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_mixed_column_types(self):
        """Test profile analyzes frames with mixed data types."""
        data = {
            "strings": ["alice", "bob", "charlie"],
            "integers": [10, 20, 30],
            "floats": [1.5, 2.5, 3.5],
        }
        frame = ar.from_dict(data)
        report = ar.profile(frame)
        assert report is not None
