import pytest
import pandas as pd
import arnio as ar
import numpy as np


class TestProfile:
    """Test suite for arnio.quality.profile function."""
    
    def test_profile_returns_report(self):
        """Profile should return a DataQualityReport."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        report = ar.profile(frame)
        assert hasattr(report, 'summary')
    
    def test_profile_basic_functionality(self):
        """Profile should work on basic frames with multiple dtypes."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.0, 92.3],
            "active": [True, False, True]
        })
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_exclude_single_column(self):
        """Profile should exclude specified columns from analysis."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "debug": ["x", "y", "z"]
        })
        report = ar.profile(frame, exclude_columns=["debug"])
        assert report is not None
    
    def test_profile_exclude_multiple_columns(self):
        """Profile should handle multiple excluded columns."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        report = ar.profile(frame, exclude_columns=["col1", "col2"])
        assert report is not None
    
    def test_profile_empty_frame(self):
        """Profile should handle empty frames gracefully."""
        frame = ar.from_dict({"name": [], "age": []})
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_with_nulls(self):
        """Profile should detect null values in columns."""
        frame = ar.from_dict({
            "name": ["Alice", None, "Charlie"],
            "age": [25, 30, None]
        })
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_custom_sample_size(self):
        """Profile should accept custom sample_size parameter."""
        frame = ar.from_dict({"col": list(range(100))})
        report1 = ar.profile(frame, sample_size=5)
        report2 = ar.profile(frame, sample_size=20)
        assert report1 is not None
        assert report2 is not None
    
    def test_profile_approx_top_values_disabled(self):
        """Profile should work with approx_top_values=False (default)."""
        frame = ar.from_dict({
            "category": ["A"] * 500 + ["B"] * 300 + ["C"] * 200
        })
        report = ar.profile(frame, approx_top_values=False)
        assert report is not None
    
    def test_profile_approx_top_values_enabled(self):
        """Profile should work with approx_top_values=True."""
        frame = ar.from_dict({
            "category": ["A"] * 500 + ["B"] * 300 + ["C"] * 200
        })
        report = ar.profile(frame, approx_top_values=True)
        assert report is not None
    
    def test_profile_approx_top_values_with_params(self):
        """Profile should accept approximation tuning parameters."""
        frame = ar.from_dict({
            "values": ["val_" + str(i) for i in range(2500)]
        })
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.15,
            approx_top_values_sample_size=3000
        )
        assert report is not None
    
    def test_profile_single_column_frame(self):
        """Profile should work on single-column frames."""
        frame = ar.from_dict({"data": [1, 2, 3, 4, 5]})
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_with_duplicates(self):
        """Profile should analyze frames with duplicate rows."""
        frame = ar.from_dict({
            "id": [1, 1, 2, 2, 3],
            "name": ["Alice", "Alice", "Bob", "Bob", "Charlie"]
        })
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_mixed_null_and_valid(self):
        """Profile should handle columns with mixed nulls and valid data."""
        frame = ar.from_dict({
            "col1": [1, None, 3, None, 5],
            "col2": ["a", "b", None, "d", "e"]
        })
        report = ar.profile(frame)
        assert report is not None
    
    def test_profile_pandas_accessor(self):
        """Profile should work through pandas DataFrame accessor."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 28]
        })
        report = df.arnio.profile()
        assert hasattr(report, 'summary')
    
    def test_profile_summary_method(self):
        """Profile report should have accessible summary method."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        report = ar.profile(frame)
        summary = report.summary()
        assert summary is not None
