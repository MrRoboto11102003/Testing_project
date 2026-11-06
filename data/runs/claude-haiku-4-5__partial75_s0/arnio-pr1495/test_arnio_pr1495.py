import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os


class TestProfileQuality:
    """Tests for arnio.quality.profile function."""

    def test_profile_returns_data_quality_report(self):
        """profile() should return a DataQualityReport."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        assert report is not None
        assert hasattr(report, "summary")
        assert hasattr(report, "to_markdown")
        assert hasattr(report, "to_html")

    def test_profile_summary_returns_dict_with_metrics(self):
        """profile().summary() should return dict with row_count and columns."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        assert isinstance(summary, dict)
        assert "row_count" in summary
        assert summary["row_count"] == 5
        assert "columns" in summary

    def test_profile_exclude_columns_parameter(self):
        """exclude_columns parameter should exclude specified columns from report."""
        df = pd.DataFrame({"keep": [1, 2, 3], "drop": [4, 5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["drop"])
        summary = report.summary()

        columns = summary.get("columns", {})
        assert "keep" in columns
        assert "drop" not in columns

    def test_profile_sample_size_parameter(self):
        """sample_size parameter should control sample values per column."""
        df = pd.DataFrame({"col": list(range(100))})
        frame = ar.from_pandas(df)

        report_small = ar.profile(frame, sample_size=2)
        report_large = ar.profile(frame, sample_size=20)

        assert report_small is not None
        assert report_large is not None

    def test_profile_with_null_values(self):
        """profile() should handle columns with null values."""
        df = pd.DataFrame({"col": [1, None, 3, None, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        assert "col" in summary.get("columns", {})
        assert summary["row_count"] == 5

    def test_profile_with_empty_dataframe(self):
        """profile() should handle empty DataFrames."""
        df = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        assert summary["row_count"] == 0
        assert "col" in summary.get("columns", {})

    def test_profile_with_all_null_column(self):
        """profile() should handle columns with all null values."""
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "has_data": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        columns = summary.get("columns", {})
        assert "all_null" in columns
        assert "has_data" in columns

    def test_profile_multiple_dtypes(self):
        """profile() should handle mixed data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        columns = summary.get("columns", {})
        assert len(columns) == 4
        assert all(col in columns for col in ["int_col", "float_col", "str_col", "bool_col"])

    def test_profile_approx_top_values_enabled(self):
        """approx_top_values=True should work with high-cardinality columns."""
        df = pd.DataFrame({"many_vals": [f"val_{i}" for i in range(2500)]})
        frame = ar.from_pandas(df)

        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.2,
            approx_top_values_sample_size=2000
        )

        assert report is not None
        summary = report.summary()
        assert "many_vals" in summary.get("columns", {})

    def test_profile_to_markdown_output(self):
        """to_markdown() should return formatted markdown string."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_profile_to_html_returns_string(self):
        """to_html() without file_path should return HTML string."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        html = report.to_html()

        assert isinstance(html, str)
        assert len(html) > 0

    def test_profile_to_html_writes_file(self):
        """to_html(file_path) should write HTML to file."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "report.html")
            result = report.to_html(file_path=filepath)

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            assert isinstance(result, str)

    def test_profile_single_column_dataframe(self):
        """profile() should work with single-column DataFrames."""
        df = pd.DataFrame({"only": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        assert "only" in summary.get("columns", {})

    def test_profile_with_duplicates(self):
        """profile() should handle DataFrames with duplicate values."""
        df = pd.DataFrame({
            "unique": [1, 2, 3, 4, 5],
            "duped": [1, 1, 1, 2, 2]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()

        columns = summary.get("columns", {})
        assert "unique" in columns
        assert "duped" in columns

    def test_profile_multiple_exclude_columns(self):
        """exclude_columns should accept list of multiple columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["b", "d"])
        summary = report.summary()

        columns = summary.get("columns", {})
        assert "a" in columns
        assert "c" in columns
        assert "b" not in columns
        assert "d" not in columns
