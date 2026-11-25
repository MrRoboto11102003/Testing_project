import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import arnio as ar


class TestProfile:
    """Tests for ar.profile() function."""

    def test_profile_basic_frame(self):
        """Test basic profiling of a simple frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert "id" in report.columns
        assert "name" in report.columns

    def test_profile_null_count_tracking(self):
        """Test profiling accurately counts null values."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["Alice", None, "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        id_profile = report.columns["id"]
        name_profile = report.columns["name"]
        assert id_profile.null_count == 1
        assert name_profile.null_count == 1

    def test_profile_exclude_columns_single(self):
        """Test exclude_columns parameter filters specified columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "debug": ["x", "y", "z"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["debug"])
        
        assert "id" in report.columns
        assert "name" in report.columns
        assert "debug" not in report.columns

    def test_profile_exclude_columns_multiple(self):
        """Test excluding multiple columns at once."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "temp1": ["a", "b", "c"],
            "temp2": ["x", "y", "z"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["temp1", "temp2"])
        
        assert len(report.columns) == 2
        assert "temp1" not in report.columns
        assert "temp2" not in report.columns

    def test_profile_sample_size_parameter(self):
        """Test sample_size parameter is accepted and used."""
        df = pd.DataFrame({
            "values": list(range(100)),
        })
        frame = ar.from_pandas(df)
        
        # Different sample sizes should both complete successfully
        report_small = ar.profile(frame, sample_size=3)
        report_large = ar.profile(frame, sample_size=50)
        
        assert report_small.row_count == 100
        assert report_large.row_count == 100

    def test_profile_empty_frame(self):
        """Test profiling an empty frame with no rows."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="string"),
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert "id" in report.columns
        assert "name" in report.columns

    def test_profile_single_row_frame(self):
        """Test profiling a single-row frame."""
        df = pd.DataFrame({
            "id": [1],
            "name": ["Alice"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 1
        assert len(report.columns) == 2

    def test_profile_all_nulls_column(self):
        """Test profiling a column with all null values."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "all_null": [None, None, None],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        null_col = report.columns["all_null"]
        assert null_col.null_count == 3

    def test_profile_summary_returns_dict(self):
        """Test summary() method returns a dictionary."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        summary = report.summary()
        
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_profile_to_markdown_output(self):
        """Test to_markdown() produces valid markdown string."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        markdown = report.to_markdown()
        
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        # Markdown typically contains pipes or dashes for tables
        assert any(char in markdown for char in ['|', '-', '#'])

    def test_profile_to_html_output(self):
        """Test to_html() produces HTML string."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        html = report.to_html()
        
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<" in html and ">" in html  # HTML tags

    def test_profile_to_html_file_write(self):
        """Test to_html() writes to file when path provided."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "report.html")
            result = report.to_html(file_path=output_path)
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            # Verify file was written
            with open(output_path, "r") as f:
                content = f.read()
                assert len(content) > 0

    def test_profile_approx_top_values_parameter(self):
        """Test approx_top_values parameter is accepted."""
        # Create high-cardinality data
        df = pd.DataFrame({
            "id": range(2000),
            "value": [f"val_{i}" for i in range(2000)],
        })
        frame = ar.from_pandas(df)
        
        # Profile with approximation settings
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.2,
            approx_top_values_sample_size=500,
        )
        
        assert report.row_count == 2000
        assert "id" in report.columns

    def test_profile_mixed_dtypes(self):
        """Test profiling frame with mixed data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert len(report.columns) == 4
        for col in ["int_col", "float_col", "str_col", "bool_col"]:
            assert col in report.columns
            assert report.columns[col].null_count == 0

    def test_profile_column_profile_object_attributes(self):
        """Test ColumnProfile objects have expected attributes."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col_profile = report.columns["values"]
        # Should have null_count attribute
        assert hasattr(col_profile, "null_count")
        assert col_profile.null_count == 0
