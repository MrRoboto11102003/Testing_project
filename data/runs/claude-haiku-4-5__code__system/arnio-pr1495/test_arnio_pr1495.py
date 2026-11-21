import pytest
import pandas as pd
import tempfile
import csv
import sys
import subprocess
from pathlib import Path

import arnio as ar


class TestProfileBasic:
    """Test basic profiling functionality."""
    
    def test_profile_simple_csv(self, tmp_path):
        """Test profiling a simple CSV file."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "age", "score"])
            writer.writerow([1, "Alice", 30, 95.5])
            writer.writerow([2, "Bob", 25, 87.0])
            writer.writerow([3, "Charlie", None, 92.3])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 4
        assert "id" in report.columns
        assert "age" in report.columns
        assert report.columns["age"].null_count == 1
        assert report.duplicate_rows == 0
        assert 0 <= report.quality_score <= 100
        assert hasattr(report, "score_components")
        assert hasattr(report, "suggestions")


class TestProfileExcludeColumns:
    """Test exclude_columns parameter."""
    
    def test_exclude_single_column(self, tmp_path):
        """Test excluding a single column from profiling."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "sensitive", "value"])
            writer.writerow([1, "secret1", 100])
            writer.writerow([2, "secret2", 200])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, exclude_columns=["sensitive"])
        
        assert report.column_count == 2
        assert "id" in report.columns
        assert "value" in report.columns
        assert "sensitive" not in report.columns
    
    def test_exclude_multiple_columns(self, tmp_path):
        """Test excluding multiple columns."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c", "d", "e"])
            writer.writerow([1, 2, 3, 4, 5])
            writer.writerow([6, 7, 8, 9, 10])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, exclude_columns=["b", "d"])
        
        assert report.column_count == 3
        assert report.columns.keys() == {"a", "c", "e"}


class TestProfileSampleSize:
    """Test sample_size parameter."""
    
    def test_sample_size_zero(self, tmp_path):
        """Test sample_size=0."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["values"])
            for i in range(5):
                writer.writerow([i])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=0)
        
        assert report.row_count == 5
        assert report.column_count == 1
    
    def test_sample_size_custom(self, tmp_path):
        """Test custom sample_size parameter."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num"])
            for i in range(100):
                writer.writerow([i])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=10)
        
        assert report.row_count == 100


class TestProfileApproxTopValues:
    """Test approximate top values functionality."""
    
    def test_approx_top_values_enabled(self, tmp_path):
        """Test profiling with approx_top_values enabled."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category"])
            for i in range(2500):
                writer.writerow([f"cat_{i % 1500}"])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.3,
            approx_top_values_sample_size=1000
        )
        
        assert report.row_count == 2500
        assert report.quality_score >= 0


class TestProfileEdgeCases:
    """Test edge cases."""
    
    def test_empty_dataframe(self, tmp_path):
        """Test profiling an empty dataframe."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2", "col3"])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 3
        assert report.duplicate_rows == 0
    
    def test_all_nulls_column(self, tmp_path):
        """Test profiling with all-null column."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["valid", "all_null"])
            for i in range(5):
                writer.writerow([i, None])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.columns["all_null"].null_count == 5
        assert report.columns["valid"].null_count == 0
    
    def test_duplicate_rows_detection(self, tmp_path):
        """Test detection of duplicate rows."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow([1, "x"])
            writer.writerow([1, "x"])
            writer.writerow([2, "y"])
            writer.writerow([1, "x"])
            writer.writerow([3, "z"])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 5
        assert report.duplicate_rows == 2
        assert report.duplicate_ratio == 0.4


class TestProfileValidation:
    """Test input validation."""
    
    def test_negative_sample_size_error(self, tmp_path):
        """Test that negative sample_size raises ValueError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col"])
            writer.writerow([1])
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)
    
    def test_non_integer_sample_size_error(self, tmp_path):
        """Test that non-integer sample_size raises TypeError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col"])
            writer.writerow([1])
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=3.14)
    
    def test_invalid_approx_ratio_error(self, tmp_path):
        """Test that out-of-range approx_top_values_min_ratio raises ValueError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col"])
            writer.writerow([1])
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="approx_top_values_min_ratio must be between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
    
    def test_nonexistent_column_in_exclude_error(self, tmp_path):
        """Test that excluding nonexistent column raises error."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2"])
            writer.writerow([1, 2])
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(Exception):
            ar.profile(frame, exclude_columns=["nonexistent"])
    
    def test_non_positive_approx_sample_size_error(self, tmp_path):
        """Test that non-positive approx_top_values_sample_size raises ValueError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col"])
            writer.writerow([1])
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
            ar.profile(frame, approx_top_values_sample_size=0)


class TestProfileReportMetrics:
    """Test report metrics and structure."""
    
    def test_quality_score_valid_range(self, tmp_path):
        """Test that quality_score is in valid range."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            writer.writerow([1, 2, 3])
            writer.writerow([4, 5, 6])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert isinstance(report.quality_score, (int, float))
        assert 0 <= report.quality_score <= 100
    
    def test_memory_usage_reported(self, tmp_path):
        """Test that memory_usage is reported in bytes."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            for i in range(10):
                writer.writerow([f"value_{i}"])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert hasattr(report, "memory_usage")
        assert isinstance(report.memory_usage, int)
        assert report.memory_usage > 0
    
    def test_report_has_column_details(self, tmp_path):
        """Test that report contains details for each column."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "value"])
            writer.writerow([1, "test", 100])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert len(report.columns) == 3
        for col_name in ["id", "name", "value"]:
            assert col_name in report.columns
            col_info = report.columns[col_name]
            assert hasattr(col_info, "null_count")


class TestProfileDataTypes:
    """Test profiling with various data types."""
    
    def test_profile_mixed_types(self, tmp_path):
        """Test profiling with mixed data types."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["int_col", "str_col", "float_col"])
            writer.writerow([1, "text", 1.1])
            writer.writerow([2, "more", 2.2])
            writer.writerow([3, None, 3.3])
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.column_count == 3
        assert report.row_count == 3
        assert "int_col" in report.columns
        assert "str_col" in report.columns
        assert "float_col" in report.columns


class TestProfileCLI:
    """Test profiling via CLI."""
    
    def test_profile_cli_basic(self, tmp_path):
        """Test profile command via CLI."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow([1, 2])
            writer.writerow([3, 4])
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "profile", str(csv_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "row_count" in result.stdout or "rows" in result.stdout.lower()
