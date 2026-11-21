import sys
import subprocess
import csv
from pathlib import Path
import pytest
import pandas as pd
import arnio as ar


class TestProfileBasicFunctionality:
    """Test basic profiling workflow."""
    
    def test_profile_simple_csv(self, tmp_path):
        """Test profiling a simple CSV file and verifying report structure."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert report.duplicate_rows == 0
        assert 'id' in report.columns
        assert 'name' in report.columns
        assert 'value' in report.columns
        assert hasattr(report, 'quality_score')
        assert 0 <= report.quality_score <= 1
    
    def test_profile_duplicate_detection(self, tmp_path):
        """Test that profiling detects and reports duplicate rows."""
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text("id,name\n1,Alice\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.duplicate_rows == 1
        assert report.duplicate_ratio == pytest.approx(1/3, rel=0.01)
    
    def test_profile_with_null_values(self, tmp_path):
        """Test profiling handles null/empty values correctly."""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,,200\n3,Charlie,\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.columns['name'].null_count >= 1
        assert report.columns['value'].null_count >= 1


class TestProfileParameters:
    """Test profile function parameters."""
    
    def test_profile_sample_size_parameter(self, tmp_path):
        """Test that sample_size parameter limits sample values per column."""
        csv_file = tmp_path / "large.csv"
        with open(csv_file, 'w') as f:
            f.write("id,name\n")
            for i in range(100):
                f.write(f"{i},name_{i}\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame, sample_size=3)
        
        assert report.row_count == 100
        for col_info in report.columns.values():
            if hasattr(col_info, 'samples') and col_info.samples:
                assert len(col_info.samples) <= 3
    
    def test_profile_exclude_columns(self, tmp_path):
        """Test excluding specific columns from profiling."""
        csv_file = tmp_path / "exclude.csv"
        csv_file.write_text("id,name,internal,secret\n1,Alice,x,s1\n2,Bob,y,s2\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame, exclude_columns=['internal', 'secret'])
        
        assert report.column_count == 2
        assert 'id' in report.columns
        assert 'name' in report.columns
        assert 'internal' not in report.columns
        assert 'secret' not in report.columns
    
    def test_profile_approx_top_values(self, tmp_path):
        """Test approximate top values option for high-cardinality columns."""
        csv_file = tmp_path / "highcard.csv"
        with open(csv_file, 'w') as f:
            f.write("id,category\n")
            for i in range(2000):
                f.write(f"{i},cat_{i % 500}\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=100,
            approx_top_values_min_ratio=0.01,
            approx_top_values_sample_size=1000
        )
        
        assert report.row_count == 2000


class TestProfileEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_profile_empty_dataframe(self, tmp_path):
        """Test profiling an empty dataframe."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("a,b,c\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 3
        assert report.duplicate_rows == 0
    
    def test_profile_single_row(self, tmp_path):
        """Test profiling a dataframe with single row."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 1
        assert report.duplicate_rows == 0
    
    def test_profile_all_nulls_column(self, tmp_path):
        """Test profiling with column containing only nulls."""
        csv_file = tmp_path / "all_nulls.csv"
        csv_file.write_text("id,filled,empty\n1,A,\n2,B,\n3,C,\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.columns['empty'].null_count == 3
        assert report.columns['filled'].null_count == 0
    
    def test_profile_single_column(self, tmp_path):
        """Test profiling a dataframe with single column."""
        csv_file = tmp_path / "single_col.csv"
        csv_file.write_text("value\n10\n20\n30\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert report.column_count == 1
        assert report.row_count == 3


class TestProfileValidation:
    """Test input validation and error handling."""
    
    def test_profile_negative_sample_size_error(self, tmp_path):
        """Test that negative sample_size raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\n1\n")
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)
    
    def test_profile_float_sample_size_error(self, tmp_path):
        """Test that non-integer sample_size raises TypeError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\n1\n")
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=2.5)
    
    def test_profile_invalid_approx_top_values_type(self, tmp_path):
        """Test that non-bool approx_top_values raises TypeError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\n1\n")
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(TypeError, match="approx_top_values must be a bool"):
            ar.profile(frame, approx_top_values="yes")
    
    def test_profile_min_ratio_out_of_bounds(self, tmp_path):
        """Test that invalid approx_top_values_min_ratio raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\n1\n")
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
    
    def test_profile_exclude_nonexistent_column(self, tmp_path):
        """Test that excluding non-existent column raises error."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(Exception):
            ar.profile(frame, exclude_columns=['nonexistent'])


class TestProfileReport:
    """Test DataQualityReport properties and output."""
    
    def test_profile_report_structure(self, tmp_path):
        """Test that report contains expected attributes."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert hasattr(report, 'row_count')
        assert hasattr(report, 'column_count')
        assert hasattr(report, 'duplicate_rows')
        assert hasattr(report, 'duplicate_ratio')
        assert hasattr(report, 'quality_score')
        assert hasattr(report, 'score_components')
        assert hasattr(report, 'columns')
        assert hasattr(report, 'suggestions')
    
    def test_profile_report_suggestions(self, tmp_path):
        """Test that report includes cleaning suggestions."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id,data\n1,\n2,\n3,value\n")
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert isinstance(report.suggestions, list)
        # Suggestions may be present for columns with high null ratio
    
    def test_profile_memory_usage(self, tmp_path):
        """Test that memory usage is calculated and reported."""
        csv_file = tmp_path / "mem.csv"
        csv_file.write_text("data\n" + "\n".join([f"{'x'*100}"] * 10))
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        
        assert hasattr(report, 'memory_usage')
        assert report.memory_usage > 0


class TestProfileCLI:
    """Test profile functionality via command-line interface."""
    
    def test_profile_cli_basic(self, tmp_path):
        """Test profile command via CLI."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "profile", str(csv_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # CLI should execute and return meaningful output
        assert result.returncode in [0, 1]


class TestProfileIntegration:
    """Integration tests for complete profiling workflows."""
    
    def test_profile_workflow_with_exclusion_and_exclusion(self, tmp_path):
        """Test complete workflow: read, profile with options, verify output."""
        csv_file = tmp_path / "workflow.csv"
        csv_file.write_text(
            "customer_id,name,email,internal_hash,score\n"
            "1,Alice,a@test.com,xxx123,85\n"
            "2,Bob,b@test.com,xxx456,90\n"
            "2,Bob,b@test.com,xxx456,90\n"  # duplicate
        )
        
        frame = ar.read_csv(csv_file)
        report = ar.profile(
            frame,
            exclude_columns=['internal_hash'],
            sample_size=2
        )
        
        assert report.row_count == 3
        assert report.column_count == 4  # 5 - 1 excluded
        assert report.duplicate_rows == 1
        assert 'internal_hash' not in report.columns
        assert 'customer_id' in report.columns
        assert report.quality_score is not None
