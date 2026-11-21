import pytest
import pandas as pd
import arnio as ar
import tempfile
from pathlib import Path


class TestProfileBasic:
    """Test basic profiling workflow."""

    def test_profile_simple_csv(self, tmp_path):
        """Profile a simple well-formed CSV file."""
        csv_file = tmp_path / "simple.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,NYC\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert set(report.columns.keys()) == {"name", "age", "city"}
        assert report.duplicate_rows == 0
        assert 0 <= report.quality_score <= 1

    def test_profile_with_nulls(self, tmp_path):
        """Profile CSV containing null values."""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text(
            "name,score,active\nAlice,95,yes\nBob,,no\n,88,yes\n"
        )
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.columns["name"].null_count >= 1
        assert report.columns["score"].null_count >= 1

    def test_profile_with_duplicates(self, tmp_path):
        """Profile CSV with duplicate rows."""
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text("id,value\n1,A\n1,A\n2,B\n1,A\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 4
        assert report.duplicate_rows == 2
        assert report.duplicate_ratio > 0.0
        assert report.duplicate_ratio < 1.0

    def test_profile_exclude_columns(self, tmp_path):
        """Profile with column exclusion."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,debug,value\n1,Alice,xyz,100\n2,Bob,abc,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, exclude_columns=["debug"])
        
        assert report.column_count == 3
        assert "debug" not in report.columns
        assert set(report.columns.keys()) == {"id", "name", "value"}

    def test_profile_exclude_multiple_columns(self, tmp_path):
        """Profile with multiple column exclusions."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d,e\n1,2,3,4,5\n6,7,8,9,10\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, exclude_columns=["b", "d", "e"])
        
        assert report.column_count == 2
        assert set(report.columns.keys()) == {"a", "c"}


class TestProfileParameters:
    """Test profiling parameters."""

    def test_profile_sample_size_zero(self, tmp_path):
        """Profile with sample_size=0."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2\nval3,val4\nval5,val6\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=0)
        
        assert report.row_count == 3
        assert len(report.columns) == 2

    def test_profile_quality_score_components(self, tmp_path):
        """Profile report includes quality score and components."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,A\n2,B\n3,C\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert hasattr(report, "quality_score")
        assert isinstance(report.quality_score, (int, float))
        assert 0 <= report.quality_score <= 1
        assert hasattr(report, "score_components")
        assert isinstance(report.score_components, dict)

    def test_profile_approx_top_values_enabled(self, tmp_path):
        """Profile with approximation enabled for high-cardinality columns."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w") as f:
            f.write("id,category\n")
            for i in range(1200):
                cat = f"cat_{i % 60}"
                f.write(f"{i},{cat}\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=50,
            approx_top_values_min_ratio=0.01,
        )
        
        assert report.row_count == 1200
        assert len(report.columns) == 2


class TestProfileEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_profile_empty_frame(self, tmp_path):
        """Profile a frame with headers only."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col1,col2,col3\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 3
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0

    def test_profile_single_row(self, tmp_path):
        """Profile data with single row."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 1
        assert report.column_count == 3

    def test_profile_all_nulls_column(self, tmp_path):
        """Profile with a column of all null values."""
        csv_file = tmp_path / "all_nulls.csv"
        csv_file.write_text("col1,col2,col3\n1,,A\n2,,B\n3,,C\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.column_count == 3
        assert report.columns["col2"].null_count == 3

    def test_profile_all_duplicates(self, tmp_path):
        """Profile where all rows are identical."""
        csv_file = tmp_path / "all_dups.csv"
        csv_file.write_text("id,value\n1,X\n1,X\n1,X\n1,X\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 4
        assert report.duplicate_rows == 3
        assert report.duplicate_ratio == 0.75


class TestProfileReportStructure:
    """Test structure and content of profiling output."""

    def test_profile_report_summary_method(self, tmp_path):
        """DataQualityReport.summary() returns dict with data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        summary = report.summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_profile_column_profiles_contain_metadata(self, tmp_path):
        """Column profiles contain expected attributes."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,A\n2,B\n3,C\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        col_profile = report.columns["col1"]
        assert hasattr(col_profile, "null_count")
        assert hasattr(col_profile, "dtype")
        assert col_profile.null_count == 0

    def test_profile_report_suggestions_list(self, tmp_path):
        """Profile report includes suggestions list."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,val\n  Alice  ,  10  \n  Bob  ,  20  \n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert hasattr(report, "suggestions")
        assert isinstance(report.suggestions, list)


class TestProfileIntegration:
    """Integration tests with other arnio operations."""

    def test_profile_pipeline_integration(self, tmp_path):
        """Profile before and after cleaning pipeline."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "name,score\n  Alice  ,  95  \n  Bob  ,  88  \n  Alice  ,  95  \n"
        )
        
        frame = ar.read_csv(str(csv_file))
        report1 = ar.profile(frame)
        assert report1.duplicate_rows > 0
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        
        report2 = ar.profile(clean)
        assert report2.row_count < report1.row_count
        assert report2.duplicate_rows == 0

    def test_profile_pandas_accessor(self, tmp_path):
        """Profile using pandas .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")
        
        df = pd.read_csv(str(csv_file))
        report = df.arnio.profile()
        
        assert report.row_count == 2
        assert report.column_count == 3
        assert "id" in report.columns

    def test_profile_pandas_roundtrip(self, tmp_path):
        """Profile ArFrame, convert to pandas, verify consistency."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,A\n2,B\n3,C\n4,D\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        df = ar.to_pandas(frame)
        assert len(df) == report.row_count
        assert len(df.columns) == report.column_count


class TestProfileErrorHandling:
    """Test error handling and validation."""

    def test_profile_error_nonexistent_excluded_column(self, tmp_path):
        """Profile raises error when excluded column does not exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises((ValueError, KeyError)):
            ar.profile(frame, exclude_columns=["nonexistent"])

    def test_profile_error_negative_sample_size(self, tmp_path):
        """Profile raises error for negative sample_size."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises((ValueError, TypeError)):
            ar.profile(frame, sample_size=-1)

    def test_profile_error_invalid_approx_ratio_bounds(self, tmp_path):
        """Profile raises error for approximation ratio outside [0, 1]."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
        
        with pytest.raises(ValueError):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)

    def test_profile_error_nonpositive_approx_sample_size(self, tmp_path):
        """Profile raises error for non-positive approx sample size."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError):
            ar.profile(frame, approx_top_values_sample_size=0)
        
        with pytest.raises(ValueError):
            ar.profile(frame, approx_top_values_sample_size=-5)
