import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestProfileBasic:
    """Test basic profiling functionality."""
    
    def test_profile_returns_data_quality_report(self):
        """Profile returns a DataQualityReport with documented attributes."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert hasattr(report, "row_count")
        assert hasattr(report, "column_count")
        assert hasattr(report, "columns")
        assert report.row_count == 3
        assert report.column_count == 2
    
    def test_profile_detects_nulls(self):
        """Profile correctly counts null values in columns."""
        df = pd.DataFrame({
            "col1": [1, 2, None, 4, 5],
            "col2": ["a", None, None, "d", "e"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.columns["col1"].null_count == 1
        assert report.columns["col2"].null_count == 2
    
    def test_profile_empty_frame(self):
        """Profile handles empty frames gracefully."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=str)})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 2


class TestProfileExcludeColumns:
    """Test exclude_columns parameter."""
    
    def test_exclude_single_column(self):
        """Exclude a single column from profiling."""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "secret": ["x", "y"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["secret"])
        
        assert report.column_count == 2
        assert "id" in report.columns
        assert "name" in report.columns
        assert "secret" not in report.columns
    
    def test_exclude_multiple_columns(self):
        """Exclude multiple columns from profiling."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, exclude_columns=["b", "d"])
        
        assert report.column_count == 2
        assert set(report.columns.keys()) == {"a", "c"}


class TestProfileSampleSize:
    """Test sample_size parameter controls sample collection."""
    
    def test_sample_size_default(self):
        """Default sample_size is respected."""
        df = pd.DataFrame({"col": ["a", "b", "c", "d", "e"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)  # default sample_size=5
        
        col_profile = report.columns["col"]
        assert col_profile.sample_values is not None
    
    def test_sample_size_zero_skips_samples(self):
        """Sample size of 0 skips sample value collection."""
        df = pd.DataFrame({"col": ["a", "b", "c", "d", "e"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=0)
        
        col_profile = report.columns["col"]
        # With sample_size=0, sample_values should be empty or minimal
        assert len(col_profile.sample_values) == 0
    
    def test_sample_size_custom_respected(self):
        """Custom sample_size is respected in profiling."""
        df = pd.DataFrame({"col": ["a", "b", "c", "d", "e", "f", "g", "h"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        col_profile = report.columns["col"]
        assert len(col_profile.sample_values) <= 2


class TestProfileDataTypes:
    """Test profiling different data types produces correct statistics."""
    
    def test_profile_numeric_column_statistics(self):
        """Numeric columns include mean, std, min, max, quantiles."""
        df = pd.DataFrame({"age": [20, 25, 30, 35, 40]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col_profile = report.columns["age"]
        assert col_profile.dtype == "int64"
        assert col_profile.mean == 30.0
        assert col_profile.min == 20
        assert col_profile.max == 40
        assert hasattr(col_profile, "std")
        assert hasattr(col_profile, "q25")
        assert hasattr(col_profile, "q50")
    
    def test_profile_string_column_top_values(self):
        """String columns include top values with counts."""
        df = pd.DataFrame({"city": ["NYC", "LA", "NYC", "Chicago", "NYC", "LA"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col_profile = report.columns["city"]
        assert col_profile.dtype == "string"
        assert col_profile.top_values is not None
        assert len(col_profile.top_values) > 0


class TestProfileDuplicates:
    """Test duplicate row detection."""
    
    def test_profile_detects_duplicate_rows(self):
        """Profile detects and reports duplicate rows."""
        df = pd.DataFrame({"id": [1, 2, 2, 3], "name": ["a", "b", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.duplicate_rows >= 1
        assert report.duplicate_ratio > 0.0
    
    def test_profile_no_duplicates(self):
        """Profile reports zero duplicates for unique data."""
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0


class TestProfileQualityScore:
    """Test quality score calculation."""
    
    def test_quality_score_high_for_clean_data(self):
        """Quality score is high for clean data."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.quality_score >= 80.0
        assert report.quality_score <= 100.0
    
    def test_quality_score_lower_with_issues(self):
        """Quality score is lower with nulls and duplicates."""
        df = pd.DataFrame({"id": [1, 1, None, 3], "name": ["a", "a", None, "d"]})
        frame = ar.from_pandas(df)
        report_messy = ar.profile(frame)
        
        df_clean = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})
        frame_clean = ar.from_pandas(df_clean)
        report_clean = ar.profile(frame_clean)
        
        assert report_messy.quality_score < report_clean.quality_score


class TestProfileReportMethods:
    """Test DataQualityReport public methods."""
    
    def test_report_summary_method(self):
        """Report.summary() returns high-level dictionary."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        summary = report.summary()
        assert isinstance(summary, dict)
    
    def test_report_to_dict_method(self):
        """Report.to_dict() exports full report as dictionary."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert len(report_dict) > 0
    
    def test_report_to_markdown_method(self):
        """Report.to_markdown() produces markdown output."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0


class TestProfileUniqueAndRatio:
    """Test unique value detection and ratios."""
    
    def test_unique_count_calculated(self):
        """Profile calculates unique value counts per column."""
        df = pd.DataFrame({"id": [1, 1, 2, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.columns["id"].unique_count == 3
    
    def test_unique_ratio_calculated(self):
        """Profile calculates unique_ratio (unique / non-null)."""
        df = pd.DataFrame({"id": [1, 1, 1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col_profile = report.columns["id"]
        expected_ratio = 3 / 5  # 3 unique out of 5 rows
        assert col_profile.unique_ratio == pytest.approx(expected_ratio)


class TestProfileOutliers:
    """Test outlier detection in numeric columns."""
    
    def test_outlier_detection_numeric(self):
        """Profile detects outliers in numeric columns."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100 is an outlier
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        col_profile = report.columns["value"]
        assert hasattr(col_profile, "outlier_count")
        assert hasattr(col_profile, "outlier_ratio")
        assert hasattr(col_profile, "outlier_lower_bound")
        assert hasattr(col_profile, "outlier_upper_bound")
