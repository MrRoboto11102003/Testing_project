import pytest
import pandas as pd
import arnio as ar
import csv


@pytest.fixture
def tmp_csv(tmp_path):
    """Factory fixture for creating CSV files on disk."""
    def _make_csv(name, content):
        path = tmp_path / name
        path.write_text(content)
        return str(path)
    return _make_csv


class TestProfileBasic:
    """Test basic profile functionality with real CSV files."""
    
    def test_profile_simple_data(self, tmp_csv):
        """Profile a simple CSV with standard tabular data."""
        csv_path = tmp_csv("data.csv", "name,age,score\nAlice,30,95\nBob,25,87\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        assert report.row_count == 2
        assert report.column_count == 3
        assert "name" in report.columns
        assert "age" in report.columns
        assert "score" in report.columns
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0
    
    def test_profile_with_nulls_and_duplicates(self, tmp_csv):
        """Profile data containing missing values and duplicate rows."""
        csv_path = tmp_csv("data.csv", "id,val,cat\n1,10,A\n2,,B\n1,10,A\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.columns["val"].null_count >= 1
        assert report.duplicate_rows >= 1
        assert report.duplicate_ratio > 0.0
    
    def test_profile_report_summary_and_markdown(self, tmp_csv):
        """Test DataQualityReport.summary() and to_markdown() methods."""
        csv_path = tmp_csv("data.csv", "id,name,value\n1,Alice,10\n2,Bob,20\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        summary = report.summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0
        
        md_text = report.to_markdown()
        assert isinstance(md_text, str)
        assert len(md_text) > 0


class TestProfileExcludeColumns:
    """Test exclude_columns parameter functionality."""
    
    def test_exclude_single_column(self, tmp_csv):
        """Profile with one column excluded."""
        csv_path = tmp_csv("data.csv", "id,name,debug_col\n1,Alice,xyz\n2,Bob,abc\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame, exclude_columns=["debug_col"])
        
        assert report.column_count == 2
        assert "debug_col" not in report.columns
        assert "id" in report.columns
        assert "name" in report.columns
    
    def test_exclude_multiple_columns(self, tmp_csv):
        """Profile with multiple columns excluded."""
        csv_path = tmp_csv("data.csv", "id,name,temp1,temp2\n1,Alice,x,y\n2,Bob,a,b\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame, exclude_columns=["temp1", "temp2"])
        
        assert report.column_count == 2
        assert "temp1" not in report.columns
        assert "temp2" not in report.columns
    
    def test_exclude_nonexistent_column_raises_error(self, tmp_csv):
        """Raise error when exclude_columns references non-existent column."""
        csv_path = tmp_csv("data.csv", "id,name\n1,Alice\n2,Bob\n")
        frame = ar.read_csv(csv_path)
        
        with pytest.raises((ValueError, KeyError)):
            ar.profile(frame, exclude_columns=["nonexistent"])


class TestProfileParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_sample_size_negative_raises_error(self, tmp_csv):
        """Raise error for negative sample_size."""
        csv_path = tmp_csv("data.csv", "value\n1\n2\n3\n")
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError):
            ar.profile(frame, sample_size=-1)
    
    def test_sample_size_non_integer_raises_error(self, tmp_csv):
        """Raise error when sample_size is not an integer."""
        csv_path = tmp_csv("data.csv", "value\n1\n2\n")
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(TypeError):
            ar.profile(frame, sample_size=3.5)
    
    def test_approx_ratio_exceeds_bounds_raises_error(self, tmp_csv):
        """Raise error when approx_top_values_min_ratio outside [0, 1]."""
        csv_path = tmp_csv("data.csv", "value\n1\n2\n")
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.5)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=-0.1)
    
    def test_approx_sample_size_non_positive_raises_error(self, tmp_csv):
        """Raise error when approx_top_values_sample_size is not positive."""
        csv_path = tmp_csv("data.csv", "value\n1\n2\n")
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError, match="positive"):
            ar.profile(frame, approx_top_values_sample_size=0)


class TestProfilePandasIntegration:
    """Test profiling via pandas DataFrame integration."""
    
    def test_profile_via_pandas_accessor(self, tmp_csv):
        """Profile using df.arnio.profile() accessor method."""
        csv_path = tmp_csv("data.csv", "id,value,category\n1,10,A\n2,20,B\n3,30,A\n")
        df = pd.read_csv(csv_path)
        report = df.arnio.profile()
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert isinstance(report.summary(), dict)
    
    def test_profile_from_pandas_dataframe(self):
        """Profile a frame created from pandas DataFrame via ar.from_pandas()."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
            "salary": [50000.0, 60000.0, 55000.0]
        })
        
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert "name" in report.columns
        assert "age" in report.columns


class TestProfileEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_profile_all_columns_excluded(self, tmp_csv):
        """Profile when all columns are excluded."""
        csv_path = tmp_csv("data.csv", "id,name\n1,Alice\n2,Bob\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame, exclude_columns=["id", "name"])
        
        assert report.column_count == 0
        assert len(report.columns) == 0
    
    def test_profile_single_column_single_row(self, tmp_csv):
        """Profile minimal frame: single column and single data row."""
        csv_path = tmp_csv("data.csv", "value\n42\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        assert report.row_count == 1
        assert report.column_count == 1
        assert report.duplicate_rows == 0
        assert report.duplicate_ratio == 0.0
    
    def test_profile_all_nulls_column(self, tmp_csv):
        """Profile a frame with a column containing all null values."""
        csv_path = tmp_csv("data.csv", "id,empty_col,value\n1,,10\n2,,20\n3,,30\n")
        frame = ar.read_csv(csv_path)
        report = ar.profile(frame)
        
        assert report.columns["empty_col"].null_count == 3
        assert report.columns["id"].null_count == 0
    
    def test_profile_invalid_frame_type_raises_error(self):
        """Raise error when frame is not an ArFrame."""
        with pytest.raises((TypeError, AttributeError)):
            ar.profile("not a frame")
        
        with pytest.raises((TypeError, AttributeError)):
            ar.profile(123)
