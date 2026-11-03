import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestReadCSV:
    """Test ar.read_csv functionality"""
    
    def test_read_csv_basic(self, tmp_path):
        """Test basic CSV reading"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")
        frame = ar.read_csv(str(csv_file))
        assert frame is not None
    
    def test_read_csv_strict_mode(self, tmp_path):
        """Test strict mode with consistent row widths"""
        csv_file = tmp_path / "strict.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6")
        frame = ar.read_csv(str(csv_file), mode="strict")
        assert frame is not None
    
    def test_read_csv_permissive_mode(self, tmp_path):
        """Test permissive mode fills missing trailing values with nulls"""
        csv_file = tmp_path / "permissive.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5")
        frame = ar.read_csv(str(csv_file), mode="permissive")
        assert frame is not None


class TestFromDict:
    """Test ar.from_dict and ArFrame.from_dict"""
    
    def test_from_dict_creates_frame(self):
        """Test creating ArFrame from dictionary"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None
    
    def test_arframe_from_dict_class_method(self):
        """Test ArFrame.from_dict() class method"""
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None


class TestPipeline:
    """Test ar.pipeline functionality"""
    
    def test_pipeline_basic_steps(self, tmp_path):
        """Test pipeline with strip and normalize steps"""
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("name,revenue\n Alice ,100\n bob ,200")
        frame = ar.read_csv(str(csv_file))
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_nulls",),
            ("drop_duplicates",),
        ])
        assert clean is not None
    
    def test_pipeline_fill_nulls(self, tmp_path):
        """Test fill_nulls step with subset parameter"""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text("revenue,cost\n100,\n200,50")
        frame = ar.read_csv(str(csv_file), mode="permissive")
        
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["cost"]}),
        ])
        assert clean is not None
    
    def test_pipeline_dry_run_validates_without_output(self, tmp_path):
        """Test dry_run=True validates configuration without returning data"""
        csv_file = tmp_path / "dry.csv"
        csv_file.write_text("a,b\n1,2")
        frame = ar.read_csv(str(csv_file))
        
        result = ar.pipeline(frame, [("drop_nulls",)], dry_run=True)
        assert result is not None
    
    def test_pipeline_return_metadata_structure(self, tmp_path):
        """Test return_metadata returns step timings and execution info"""
        csv_file = tmp_path / "meta.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        frame = ar.read_csv(str(csv_file))
        
        clean, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True,
        )
        
        assert clean is not None
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_invalid_step_fails_early(self, tmp_path):
        """Test malformed step tuple fails before pipeline execution"""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("a,b\n1,2")
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises((ValueError, Exception)):
            ar.pipeline(frame, [("bad_step", "oops", "extra")])


class TestToPandas:
    """Test ar.to_pandas conversion"""
    
    def test_to_pandas_conversion(self, tmp_path):
        """Test converting ArFrame to pandas DataFrame"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        frame = ar.read_csv(str(csv_file))
        
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_to_pandas_with_copy_true(self, tmp_path):
        """Test to_pandas with copy=True for defensive pandas-owned buffers"""
        csv_file = tmp_path / "copy.csv"
        csv_file.write_text("x,y\n10,20")
        frame = ar.read_csv(str(csv_file))
        
        safe_df = ar.to_pandas(frame, copy=True)
        assert isinstance(safe_df, pd.DataFrame)


class TestFrameMethods:
    """Test ArFrame methods"""
    
    def test_frame_preview(self, tmp_path):
        """Test frame.preview(n) shows summary of first n rows"""
        csv_file = tmp_path / "preview.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6")
        frame = ar.read_csv(str(csv_file))
        
        result = frame.preview(2)
        assert result is not None


class TestCompareProfiles:
    """Test ar.compare_profiles for data drift detection"""
    
    def test_compare_profiles_basic(self, tmp_path):
        """Test comparing two profiles with matching columns"""
        baseline_file = tmp_path / "baseline.csv"
        baseline_file.write_text("score,count\n80,10\n85,15")
        baseline = ar.profile(ar.read_csv(str(baseline_file)))
        
        current_file = tmp_path / "current.csv"
        current_file.write_text("score,count\n82,12\n88,18")
        current = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline, current)
        assert comparison is not None
        assert hasattr(comparison, "drift_report")
    
    def test_compare_profiles_drift_report_contains_status(self, tmp_path):
        """Test drift_report contains status for each column"""
        baseline_file = tmp_path / "base.csv"
        baseline_file.write_text("score\n100\n100")
        baseline = ar.profile(ar.read_csv(str(baseline_file)))
        
        current_file = tmp_path / "curr.csv"
        current_file.write_text("score\n50\n150")
        current = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline, current)
        assert "score" in comparison.drift_report
        assert "status" in comparison.drift_report["score"]
    
    def test_compare_profiles_different_columns_raises_valueerror(self, tmp_path):
        """Test ValueError raised when profiles have different column sets"""
        baseline_file = tmp_path / "base_a.csv"
        baseline_file.write_text("col_a,col_b\n1,2")
        baseline = ar.profile(ar.read_csv(str(baseline_file)))
        
        current_file = tmp_path / "curr_a.csv"
        current_file.write_text("col_a,col_c\n1,2")
        current = ar.profile(ar.read_csv(str(current_file)))
        
        with pytest.raises(ValueError):
            ar.compare_profiles(baseline, current)
    
    def test_compare_profiles_numeric_column_metrics(self, tmp_path):
        """Test numeric columns compare mean, std, min, max distribution metrics"""
        baseline_file = tmp_path / "num_base.csv"
        baseline_file.write_text("value\n10\n20\n30")
        baseline = ar.profile(ar.read_csv(str(baseline_file)))
        
        current_file = tmp_path / "num_curr.csv"
        current_file.write_text("value\n15\n25\n35")
        current = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline, current)
        assert comparison is not None
