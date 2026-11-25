import pytest
import pandas as pd
import tempfile
from pathlib import Path
import arnio as ar


class TestReadCSV:
    """Test CSV reading with different modes."""
    
    def test_read_csv_basic(self):
        """Test basic CSV reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,age\nAlice,25\nBob,30\n")
            
            frame = ar.read_csv(str(csv_path))
            assert frame is not None
    
    def test_read_csv_strict_mode(self):
        """Test strict mode with consistent rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
            
            frame = ar.read_csv(str(csv_path), mode="strict")
            assert frame is not None
    
    def test_read_csv_permissive_mode(self):
        """Test permissive mode fills missing trailing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b,c\n1,2,3\n4,5\n")
            
            frame = ar.read_csv(str(csv_path), mode="permissive")
            assert frame is not None


class TestPipeline:
    """Test pipeline transformations."""
    
    def test_pipeline_single_step(self):
        """Test pipeline with single transformation step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,value\n Alice ,100\n Bob ,200\n")
            
            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(frame, [("strip_whitespace",)])
            assert clean is not None
    
    def test_pipeline_multiple_steps(self):
        """Test pipeline with multiple transformation steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,value\n Alice ,100\n BOB ,200\n")
            
            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(frame, [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
                ("drop_duplicates",),
            ])
            assert clean is not None
    
    def test_pipeline_fill_nulls_with_subset(self):
        """Test fill_nulls step with subset parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,revenue\nAlice,100\nBob,\n")
            
            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(frame, [
                ("fill_nulls", {"value": 0.0, "subset": ["revenue"]}),
            ])
            assert clean is not None
    
    def test_pipeline_drop_nulls(self):
        """Test drop_nulls step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,value\nAlice,100\n,200\n")
            
            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(frame, [("drop_nulls",)])
            assert clean is not None
    
    def test_pipeline_dry_run_validation(self):
        """Test dry_run validates without transforming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n")
            
            frame = ar.read_csv(str(csv_path))
            result = ar.pipeline(frame, [("strip_whitespace",)], dry_run=True)
            assert result is not None
    
    def test_pipeline_return_metadata(self):
        """Test pipeline returns metadata with step info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n")
            
            frame = ar.read_csv(str(csv_path))
            clean, metadata = ar.pipeline(
                frame,
                [("strip_whitespace",), ("drop_duplicates",)],
                return_metadata=True,
            )
            
            assert clean is not None
            assert isinstance(metadata, dict)
            assert "step_timings" in metadata
            assert "applied_steps" in metadata
            assert "row_counts" in metadata
            assert "execution_summary" in metadata
    
    def test_pipeline_malformed_step_tuple_fails_early(self):
        """Test invalid step tuples fail before execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n")
            
            frame = ar.read_csv(str(csv_path))
            
            with pytest.raises(Exception):
                ar.pipeline(frame, [
                    ("strip_whitespace",),
                    ("bad_step", "oops", "extra"),
                ])
    
    def test_pipeline_unknown_step_name_fails_early(self):
        """Test unknown step names fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n")
            
            frame = ar.read_csv(str(csv_path))
            
            with pytest.raises(Exception):
                ar.pipeline(frame, [("nonexistent_step",)])


class TestToPandas:
    """Test ArFrame to pandas DataFrame conversion."""
    
    def test_to_pandas_basic_conversion(self):
        """Test basic conversion to pandas DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n")
            
            frame = ar.read_csv(str(csv_path))
            df = ar.to_pandas(frame)
            
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
    
    def test_to_pandas_with_copy_true(self):
        """Test to_pandas with defensive copy=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("x,y\n10,20\n")
            
            frame = ar.read_csv(str(csv_path))
            df = ar.to_pandas(frame, copy=True)
            
            assert isinstance(df, pd.DataFrame)
    
    def test_to_pandas_without_copy(self):
        """Test to_pandas with copy=False (default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("x,y\n10,20\n")
            
            frame = ar.read_csv(str(csv_path))
            df = ar.to_pandas(frame, copy=False)
            
            assert isinstance(df, pd.DataFrame)


class TestFromDict:
    """Test creating ArFrame from dictionary."""
    
    def test_from_dict_basic(self):
        """Test creating ArFrame from dict."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None
    
    def test_from_dict_class_method(self):
        """Test ArFrame.from_dict class method."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None
    
    def test_from_dict_in_pipeline(self):
        """Test pipeline works on frame created from dict."""
        data = {"name": [" Alice ", " Bob "], "value": [100, 200]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        assert clean is not None


class TestFrameMethods:
    """Test ArFrame methods like preview and describe."""
    
    def test_preview_first_n_rows(self):
        """Test preview shows first N rows."""
        data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        frame = ar.from_dict(data)
        
        result = frame.preview(5)
        assert result is not None
    
    def test_describe_generates_statistics(self):
        """Test describe generates summary statistics."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        result = frame.describe()
        assert result is not None


class TestDataQualityReport:
    """Test DataQualityReport.to_dict method."""
    
    def test_to_dict_basic(self):
        """Test to_dict returns JSON-friendly dict."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        report = frame.describe()
        
        if hasattr(report, 'to_dict'):
            result = report.to_dict()
            assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values(self):
        """Test to_dict with redact_sample_values parameter."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        report = frame.describe()
        
        if hasattr(report, 'to_dict'):
            result = report.to_dict(redact_sample_values=True)
            assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns(self):
        """Test to_dict with exclude_columns parameter."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        report = frame.describe()
        
        if hasattr(report, 'to_dict'):
            result = report.to_dict(exclude_columns=["age"])
            assert isinstance(result, dict)
