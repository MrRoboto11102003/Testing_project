import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os


class TestReadCSV:
    """Test CSV reading with strict and permissive modes"""
    
    def test_read_csv_permissive_fills_missing_values(self):
        """Test permissive mode fills missing trailing values with nulls"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5\n")
            f.flush()
            
            frame = ar.read_csv(f.name, mode="permissive")
            df = ar.to_pandas(frame)
            
            assert df.shape == (2, 3)
            assert pd.isna(df.iloc[1, 2])
            os.unlink(f.name)
    
    def test_read_csv_strict_mode_fails_on_inconsistent_rows(self):
        """Test strict mode fails on inconsistent row widths"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5\n")
            f.flush()
            
            with pytest.raises(Exception):
                ar.read_csv(f.name, mode="strict")
            os.unlink(f.name)


class TestFromDictAndToPandas:
    """Test ArFrame creation and pandas conversion"""
    
    def test_from_dict_function_and_class_method(self):
        """Test ar.from_dict() function and ar.ArFrame.from_dict() class method"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        
        frame1 = ar.from_dict(data)
        frame2 = ar.ArFrame.from_dict(data)
        
        df1 = ar.to_pandas(frame1)
        df2 = ar.to_pandas(frame2)
        
        assert list(df1.columns) == ["name", "age"]
        assert len(df1) == 2
        assert len(df2) == 2
    
    def test_to_pandas_with_copy_parameter(self):
        """Test to_pandas copy=False and copy=True"""
        data = {"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}
        frame = ar.from_dict(data)
        
        df_no_copy = ar.to_pandas(frame, copy=False)
        df_copy = ar.to_pandas(frame, copy=True)
        
        assert isinstance(df_no_copy, pd.DataFrame)
        assert isinstance(df_copy, pd.DataFrame)
        assert len(df_no_copy) == 3 and len(df_copy) == 3


class TestPipelineSteps:
    """Test pipeline step transformations"""
    
    def test_pipeline_strip_whitespace_normalize_case_drop_duplicates(self):
        """Test strip_whitespace, normalize_case, and drop_duplicates steps"""
        data = {"name": [" ALICE ", " BOB ", " ALICE "], "age": [25, 30, 25]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert df["name"].iloc[0] == "alice"
        assert df["name"].iloc[1] == "bob"
    
    def test_pipeline_fill_nulls_and_drop_nulls(self):
        """Test fill_nulls with value/subset and drop_nulls steps"""
        data = {"revenue": [100.0, None, 300.0], "name": ["A", "B", None]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["revenue"]}),
            ("drop_nulls",),
        ])
        
        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert df["revenue"].iloc[0] == 100.0
        assert df["revenue"].iloc[1] == 0.0
    
    def test_pipeline_drop_duplicates_removes_rows(self):
        """Test drop_duplicates removes exact duplicate rows"""
        data = {"x": [1, 1, 2], "y": [3, 3, 4]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [("drop_duplicates",)])
        df = ar.to_pandas(clean)
        
        assert len(df) == 2


class TestPipelineValidation:
    """Test pipeline validation and configuration"""
    
    def test_pipeline_dry_run_validates_without_executing(self):
        """Test dry_run=True validates steps without transforming data"""
        data = {"a": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        result = ar.pipeline(frame, [
            ("drop_nulls",),
        ], dry_run=True)
        
        assert result is None
    
    def test_pipeline_return_metadata_provides_execution_info(self):
        """Test return_metadata=True returns execution metadata"""
        data = {"val": [1, 2, 1]}
        frame = ar.from_dict(data)
        
        clean, metadata = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], return_metadata=True)
        
        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_validates_steps_before_execution(self):
        """Test malformed step tuple raises early before any execution"""
        data = {"a": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        with pytest.raises(Exception):
            ar.pipeline(frame, [
                ("strip_whitespace",),
                ("bad_step", "oops", "extra"),
            ])


class TestFrameMethodsAndAccessor:
    """Test ArFrame methods and pandas accessor integration"""
    
    def test_frame_preview_and_describe(self):
        """Test frame.preview() and frame.describe() methods"""
        data = {"a": range(10), "b": range(10, 20)}
        frame = ar.from_dict(data)
        
        preview = frame.preview(5)
        desc = frame.describe()
        
        assert preview is not None
        assert desc is not None
    
    def test_pandas_arnio_accessor_method_chaining(self):
        """Test pandas .arnio accessor supports method chaining"""
        df = pd.DataFrame({
            "name": [" ALICE ", " BOB "],
        })
        
        result = (df.arnio
                  .strip_whitespace(subset=["name"])
                  .arnio
                  .normalize_case(case_type="lower", subset=["name"]))
        
        assert isinstance(result, pd.DataFrame)
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[1] == "bob"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
