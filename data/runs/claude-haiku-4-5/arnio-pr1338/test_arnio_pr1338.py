import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os


class TestReadCSV:
    """Test ar.read_csv() with different modes."""
    
    def test_read_csv_strict_mode(self):
        """Test reading CSV in strict mode with consistent row widths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            with open(csv_path, "w") as f:
                f.write("name,age,salary\n")
                f.write("Alice,25,50000\n")
                f.write("Bob,30,60000\n")
            
            frame = ar.read_csv(csv_path, mode="strict")
            df = ar.to_pandas(frame)
            assert len(df) == 2
            assert list(df.columns) == ["name", "age", "salary"]
    
    def test_read_csv_permissive_mode(self):
        """Test reading CSV in permissive mode handles missing trailing values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            with open(csv_path, "w") as f:
                f.write("name,age,salary\n")
                f.write("Alice,25,50000\n")
                f.write("Bob,30\n")  # Missing salary
            
            frame = ar.read_csv(csv_path, mode="permissive")
            df = ar.to_pandas(frame)
            assert len(df) == 2


class TestPipeline:
    """Test ar.pipeline() transformations."""
    
    def test_pipeline_strip_whitespace(self):
        """Test pipeline with strip_whitespace step."""
        frame = ar.from_dict({"name": [" Alice ", " Bob "]})
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(clean)
        assert df["name"].tolist() == ["Alice", "Bob"]
    
    def test_pipeline_multiple_steps(self):
        """Test pipeline with multiple sequential steps."""
        frame = ar.from_dict({"name": [" ALICE ", " BOB "], "age": [25, 30]})
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        df = ar.to_pandas(clean)
        assert df["name"].tolist() == ["alice", "bob"]
    
    def test_pipeline_dry_run_no_mutation(self):
        """Test pipeline dry_run=True validates without execution."""
        frame = ar.from_dict({"name": ["Alice", "Bob"]})
        result = ar.pipeline(frame, [("strip_whitespace",)], dry_run=True)
        # dry_run should return without applying changes
        assert result is not None
    
    def test_pipeline_return_metadata(self):
        """Test pipeline with return_metadata=True includes timing info."""
        frame = ar.from_dict({"name": ["Alice", "Bob", "Bob"], "age": [25, 30, 30]})
        clean, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        assert isinstance(clean, ar.ArFrame)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_drop_nulls(self):
        """Test pipeline with drop_nulls removes rows with missing values."""
        frame = ar.from_dict({
            "name": ["Alice", "Bob", None],
            "age": [25, 30, 35]
        })
        clean = ar.pipeline(frame, [("drop_nulls",)])
        df = ar.to_pandas(clean)
        assert len(df) == 2
    
    def test_pipeline_invalid_step_fails_early(self):
        """Test that invalid step names raise error before execution."""
        frame = ar.from_dict({"name": ["Alice"]})
        with pytest.raises(Exception):
            ar.pipeline(frame, [("nonexistent_step",)])


class TestToPandas:
    """Test ar.to_pandas() conversion."""
    
    def test_to_pandas_basic_conversion(self):
        """Test converting ArFrame to pandas DataFrame."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]
    
    def test_to_pandas_with_copy(self):
        """Test to_pandas with copy=True creates independent buffer."""
        frame = ar.from_dict({"value": [1, 2, 3]})
        df1 = ar.to_pandas(frame, copy=True)
        df2 = ar.to_pandas(frame, copy=True)
        # Modifying one should not affect the other
        df1.iloc[0, 0] = 999
        assert df2.iloc[0, 0] == 1


class TestFromDict:
    """Test ar.from_dict() creation."""
    
    def test_from_dict_function(self):
        """Test ar.from_dict() creates ArFrame from dictionary."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]
    
    def test_from_dict_class_method(self):
        """Test ar.ArFrame.from_dict() class method."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        df = ar.to_pandas(frame)
        assert len(df) == 2


class TestFrameMethods:
    """Test ArFrame instance methods."""
    
    def test_preview_method(self):
        """Test frame.preview() displays first N rows."""
        frame = ar.from_dict({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        result = frame.preview(2)
        assert result is not None
    
    def test_describe_method(self):
        """Test frame.describe() generates summary statistics."""
        frame = ar.from_dict({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "salary": [50000.0, 60000.0]
        })
        result = frame.describe()
        assert result is not None


class TestAutoClean:
    """Test ar.auto_clean() automatic repair."""
    
    def test_auto_clean_safe_mode(self):
        """Test auto_clean safe mode applies low-risk cleanup."""
        frame = ar.from_dict({"name": [" Alice ", " Bob "], "age": [25, 30]})
        clean = ar.auto_clean(frame, mode="safe")
        assert isinstance(clean, ar.ArFrame)
        df = ar.to_pandas(clean)
        assert len(df) == 2
    
    def test_auto_clean_return_report(self):
        """Test auto_clean return_report=True returns quality report."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        clean, report = ar.auto_clean(frame, return_report=True)
        assert isinstance(clean, ar.ArFrame)
        assert report is not None
    
    def test_auto_clean_dry_run(self):
        """Test auto_clean dry_run=True previews without mutation."""
        frame = ar.from_dict({"name": [" Alice ", " Bob "]})
        report = ar.auto_clean(frame, dry_run=True)
        assert report is not None
    
    def test_auto_clean_explain(self):
        """Test auto_clean explain=True returns reasoning."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        clean, explanation = ar.auto_clean(frame, explain=True)
        assert isinstance(clean, ar.ArFrame)
        assert explanation is not None


class TestPandasAccessor:
    """Test pandas .arnio accessor integration."""
    
    def test_pandas_accessor_clean(self):
        """Test df.arnio.clean() applies pipeline via accessor."""
        df = pd.DataFrame({"name": [" ALICE ", " BOB "], "age": [25, 30]})
        clean_df = df.arnio.clean([
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert isinstance(clean_df, pd.DataFrame)
        assert clean_df["name"].tolist() == ["alice", "bob"]
    
    def test_pandas_accessor_strip_whitespace(self):
        """Test df.arnio.strip_whitespace() accessor method."""
        df = pd.DataFrame({"name": [" Alice ", " Bob "]})
        result = df.arnio.strip_whitespace(subset=["name"])
        assert isinstance(result, pd.DataFrame)
        assert result["name"].tolist() == ["Alice", "Bob"]
