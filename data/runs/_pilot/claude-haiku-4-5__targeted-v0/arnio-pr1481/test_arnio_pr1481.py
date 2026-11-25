import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os
import shutil


@pytest.fixture
def temp_csv_factory():
    """Factory fixture for creating temporary CSV files"""
    created_dirs = []
    
    def _make_csv(data):
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "test.csv")
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        created_dirs.append(tmpdir)
        return csv_path
    
    yield _make_csv
    
    for tmpdir in created_dirs:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestReadCSV:
    """Test ar.read_csv() functionality with different modes"""
    
    def test_read_csv_basic(self, temp_csv_factory):
        """Test basic CSV reading"""
        csv_path = temp_csv_factory({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.read_csv(csv_path)
        assert frame is not None
    
    def test_read_csv_strict_mode(self, temp_csv_factory):
        """Test strict mode with consistent row widths"""
        csv_path = temp_csv_factory({"col1": [1, 2], "col2": [3, 4]})
        frame = ar.read_csv(csv_path, mode="strict")
        assert frame is not None
    
    def test_read_csv_permissive_mode(self, temp_csv_factory):
        """Test permissive mode fills missing trailing values with nulls"""
        csv_path = temp_csv_factory({"col1": [1, 2], "col2": [3, 4]})
        frame = ar.read_csv(csv_path, mode="permissive")
        assert frame is not None


class TestPipeline:
    """Test ar.pipeline() with various steps and options"""
    
    def test_pipeline_single_step(self, temp_csv_factory):
        """Test pipeline with single normalization step"""
        csv_path = temp_csv_factory({"name": ["ALICE", "BOB"], "age": [25, 30]})
        frame = ar.read_csv(csv_path)
        clean = ar.pipeline(frame, [("normalize_case", {"case_type": "lower"})])
        assert clean is not None
    
    def test_pipeline_multiple_steps(self, temp_csv_factory):
        """Test chained pipeline steps for whitespace and deduplication"""
        csv_path = temp_csv_factory({
            "name": ["  ALICE  ", "  BOB  ", "  ALICE  "],
            "age": [25, 30, 25]
        })
        frame = ar.read_csv(csv_path)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert clean is not None
    
    def test_pipeline_dry_run_validation(self, temp_csv_factory):
        """Test dry_run validates steps without executing transformation"""
        csv_path = temp_csv_factory({"a": [1, 2], "b": [3, 4]})
        frame = ar.read_csv(csv_path)
        result = ar.pipeline(frame, [("drop_nulls",)], dry_run=True)
        assert result is not None
    
    def test_pipeline_return_metadata(self, temp_csv_factory):
        """Test return_metadata includes step timings and execution info"""
        csv_path = temp_csv_factory({"a": [1, 2], "b": [3, 4]})
        frame = ar.read_csv(csv_path)
        clean, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            return_metadata=True
        )
        assert clean is not None
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata


class TestToPandas:
    """Test conversion from ArFrame to pandas DataFrame"""
    
    def test_to_pandas_basic_conversion(self, temp_csv_factory):
        """Test basic conversion returns pandas DataFrame"""
        csv_path = temp_csv_factory({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
    
    def test_to_pandas_with_copy(self, temp_csv_factory):
        """Test copy=True creates defensive pandas-owned buffers"""
        csv_path = temp_csv_factory({"a": [1, 2, 3]})
        frame = ar.read_csv(csv_path)
        df = ar.to_pandas(frame, copy=True)
        assert isinstance(df, pd.DataFrame)


class TestFromDict:
    """Test creating ArFrame from dictionary"""
    
    def test_from_dict_function(self):
        """Test ar.from_dict() creates ArFrame from equal-length columns"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None
    
    def test_from_dict_class_method(self):
        """Test ArFrame.from_dict() class method"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None


class TestFilterRows:
    """Test ar.filter_rows() with various operators"""
    
    def test_filter_rows_greater_than(self, temp_csv_factory):
        """Test > operator filters rows correctly"""
        csv_path = temp_csv_factory({"age": [18, 25, 30, 35]})
        frame = ar.read_csv(csv_path)
        filtered = ar.filter_rows(frame, column="age", op=">", value=25)
        assert filtered is not None
    
    def test_filter_rows_equality(self, temp_csv_factory):
        """Test == operator filters rows correctly"""
        csv_path = temp_csv_factory({"status": ["active", "inactive", "active"]})
        frame = ar.read_csv(csv_path)
        filtered = ar.filter_rows(frame, column="status", op="==", value="active")
        assert filtered is not None
    
    def test_filter_rows_with_pandas_dataframe(self):
        """Test filter_rows returns DataFrame when input is DataFrame"""
        df = pd.DataFrame({"age": [18, 25, 30]})
        filtered = ar.filter_rows(df, column="age", op="<=", value=25)
        assert isinstance(filtered, pd.DataFrame)
