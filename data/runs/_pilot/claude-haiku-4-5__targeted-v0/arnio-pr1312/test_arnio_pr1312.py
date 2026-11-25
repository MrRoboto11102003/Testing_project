import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json
import tempfile
import os


@pytest.fixture
def temp_csv_clean():
    """Create a temporary clean CSV file."""
    content = """name,age,salary
Alice,25,50000
Bob,30,60000
Charlie,35,70000
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_messy():
    """Create a temporary CSV with inconsistent row widths."""
    content = """name,age,salary
Alice,25,50000
Bob,30
Charlie,35,70000,extra
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    return ar.from_dict({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000.0, 60000.0, 70000.0]
    })


class TestReadCSV:
    """Test CSV reading with different modes."""
    
    def test_read_csv_basic(self, temp_csv_clean):
        """Test basic CSV reading."""
        frame = ar.read_csv(temp_csv_clean)
        assert frame is not None
    
    def test_read_csv_strict_mode(self, temp_csv_clean):
        """Test CSV reading in strict mode."""
        frame = ar.read_csv(temp_csv_clean, mode="strict")
        assert frame is not None
    
    def test_read_csv_permissive_mode(self, temp_csv_messy):
        """Test CSV reading in permissive mode fills missing values."""
        frame = ar.read_csv(temp_csv_messy, mode="permissive")
        assert frame is not None
    
    def test_read_csv_strict_fails_inconsistent(self, temp_csv_messy):
        """Test strict mode fails on inconsistent row widths."""
        with pytest.raises(Exception):
            ar.read_csv(temp_csv_messy, mode="strict")


class TestPipeline:
    """Test pipeline transformation execution."""
    
    def test_pipeline_single_step(self, sample_frame):
        """Test pipeline with single transformation."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert result is not None
    
    def test_pipeline_multiple_steps(self, sample_frame):
        """Test pipeline with multiple chained steps."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert result is not None
    
    def test_pipeline_step_with_kwargs(self, sample_frame):
        """Test pipeline step with parameter dictionary."""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert result is not None
    
    def test_pipeline_dry_run_validation(self, sample_frame):
        """Test dry_run validates pipeline without executing."""
        ar.pipeline(sample_frame, [("drop_nulls",)], dry_run=True)
    
    def test_pipeline_return_metadata(self, sample_frame):
        """Test pipeline returns execution metadata."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True,
        )
        assert result is not None
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_invalid_step_fails_early(self, sample_frame):
        """Test invalid step name raises error before execution."""
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("nonexistent_step",),
            ])
    
    def test_pipeline_malformed_tuple_fails_early(self, sample_frame):
        """Test malformed step tuple raises error before execution."""
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [
                ("step", "extra", "args"),
            ])


class TestToPandas:
    """Test conversion from ArFrame to pandas DataFrame."""
    
    def test_to_pandas_basic_conversion(self, sample_frame):
        """Test basic conversion to pandas DataFrame."""
        df = ar.to_pandas(sample_frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
    
    def test_to_pandas_with_copy_true(self, sample_frame):
        """Test conversion with defensive copy enabled."""
        df = ar.to_pandas(sample_frame, copy=True)
        assert isinstance(df, pd.DataFrame)
    
    def test_to_pandas_preserves_columns(self, sample_frame):
        """Test that column structure is preserved."""
        df = ar.to_pandas(sample_frame)
        assert "name" in df.columns
        assert "age" in df.columns
        assert "salary" in df.columns


class TestFromDict:
    """Test ArFrame creation from dictionary."""
    
    def test_from_dict_basic(self):
        """Test creating ArFrame from dictionary."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None
    
    def test_from_dict_class_method(self):
        """Test ArFrame.from_dict() class method syntax."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None
    
    def test_from_dict_unequal_column_lengths(self):
        """Test that unequal column lengths raise error."""
        data = {"name": ["Alice", "Bob"], "age": [25]}
        with pytest.raises(Exception):
            ar.from_dict(data)


class TestFrameMethods:
    """Test ArFrame instance methods."""
    
    def test_frame_preview(self, sample_frame):
        """Test frame.preview() shows first N rows."""
        result = sample_frame.preview(2)
        assert result is not None
    
    def test_frame_describe(self, sample_frame):
        """Test frame.describe() generates summary statistics."""
        result = sample_frame.describe()
        assert result is not None


class TestSchema:
    """Test schema creation, serialization, and validation."""
    
    def test_schema_bootstrap_from_report(self, sample_frame):
        """Test creating Schema from DataQualityReport."""
        report = ar.profile(sample_frame)
        schema = ar.Schema.bootstrap_from_report(report)
        assert schema is not None
    
    def test_schema_to_json_serialization(self, sample_frame):
        """Test schema serialization to JSON string."""
        report = ar.profile(sample_frame)
        schema = ar.Schema.bootstrap_from_report(report)
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        json.loads(json_str)  # Verify valid JSON
    
    def test_schema_from_json_deserialization(self, sample_frame):
        """Test schema deserialization from JSON string."""
        report = ar.profile(sample_frame)
        schema = ar.Schema.bootstrap_from_report(report)
        json_str = schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)
        assert restored_schema is not None
    
    def test_schema_validate_basic(self, sample_frame):
        """Test validating ArFrame against schema."""
        report = ar.profile(sample_frame)
        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(sample_frame)
        assert result is not None
    
    def test_schema_validate_with_max_errors(self, sample_frame):
        """Test validation with error limit."""
        report = ar.profile(sample_frame)
        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(sample_frame, max_errors=5)
        assert result is not None
    
    def test_schema_bootstrap_invalid_input(self):
        """Test bootstrap_from_report rejects non-report input."""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not_a_report")
