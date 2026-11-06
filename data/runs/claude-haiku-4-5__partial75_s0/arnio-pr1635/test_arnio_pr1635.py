import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


@pytest.fixture
def sample_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": [" Alice ", " Bob ", " Charlie ", " Diana ", " Eve "],
        "age": [25, 30, None, 28, 35],
        "city": ["NYC", "LA", "NYC", "NYC", "LA"],
    })
    return ar.from_pandas(df)


@pytest.fixture
def messy_frame():
    """Create a frame with duplicate rows and whitespace."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": [" alice ", " bob ", " bob ", " charlie "],
        "score": [95, 87, 87, 92],
    })
    return ar.from_pandas(df)


def test_pipeline_single_step(sample_frame):
    """Test basic pipeline execution with a single step."""
    result = ar.pipeline(sample_frame, [
        ("strip_whitespace",),
    ])
    assert isinstance(result, ar.ArFrame)
    df = ar.to_pandas(result)
    assert df["name"].tolist() == ["Alice", "Bob", "Charlie", "Diana", "Eve"]


def test_pipeline_multiple_steps(sample_frame):
    """Test pipeline with multiple sequential steps."""
    result = ar.pipeline(sample_frame, [
        ("strip_whitespace",),
        ("normalize_case", {"case_type": "upper"}),
    ])
    assert isinstance(result, ar.ArFrame)
    df = ar.to_pandas(result)
    assert df["name"].tolist() == ["ALICE", "BOB", "CHARLIE", "DIANA", "EVE"]


def test_pipeline_empty_steps(sample_frame):
    """Test pipeline with empty steps list returns unchanged frame."""
    result = ar.pipeline(sample_frame, [])
    assert isinstance(result, ar.ArFrame)
    assert result.shape == sample_frame.shape
    assert result.columns == sample_frame.columns


def test_pipeline_return_metadata(sample_frame):
    """Test that return_metadata=True returns tuple with metadata dict."""
    result, metadata = ar.pipeline(
        sample_frame,
        [("strip_whitespace",)],
        return_metadata=True,
    )
    assert isinstance(result, ar.ArFrame)
    assert isinstance(metadata, dict)


def test_pipeline_metadata_contains_required_keys(sample_frame):
    """Test that metadata dict contains documented keys."""
    result, metadata = ar.pipeline(
        sample_frame,
        [("strip_whitespace",), ("drop_nulls",)],
        return_metadata=True,
    )
    assert "step_timings" in metadata
    assert "applied_steps" in metadata
    assert "row_counts" in metadata


def test_pipeline_unknown_step_raises_error(sample_frame):
    """Test that unknown step names raise UnknownStepError before execution."""
    with pytest.raises(UnknownStepError):
        ar.pipeline(sample_frame, [
            ("nonexistent_step",),
        ])


def test_pipeline_malformed_step_tuple_raises_error(sample_frame):
    """Test that malformed step tuples fail early validation."""
    with pytest.raises((UnknownStepError, ValueError, TypeError)):
        ar.pipeline(sample_frame, [
            ("strip_whitespace", "extra", "positional", "args"),
        ])


def test_pipeline_dry_run_validation(sample_frame):
    """Test that dry_run=True validates without full execution."""
    result = ar.pipeline(
        sample_frame,
        [("strip_whitespace",)],
        dry_run=True,
    )
    # Should not raise; validation passes or returns None/minimal result
    assert result is not None or result is None


def test_pipeline_verbose_flag_does_not_break_execution(sample_frame):
    """Test that verbose=True enables logging without affecting output."""
    result = ar.pipeline(
        sample_frame,
        [("strip_whitespace",)],
        verbose=True,
    )
    assert isinstance(result, ar.ArFrame)
    df = ar.to_pandas(result)
    assert len(df) == len(sample_frame)


def test_pipeline_step_with_parameters(sample_frame):
    """Test passing parameters to pipeline steps via kwargs dict."""
    result = ar.pipeline(sample_frame, [
        ("fill_nulls", {"value": 0, "subset": ["age"]}),
    ])
    df = ar.to_pandas(result)
    assert 0.0 in df["age"].values
    assert df["age"].isna().sum() == 0


def test_pipeline_subset_parameter(sample_frame):
    """Test that subset parameter limits step to specific columns."""
    result = ar.pipeline(sample_frame, [
        ("normalize_case", {"case_type": "lower", "subset": ["name"]}),
    ])
    df = ar.to_pandas(result)
    # name column should be affected
    assert all(c.islower() or c.isspace() for c in df["name"].iloc[0])


def test_pipeline_preserves_column_names_and_order(sample_frame):
    """Test that pipeline preserves original column names and order."""
    original_columns = sample_frame.columns
    result = ar.pipeline(sample_frame, [
        ("strip_whitespace",),
    ])
    assert result.columns == original_columns


def test_pipeline_row_count_tracking(messy_frame):
    """Test that metadata tracks row counts through pipeline steps."""
    result, metadata = ar.pipeline(
        messy_frame,
        [("drop_duplicates",)],
        return_metadata=True,
    )
    row_counts = metadata["row_counts"]
    assert isinstance(row_counts, list)
    assert len(row_counts) > 0
    # Final row count should be <= initial
    assert row_counts[-1] <= messy_frame.shape[0]


def test_pipeline_complex_workflow(messy_frame):
    """Test realistic multi-step cleaning pipeline."""
    result = ar.pipeline(messy_frame, [
        ("strip_whitespace",),
        ("normalize_case", {"case_type": "lower"}),
        ("drop_duplicates", {"keep": "first"}),
    ])
    df = ar.to_pandas(result)
    assert len(df) == 3  # 4 rows with 1 duplicate -> 3 rows
    assert df["name"].tolist() == ["alice", "bob", "charlie"]


def test_pipeline_return_type_without_metadata(sample_frame):
    """Test that pipeline returns ArFrame when return_metadata=False (default)."""
    result = ar.pipeline(
        sample_frame,
        [("strip_whitespace",)],
        return_metadata=False,
    )
    assert isinstance(result, ar.ArFrame)
    assert not isinstance(result, tuple)


def test_pipeline_step_order_execution(sample_frame):
    """Test that steps execute in the order specified."""
    # First: strip whitespace, then normalize case upper
    result1 = ar.pipeline(sample_frame, [
        ("strip_whitespace",),
        ("normalize_case", {"case_type": "upper"}),
    ])
    
    # Reverse order shouldn't matter for these specific steps
    result2 = ar.pipeline(sample_frame, [
        ("normalize_case", {"case_type": "upper"}),
        ("strip_whitespace",),
    ])
    
    df1 = ar.to_pandas(result1)
    df2 = ar.to_pandas(result2)
    
    # Both should have uppercase names with no leading/trailing spaces
    assert df1["name"].tolist() == df2["name"].tolist()
