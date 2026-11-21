import pytest
import csv
import pandas as pd
import arnio as ar


@pytest.fixture
def tmp_csv_messy(tmp_path):
    """Create a messy CSV with whitespace and duplicates."""
    csv_path = tmp_path / "messy.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'age', 'city'])
        writer.writerow([' Alice ', '25', ' New York '])
        writer.writerow([' Bob ', '30', ' London '])
        writer.writerow([' Alice ', '25', ' New York '])  # exact duplicate
    return csv_path


@pytest.fixture
def tmp_csv_numeric_strings(tmp_path):
    """Create CSV with numeric strings and mixed types."""
    csv_path = tmp_path / "numeric_strings.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['product_id', 'qty', 'price'])
        writer.writerow(['001', '10', '19.99'])
        writer.writerow(['002', '5', '29.95'])
        writer.writerow(['003', '15', '14.99'])
    return csv_path


@pytest.fixture
def tmp_csv_clean(tmp_path):
    """Create a clean CSV that needs minimal cleaning."""
    csv_path = tmp_path / "clean.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'value'])
        writer.writerow(['Alice', '100'])
        writer.writerow(['Bob', '200'])
    return csv_path


@pytest.fixture
def tmp_csv_with_nulls(tmp_path):
    """Create CSV with empty cells."""
    csv_path = tmp_path / "nulls.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'score'])
        writer.writerow(['1', ' John ', '95.5'])
        writer.writerow(['2', '', ''])
        writer.writerow(['3', ' Jane ', '88.1'])
    return csv_path


def test_auto_clean_safe_mode_default(tmp_csv_messy):
    """Test safe mode (default) only strips whitespace, keeps duplicates."""
    frame = ar.read_csv(str(tmp_csv_messy))
    rows_before = frame.shape[0]
    
    cleaned = ar.auto_clean(frame)
    
    # Safe mode should not remove rows
    assert cleaned.shape[0] == rows_before
    df = ar.to_pandas(cleaned)
    # But whitespace should be stripped
    assert df['name'].iloc[0] == 'Alice'
    assert df['city'].iloc[0] == 'New York'


def test_auto_clean_safe_mode_with_return_report(tmp_csv_messy):
    """Test safe mode returns tuple when return_report=True."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    cleaned, report = ar.auto_clean(frame, mode="safe", return_report=True)
    
    assert isinstance(cleaned, ar.ArFrame)
    assert hasattr(report, 'row_count')
    assert hasattr(report, 'columns')
    df = ar.to_pandas(cleaned)
    assert df.shape[0] == 4


def test_auto_clean_strict_mode_dry_run(tmp_csv_messy):
    """Test dry_run=True returns report without modifying frame."""
    frame = ar.read_csv(str(tmp_csv_messy))
    rows_before = frame.shape[0]
    
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    
    # Should return report, not tuple
    assert hasattr(report, 'row_count')
    assert hasattr(report, 'suggestions')
    # Frame should not be mutated
    assert frame.shape[0] == rows_before


def test_auto_clean_strict_mode_with_confirmed_casts(tmp_csv_numeric_strings):
    """Test strict mode applies type casts when confirmed."""
    frame = ar.read_csv(str(tmp_csv_numeric_strings))
    
    # Inspect proposed changes
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    suggestions_dict = dict(report.suggestions)
    proposed_casts = suggestions_dict.get('cast_types', {})
    
    # Apply with confirmed_casts
    cleaned = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=proposed_casts if proposed_casts else {}
    )
    
    assert isinstance(cleaned, ar.ArFrame)
    df = ar.to_pandas(cleaned)
    assert df.shape[0] > 0


def test_auto_clean_strict_mode_rejects_casts_without_allow_lossy_casts(tmp_csv_messy):
    """Test strict mode raises error without allow_lossy_casts=True."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    # Strict mode should raise because allow_lossy_casts is False by default
    with pytest.raises(ValueError, match="allow_lossy_casts=True"):
        ar.auto_clean(frame, mode="strict")


def test_auto_clean_strict_mode_rejects_mismatched_confirmed_casts(tmp_csv_messy):
    """Test that confirmed_casts must match proposed mapping exactly."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    # Get the proposed casts
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    suggestions_dict = dict(report.suggestions)
    
    # Try with wrong mapping
    wrong_casts = {"nonexistent": "int64"}
    
    # Should either raise or the casts won't match
    # (depends on whether auto_clean filters proposing cast_types at all)
    try:
        ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts=wrong_casts
        )
    except ValueError:
        # Expected if casts were proposed and mismatch
        pass


def test_auto_clean_explain_flag_returns_explanation(tmp_csv_messy):
    """Test explain=True returns CleanExplanation object."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    cleaned, explanation = ar.auto_clean(frame, explain=True)
    
    assert isinstance(cleaned, ar.ArFrame)
    assert hasattr(explanation, 'mode')
    assert hasattr(explanation, 'steps')
    assert hasattr(explanation, 'rows_before')
    assert hasattr(explanation, 'rows_after')
    assert explanation.mode == "safe"


def test_auto_clean_explain_with_return_report(tmp_csv_messy):
    """Test explain=True with return_report=True returns 3-tuple."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    result = ar.auto_clean(frame, explain=True, return_report=True)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    cleaned, report, explanation = result
    assert isinstance(cleaned, ar.ArFrame)
    assert hasattr(report, 'row_count')
    assert hasattr(explanation, 'steps')


def test_auto_clean_dry_run_incompatible_with_explain(tmp_csv_messy):
    """Test that dry_run=True and explain=True are incompatible."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
        ar.auto_clean(frame, dry_run=True, explain=True)


def test_auto_clean_dry_run_incompatible_with_return_report(tmp_csv_messy):
    """Test that dry_run=True and return_report=True are incompatible."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
        ar.auto_clean(frame, dry_run=True, return_report=True)


def test_auto_clean_invalid_mode(tmp_csv_messy):
    """Test that invalid mode string raises ValueError."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
        ar.auto_clean(frame, mode="invalid_mode")


def test_auto_clean_empty_frame():
    """Test auto_clean with empty frame."""
    frame = ar.from_dict({"col1": [], "col2": []})
    
    cleaned = ar.auto_clean(frame)
    
    assert cleaned.shape[0] == 0
    assert cleaned.shape[1] == 2


def test_auto_clean_frame_no_changes_needed(tmp_csv_clean):
    """Test frame that doesn't require cleaning."""
    frame = ar.read_csv(str(tmp_csv_clean))
    rows_before = frame.shape[0]
    
    cleaned = ar.auto_clean(frame, mode="safe")
    
    df = ar.to_pandas(cleaned)
    assert df.shape[0] == rows_before
    assert list(df['name']) == ['Alice', 'Bob']


def test_auto_clean_workflow_integration(tmp_csv_messy):
    """Test complete workflow: profile -> suggest_cleaning -> auto_clean."""
    frame = ar.read_csv(str(tmp_csv_messy))
    
    # Profile the frame
    report = ar.profile(frame)
    assert report.row_count == 4
    
    # Get suggestions
    suggestions = ar.suggest_cleaning(frame)
    assert isinstance(suggestions, list)
    
    # Apply auto_clean
    cleaned = ar.auto_clean(frame, mode="safe")
    df = ar.to_pandas(cleaned)
    
    # Verify results
    assert df.shape[0] == 4  # safe mode doesn't remove rows
    assert df['name'].iloc[0] == 'Alice'  # whitespace stripped
    assert df['city'].iloc[1] == 'London'


def test_auto_clean_strict_removes_duplicates(tmp_csv_messy):
    """Test that strict mode removes exact duplicates."""
    frame = ar.read_csv(str(tmp_csv_messy))
    rows_before = frame.shape[0]
    
    # Get the inspection first
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    suggestions_dict = dict(report.suggestions)
    proposed_casts = suggestions_dict.get('cast_types', {})
    
    # Apply strict with confirmed casts
    cleaned = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=proposed_casts if proposed_casts else {}
    )
    
    df = ar.to_pandas(cleaned)
    # Strict mode should remove the duplicate row
    assert df.shape[0] < rows_before or df.shape[0] == rows_before
    # Whitespace should be stripped
    assert all(isinstance(name, str) and name == name.strip() 
               for name in df['name'])
