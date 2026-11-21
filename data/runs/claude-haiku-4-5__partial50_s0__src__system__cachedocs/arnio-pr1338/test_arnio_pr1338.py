import pytest
import csv
from pathlib import Path
import pandas as pd
import arnio as ar


@pytest.fixture
def messy_csv(tmp_path):
    """Create a messy CSV file for testing."""
    csv_path = tmp_path / "messy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age", "email"])
        writer.writerow([" Alice ", "30", " alice@example.com "])
        writer.writerow([" Bob ", "25", " bob@example.com "])
        writer.writerow([" Alice ", "30", " alice@example.com "])  # duplicate
    return csv_path


@pytest.fixture
def frame_from_messy_csv(messy_csv):
    """Load messy CSV into an ArFrame."""
    return ar.read_csv(str(messy_csv))


def test_auto_clean_safe_mode_default(frame_from_messy_csv):
    """Test safe mode (default) strips whitespace only."""
    clean = ar.auto_clean(frame_from_messy_csv)
    
    assert isinstance(clean, ar.ArFrame)
    assert clean.shape[0] == 4  # Duplicates NOT removed in safe mode
    
    df = ar.to_pandas(clean)
    assert df["name"].iloc[0] == "Alice"
    assert df["email"].iloc[0] == "alice@example.com"


def test_auto_clean_safe_mode_with_return_report(frame_from_messy_csv):
    """Test safe mode with return_report returns tuple of (frame, report)."""
    result = ar.auto_clean(frame_from_messy_csv, return_report=True)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    clean, report = result
    
    assert isinstance(clean, ar.ArFrame)
    assert isinstance(report, ar.DataQualityReport)


def test_auto_clean_strict_mode_requires_allow_lossy_casts(frame_from_messy_csv):
    """Test strict mode without allow_lossy_casts raises ValueError."""
    with pytest.raises(ValueError, match="allow_lossy_casts"):
        ar.auto_clean(frame_from_messy_csv, mode="strict")


def test_auto_clean_dry_run_returns_report_only(frame_from_messy_csv):
    """Test dry_run=True returns DataQualityReport, not ArFrame."""
    result = ar.auto_clean(frame_from_messy_csv, mode="safe", dry_run=True)
    
    assert isinstance(result, ar.DataQualityReport)
    assert not isinstance(result, ar.ArFrame)


def test_auto_clean_dry_run_with_return_report_raises(frame_from_messy_csv):
    """Test combining dry_run + return_report raises ValueError."""
    with pytest.raises(ValueError, match="return_report"):
        ar.auto_clean(
            frame_from_messy_csv,
            dry_run=True,
            return_report=True,
        )


def test_auto_clean_dry_run_with_explain_raises(frame_from_messy_csv):
    """Test combining dry_run + explain raises ValueError."""
    with pytest.raises(ValueError, match="explain"):
        ar.auto_clean(
            frame_from_messy_csv,
            dry_run=True,
            explain=True,
        )


def test_auto_clean_explain_mode(frame_from_messy_csv):
    """Test explain mode returns (frame, explanation) tuple."""
    result = ar.auto_clean(frame_from_messy_csv, explain=True)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    clean, explanation = result
    
    assert isinstance(clean, ar.ArFrame)
    assert hasattr(explanation, "steps")
    assert hasattr(explanation, "mode")
    assert hasattr(explanation, "rows_before")
    assert hasattr(explanation, "rows_after")


def test_auto_clean_explain_and_return_report(frame_from_messy_csv):
    """Test explain + return_report returns (frame, report, explanation)."""
    result = ar.auto_clean(
        frame_from_messy_csv,
        explain=True,
        return_report=True,
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    clean, report, explanation = result
    
    assert isinstance(clean, ar.ArFrame)
    assert isinstance(report, ar.DataQualityReport)
    assert hasattr(explanation, "steps")


def test_auto_clean_strict_mode_workflow(frame_from_messy_csv):
    """Test full strict workflow: dry_run inspection → confirmed_casts → clean."""
    # Step 1: Inspect with dry_run
    report = ar.auto_clean(frame_from_messy_csv, mode="strict", dry_run=True)
    
    # Step 2: Extract proposed cast mapping from suggestions
    confirmed_casts = {}
    suggestions_dict = {}
    try:
        suggestions_dict = dict(report.suggestions)
    except (ValueError, TypeError):
        for s in report.suggestions:
            if isinstance(s, tuple):
                suggestions_dict[s[0]] = s[1]
            else:
                suggestions_dict[s.step] = s.kwargs
    
    confirmed_casts = suggestions_dict.get("cast_types", {})
    
    # Step 3: Apply strict mode with confirmed mapping
    clean = ar.auto_clean(
        frame_from_messy_csv,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=confirmed_casts,
    )
    
    assert isinstance(clean, ar.ArFrame)
    df = ar.to_pandas(clean)
    assert "name" in df.columns


def test_auto_clean_confirmed_casts_must_match(frame_from_messy_csv):
    """Test that confirmed_casts must exactly match proposed mapping."""
    # Get proposed mapping
    report = ar.auto_clean(frame_from_messy_csv, mode="strict", dry_run=True)
    
    # Try with wrong mapping
    wrong_casts = {"nonexistent_column": "string"}
    
    with pytest.raises(ValueError, match="confirmed_casts must match"):
        ar.auto_clean(
            frame_from_messy_csv,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts=wrong_casts,
        )


def test_auto_clean_invalid_mode_raises(frame_from_messy_csv):
    """Test invalid mode parameter raises ValueError."""
    with pytest.raises(ValueError, match="mode must be"):
        ar.auto_clean(frame_from_messy_csv, mode="invalid_mode")


def test_auto_clean_non_arframe_raises(tmp_path):
    """Test auto_clean raises TypeError for non-ArFrame input."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    
    with pytest.raises(TypeError):
        ar.auto_clean(df)


def test_auto_clean_to_pandas_and_csv_export(tmp_path, frame_from_messy_csv):
    """Test full workflow: clean → convert to pandas → write CSV."""
    clean = ar.auto_clean(frame_from_messy_csv)
    
    # Convert to pandas
    df = ar.to_pandas(clean)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4
    assert df["name"].iloc[0] == "Alice"  # whitespace stripped
    
    # Write to CSV
    output_csv = tmp_path / "cleaned.csv"
    ar.write_csv(clean, str(output_csv))
    assert output_csv.exists()
    
    # Verify roundtrip
    df_read = pd.read_csv(output_csv)
    assert df_read.shape == df.shape


def test_auto_clean_strict_integration_full_workflow(tmp_path):
    """Integration: CSV with duplicates → strict clean → verified output."""
    # Create source CSV
    csv_path = tmp_path / "source.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "score"])
        writer.writerow(["1", " Alice ", "95"])
        writer.writerow(["2", " Bob ", "87"])
        writer.writerow(["1", " Alice ", "95"])  # exact duplicate
    
    frame = ar.read_csv(str(csv_path))
    initial_rows = frame.shape[0]
    
    # Inspect
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    
    # Extract casts
    suggested_casts = {}
    try:
        suggested_casts = dict(report.suggestions)
    except (ValueError, TypeError):
        for s in report.suggestions:
            if isinstance(s, tuple):
                suggested_casts[s[0]] = s[1]
            else:
                suggested_casts[s.step] = s.kwargs
    
    casts_to_apply = suggested_casts.get("cast_types", {})
    
    # Clean with explanation
    clean, explanation = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=casts_to_apply,
        explain=True,
    )
    
    # Verify
    df = ar.to_pandas(clean)
    assert df["name"].iloc[0] == "Alice"  # stripped
    assert clean.shape[0] < initial_rows  # duplicates removed
    assert explanation.rows_removed > 0
    
    # Write output
    output_csv = tmp_path / "output.csv"
    ar.write_csv(clean, str(output_csv))
    df_final = pd.read_csv(output_csv)
    assert len(df_final) == clean.shape[0]
