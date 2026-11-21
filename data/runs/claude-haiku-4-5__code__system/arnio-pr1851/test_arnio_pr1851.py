import pytest
import sys
import subprocess
import csv
import unicodedata
import pandas as pd
import arnio as ar


@pytest.fixture
def tmp_case_csv(tmp_path):
    """Create a test CSV for case normalization."""
    csv_path = tmp_path / "case_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "city", "description"])
        writer.writerow(["Alice", "NEW YORK", "A person"])
        writer.writerow(["Bob", "los angeles", "Another person"])
        writer.writerow(["Charlie", "SAN FRANCISCO", "Third person"])
    return csv_path


@pytest.fixture
def tmp_unicode_csv(tmp_path):
    """Create a CSV file with Unicode data."""
    csv_path = tmp_path / "unicode_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "text"])
        writer.writerow(["José", "café"])
        writer.writerow(["Mário", "naïve"])
        writer.writerow(["François", "résumé"])
    return csv_path


def test_normalize_case_lower(tmp_case_csv):
    """Test lowercase normalization on all string columns."""
    frame = ar.read_csv(str(tmp_case_csv))
    result = ar.normalize_case(frame, case_type="lower")
    df = result.to_pandas()
    
    assert df["name"].iloc[0] == "alice"
    assert df["city"].iloc[0] == "new york"
    assert df["description"].iloc[2] == "third person"


def test_normalize_case_upper(tmp_case_csv):
    """Test uppercase normalization."""
    frame = ar.read_csv(str(tmp_case_csv))
    result = ar.normalize_case(frame, case_type="upper")
    df = result.to_pandas()
    
    assert df["name"].iloc[0] == "ALICE"
    assert df["city"].iloc[1] == "LOS ANGELES"


def test_normalize_case_title(tmp_case_csv):
    """Test title case normalization."""
    frame = ar.read_csv(str(tmp_case_csv))
    result = ar.normalize_case(frame, case_type="title")
    df = result.to_pandas()
    
    assert df["name"].iloc[0] == "Alice"
    assert df["city"].iloc[0] == "New York"


def test_normalize_case_subset(tmp_case_csv):
    """Test normalization on subset of columns only."""
    frame = ar.read_csv(str(tmp_case_csv))
    result = ar.normalize_case(frame, subset=["name"], case_type="lower")
    df = result.to_pandas()
    
    assert df["name"].iloc[0] == "alice"
    assert df["city"].iloc[0] == "NEW YORK"


def test_normalize_case_non_ascii_preserved(tmp_unicode_csv):
    """Test that non-ASCII characters are preserved during case conversion."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    result = ar.normalize_case(frame, case_type="lower")
    df = result.to_pandas()
    
    name = df["name"].iloc[0].lower()
    assert "é" in name or "jose" in name
    text = df["text"].iloc[0].lower()
    assert "é" in text or "caf" in text


def test_normalize_case_invalid_case_type(tmp_case_csv):
    """Test ValueError raised for unsupported case_type."""
    frame = ar.read_csv(str(tmp_case_csv))
    
    with pytest.raises(ValueError, match="case_type must be one of"):
        ar.normalize_case(frame, case_type="invalid")


def test_normalize_case_non_string_case_type(tmp_case_csv):
    """Test TypeError raised for non-string case_type."""
    frame = ar.read_csv(str(tmp_case_csv))
    
    with pytest.raises(TypeError, match="case_type must be a string"):
        ar.normalize_case(frame, case_type=42)


def test_normalize_case_missing_column(tmp_case_csv):
    """Test KeyError raised for missing column in subset."""
    frame = ar.read_csv(str(tmp_case_csv))
    
    with pytest.raises(KeyError):
        ar.normalize_case(frame, subset=["nonexistent"])


def test_normalize_unicode_nfc(tmp_unicode_csv):
    """Test NFC Unicode normalization."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    result = ar.normalize_unicode(frame, form="NFC")
    df = result.to_pandas()
    
    for col in ["name", "text"]:
        for val in df[col].values:
            if pd.notna(val):
                assert unicodedata.is_normalized("NFC", val)


def test_normalize_unicode_nfd(tmp_unicode_csv):
    """Test NFD Unicode normalization."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    result = ar.normalize_unicode(frame, form="NFD")
    df = result.to_pandas()
    
    for col in ["name", "text"]:
        for val in df[col].values:
            if pd.notna(val):
                assert unicodedata.is_normalized("NFD", val)


def test_normalize_unicode_invalid_form(tmp_unicode_csv):
    """Test ValueError raised for invalid normalization form."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    
    with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
        ar.normalize_unicode(frame, form="invalid")


def test_normalize_unicode_non_string_form(tmp_unicode_csv):
    """Test TypeError raised for non-string form."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    
    with pytest.raises(TypeError, match="form must be a string"):
        ar.normalize_unicode(frame, form=123)


def test_normalize_unicode_missing_column(tmp_unicode_csv):
    """Test KeyError raised for missing column in subset."""
    frame = ar.read_csv(str(tmp_unicode_csv))
    
    with pytest.raises(KeyError):
        ar.normalize_unicode(frame, subset=["nonexistent"])


def test_normalize_case_cli(tmp_case_csv, tmp_path):
    """Test normalize_case via CLI, write output, and verify."""
    output_file = tmp_path / "cli_output.csv"
    
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arnio.cli",
            "normalize_case",
            "--case_type",
            "lower",
            str(tmp_case_csv),
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert output_file.exists()
    
    df = pd.read_csv(output_file)
    assert df["name"].iloc[0] == "alice"
    assert df["city"].iloc[1] == "los angeles"
    assert df["description"].iloc[0] == "a person"
