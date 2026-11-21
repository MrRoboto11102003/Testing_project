import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path


@pytest.fixture
def messy_csv(tmp_path):
    """Create a CSV file with messy column names."""
    csv_file = tmp_path / "messy.csv"
    df = pd.DataFrame({
        "First-Name": ["Alice", "Bob", "Charlie"],
        "Last@Name": ["Smith", "Jones", "Brown"],
        "Age!!!": [30, 25, 35],
        "email@domain.com": ["a@b.com", "c@d.com", "e@f.com"],
        "Phone #": ["555-1234", "555-5678", "555-9999"]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_clean_column_names_basic_lowercase(messy_csv, tmp_path):
    """Test basic column cleaning with default lowercase case_type."""
    import arnio as ar
    
    frame = ar.read_csv(messy_csv)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    expected_cols = ["first_name", "last_name", "age", "email_domain_com", "phone"]
    assert list(cleaned.columns) == expected_cols
    
    # Verify data integrity
    assert len(cleaned) == 3
    
    # Save and verify output
    output_file = tmp_path / "cleaned.csv"
    pd.DataFrame(cleaned).to_csv(output_file, index=False)
    result = pd.read_csv(output_file)
    assert list(result.columns) == expected_cols


def test_clean_column_names_uppercase(tmp_path):
    """Test clean_column_names with uppercase case_type."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "first_name": [1, 2],
        "last-name": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame, case_type="upper")
    
    assert list(cleaned.columns) == ["FIRST_NAME", "LAST_NAME"]


def test_clean_column_names_title_case(tmp_path):
    """Test clean_column_names with title case_type."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "first_name": [1, 2],
        "last_name": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame, case_type="title")
    
    assert list(cleaned.columns) == ["First_Name", "Last_Name"]


def test_clean_column_names_camel_case(tmp_path):
    """Test clean_column_names with camel case_type."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "first_name": [1, 2],
        "last_name": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame, case_type="camel")
    
    assert list(cleaned.columns) == ["firstName", "lastName"]


def test_clean_column_names_case_none(tmp_path):
    """Test clean_column_names with case_type='none'."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "FirstName": [1, 2],
        "LASTNAME": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame, case_type="none")
    
    assert list(cleaned.columns) == ["firstname", "lastname"]


def test_clean_column_names_consecutive_underscores(tmp_path):
    """Test that consecutive underscores are collapsed."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "Name___First": [1, 2],
        "Age---Years": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    assert list(cleaned.columns) == ["name_first", "age_years"]


def test_clean_column_names_boundary_underscores_trimmed(tmp_path):
    """Test that leading and trailing underscores are trimmed."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "_name_first_": [1, 2],
        "___age___": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    assert list(cleaned.columns) == ["name_first", "age"]


def test_clean_column_names_empty_becomes_column(tmp_path):
    """Test that empty column names after cleaning become 'column'."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "valid_col": [1, 2],
        "___": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    assert "column" in cleaned.columns
    assert len(cleaned.columns) == 2


def test_clean_column_names_invalid_case_type_not_string(messy_csv):
    """Test that non-string case_type raises TypeError."""
    import arnio as ar
    
    frame = ar.read_csv(messy_csv)
    
    with pytest.raises(TypeError, match="case_type must be a string"):
        ar.cleaning.clean_column_names(frame, case_type=123)


def test_clean_column_names_invalid_case_type_value(messy_csv):
    """Test that invalid case_type value raises ValueError."""
    import arnio as ar
    
    frame = ar.read_csv(messy_csv)
    
    with pytest.raises(ValueError, match="case_type must be one of"):
        ar.cleaning.clean_column_names(frame, case_type="invalid_case")


def test_clean_column_names_duplicate_after_cleaning(tmp_path):
    """Test that duplicate column names after cleaning raise ValueError."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "Name-First": [1, 2],
        "Name_First": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    
    with pytest.raises(ValueError, match="would create duplicates"):
        ar.cleaning.clean_column_names(frame)


def test_clean_column_names_already_clean(tmp_path):
    """Test that already clean columns return successfully."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "first_name": [1, 2],
        "last_name": [3, 4]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    assert list(cleaned.columns) == ["first_name", "last_name"]
    assert len(cleaned) == 2


def test_clean_column_names_with_numbers(tmp_path):
    """Test cleaning column names containing numbers."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "col1-name": [1, 2],
        "2nd_column": [3, 4],
        "3rd!col": [5, 6]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    assert list(cleaned.columns) == ["col1_name", "2nd_column", "3rd_col"]


def test_clean_column_names_special_characters(tmp_path):
    """Test cleaning column names with various special characters."""
    import arnio as ar
    
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({
        "col@#$%^&*()": [1, 2],
        "col[]{}<>|": [3, 4],
        "col/\\:;'\"": [5, 6]
    }).to_csv(csv_file, index=False)
    
    frame = ar.read_csv(csv_file)
    cleaned = ar.cleaning.clean_column_names(frame)
    
    # All special characters should be replaced with underscores
    for col in cleaned.columns:
        assert all(c.isalnum() or c == '_' for c in col)
    
    # No leading/trailing underscores
    for col in cleaned.columns:
        assert not col.startswith('_') and not col.endswith('_')


def test_clean_column_names_end_to_end_pipeline(tmp_path):
    """Test complete pipeline: load, clean, transform, save."""
    import arnio as ar
    
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    
    # Create input with messy names
    pd.DataFrame({
        "Customer-ID": [1, 2, 3],
        "First@Name": ["Alice", "Bob", "Charlie"],
        "Total$Amount": [100.5, 200.75, 300.25]
    }).to_csv(input_file, index=False)
    
    # Load and clean
    frame = ar.read_csv(input_file)
    cleaned = ar.cleaning.clean_column_names(frame, case_type="lower")
    
    # Save result
    result_df = pd.DataFrame(cleaned)
    result_df.to_csv(output_file, index=False)
    
    # Verify saved output
    assert output_file.exists()
    final = pd.read_csv(output_file)
    assert list(final.columns) == ["customer_id", "first_name", "total_amount"]
    assert len(final) == 3
