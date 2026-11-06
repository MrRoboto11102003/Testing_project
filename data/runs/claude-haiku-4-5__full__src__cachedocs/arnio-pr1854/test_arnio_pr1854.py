import pytest
import arnio as ar
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import io
import warnings


@pytest.fixture
def tmp_csv(tmp_path):
    """Create a simple CSV file for testing."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "id,name,value\n"
        "1,alice,10.5\n"
        "2,bob,20.3\n"
        "3,charlie,30.1\n"
        "4,diana,40.9\n"
        "5,eve,50.2\n"
    )
    return str(csv_path)


@pytest.fixture
def tmp_tsv(tmp_path):
    """Create a simple TSV file for testing."""
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text(
        "id\tname\tvalue\n"
        "1\talice\t10.5\n"
        "2\tbob\t20.3\n"
        "3\tcharlie\t30.1\n"
    )
    return str(tsv_path)


@pytest.fixture
def tmp_csv_with_nulls(tmp_path):
    """Create a CSV with empty cells (null values)."""
    csv_path = tmp_path / "test_nulls.csv"
    csv_path.write_text(
        "id,name,value\n"
        "1,alice,10\n"
        "2,,20\n"
        "3,charlie,\n"
        "4,diana,40\n"
    )
    return str(csv_path)


@pytest.fixture
def tmp_csv_header_only(tmp_path):
    """Create an empty CSV (header only)."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("id,name,value\n")
    return str(csv_path)


def test_basic_chunked_iteration(tmp_csv):
    """Test basic iteration and chunk size enforcement."""
    chunks = list(ar.read_csv_chunked(tmp_csv, chunksize=2))

    assert len(chunks) == 3, "Should yield 3 chunks (2+2+1 rows)"
    assert chunks[0].shape[0] == 2
    assert chunks[1].shape[0] == 2
    assert chunks[2].shape[0] == 1
    assert all(c.shape[1] == 3 for c in chunks)


def test_chunk_boundaries_respected(tmp_csv):
    """Verify all chunks except possibly the last respect chunksize."""
    chunksize = 2
    chunks = list(ar.read_csv_chunked(tmp_csv, chunksize=chunksize))

    for chunk in chunks[:-1]:
        assert chunk.shape[0] == chunksize, f"Non-last chunk should have {chunksize} rows"
    assert chunks[-1].shape[0] <= chunksize, "Last chunk should not exceed chunksize"


def test_tsv_auto_delimiter_detection(tmp_tsv):
    """Test that .tsv extension auto-detects tab delimiter without explicit parameter."""
    chunks = list(ar.read_csv_chunked(tmp_tsv, chunksize=2))

    assert len(chunks) == 2, "Should correctly parse TSV with auto-detected tab delimiter"
    assert chunks[0].shape == (2, 3), "Should have 2 rows, 3 columns"


def test_explicit_delimiter_overrides_extension(tmp_path):
    """Test that explicit delimiter takes precedence over extension-based detection."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("id;name;value\n1;alice;10\n2;bob;20\n")

    chunks = list(ar.read_csv_chunked(csv_path, delimiter=";", chunksize=10))

    assert chunks[0].shape == (2, 3), "Should parse with explicit semicolon delimiter"
    assert "id" in chunks[0].columns


def test_usecols_selects_subset(tmp_csv):
    """Test column selection with usecols parameter."""
    chunks = list(ar.read_csv_chunked(tmp_csv, usecols=["id", "name"], chunksize=2))

    assert all(c.shape[1] == 2 for c in chunks), "All chunks should have 2 columns"
    assert all("id" in c.columns and "name" in c.columns for c in chunks)
    assert all("value" not in c.columns for c in chunks)


def test_usecols_preserves_column_order(tmp_csv):
    """Test that usecols preserves the requested column order."""
    requested_cols = ["value", "id", "name"]
    chunks = list(ar.read_csv_chunked(tmp_csv, usecols=requested_cols, chunksize=10))

    assert list(chunks[0].columns) == requested_cols


def test_nrows_limits_total_rows_across_chunks(tmp_csv):
    """Test that nrows limits total rows read across all chunks."""
    total_without_limit = sum(
        c.shape[0] for c in ar.read_csv_chunked(tmp_csv, chunksize=2)
    )
    assert total_without_limit == 5, "CSV has 5 data rows"

    total_with_limit = sum(
        c.shape[0] for c in ar.read_csv_chunked(tmp_csv, chunksize=2, nrows=3)
    )
    assert total_with_limit == 3, "nrows=3 should limit to 3 rows total"


def test_skip_rows_skips_data_rows_after_header(tmp_csv):
    """Test that skip_rows skips data rows (not header)."""
    chunks_no_skip = list(ar.read_csv_chunked(tmp_csv, chunksize=10))
    total_no_skip = sum(c.shape[0] for c in chunks_no_skip)

    chunks_skip_2 = list(ar.read_csv_chunked(tmp_csv, skip_rows=2, chunksize=10))
    total_skip_2 = sum(c.shape[0] for c in chunks_skip_2)

    assert total_no_skip == 5
    assert total_skip_2 == 3, "Should skip 2 data rows after header"


def test_skiprows_alias_for_skip_rows(tmp_csv):
    """Test that skiprows works as alias for skip_rows."""
    chunks = list(ar.read_csv_chunked(tmp_csv, skiprows=1, chunksize=10))
    total = sum(c.shape[0] for c in chunks)

    assert total == 4, "skiprows=1 should skip 1 data row"


def test_column_types_consistent_across_chunks(tmp_csv):
    """Test that inferred types from first chunk apply to all chunks."""
    chunks = list(ar.read_csv_chunked(tmp_csv, chunksize=1))

    first_dtypes = chunks[0].dtypes
    for chunk in chunks[1:]:
        assert chunk.dtypes == first_dtypes, "All chunks should have same dtypes"


def test_dtype_forces_column_types(tmp_csv):
    """Test dtype parameter forces specific column types."""
    chunks = list(
        ar.read_csv_chunked(
            tmp_csv,
            dtype={"id": "int64", "value": "float64"},
            chunksize=10,
        )
    )

    df = ar.to_pandas(chunks[0])
    assert df["id"].dtype == np.int64
    assert df["value"].dtype == np.float64


def test_empty_csv_with_header_only(tmp_csv_header_only):
    """Test reading CSV with header but no data rows."""
    chunks = list(ar.read_csv_chunked(tmp_csv_header_only, chunksize=10))

    total_rows = sum(c.shape[0] for c in chunks)
    assert total_rows == 0, "Header-only CSV should yield no data rows"


def test_single_row_csv(tmp_path):
    """Test CSV with exactly one data row."""
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("id,name,value\n1,alice,10\n")

    chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10))

    assert len(chunks) == 1
    assert chunks[0].shape == (1, 3)


def test_null_values_detected_and_preserved(tmp_csv_with_nulls):
    """Test that empty cells are detected as nulls and preserved across chunks."""
    chunks = list(ar.read_csv_chunked(tmp_csv_with_nulls, chunksize=2))

    df = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
    # Row 2 (index 1) should have null in name column
    # Row 3 (index 2) should have null in value column
    assert df.loc[1, "name"] is pd.NA or pd.isna(df.loc[1, "name"])


def test_trim_headers_removes_whitespace(tmp_path):
    """Test that trim_headers=True strips whitespace from column names."""
    csv_path = tmp_path / "ws_headers.csv"
    csv_path.write_text(" id , name , value \n1,alice,10\n")

    chunks = list(
        ar.read_csv_chunked(str(csv_path), trim_headers=True, chunksize=10)
    )

    assert list(chunks[0].columns) == ["id", "name", "value"]
