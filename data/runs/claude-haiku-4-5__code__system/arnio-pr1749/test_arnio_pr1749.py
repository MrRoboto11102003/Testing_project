import pytest
import sys
import os
import io
import csv
import gzip
import tempfile
from pathlib import Path
import pandas as pd
import warnings

import arnio as ar


class TestReadCsvChunked:
    """System tests for arnio.io.read_csv_chunked"""

    def test_basic_chunked_reading(self, tmp_path):
        """Test basic CSV chunked reading with default parameters."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "name", "value"]]
        for i in range(25):
            rows.append([str(i), f"name_{i}", str(i * 1.5)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=10))
        
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5

    def test_chunksize_respects_limit(self, tmp_path):
        """Test that chunksize parameter controls chunk size."""
        csv_file = tmp_path / "data.csv"
        rows = [["a", "b"]]
        for i in range(100):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=25))
        
        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk.shape[0] == 25

    def test_tsv_delimiter_auto_detection(self, tmp_path):
        """Test automatic delimiter detection for .tsv files."""
        tsv_file = tmp_path / "data.tsv"
        
        with open(tsv_file, "w", newline="") as f:
            f.write("col1\tcol2\tcol3\n")
            f.write("a\tb\tc\n")
            f.write("d\te\tf\n")
        
        chunks = list(ar.read_csv_chunked(str(tsv_file)))
        
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2

    def test_explicit_delimiter_overrides_extension(self, tmp_path):
        """Test that explicit delimiter overrides auto-detection."""
        tsv_file = tmp_path / "data.tsv"
        
        with open(tsv_file, "w", newline="") as f:
            f.write("col1,col2,col3\n")
            f.write("a,b,c\n")
        
        chunks = list(ar.read_csv_chunked(str(tsv_file), delimiter=","))
        assert chunks[0].shape[0] == 1

    def test_usecols_selects_columns(self, tmp_path):
        """Test that usecols parameter selects specific columns."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "name", "value", "extra"]]
        for i in range(10):
            rows.append([str(i), f"name_{i}", str(i * 1.5), f"extra_{i}"])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_file), 
            usecols=["id", "name", "value"]
        ))
        
        assert chunks[0].shape[1] == 3

    def test_nrows_limits_total_rows(self, tmp_path):
        """Test that nrows parameter limits total rows read."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        for i in range(50):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_file), 
            chunksize=10,
            nrows=25
        ))
        
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 25

    def test_skip_rows_skips_after_header(self, tmp_path):
        """Test that skip_rows parameter skips rows after header."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        for i in range(20):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(str(csv_file), skip_rows=5))
        
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 15

    def test_skiprows_as_alias_for_skip_rows(self, tmp_path):
        """Test that skiprows parameter works as alias for skip_rows."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        for i in range(15):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(str(csv_file), skiprows=3))
        
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 12

    def test_conflicting_skip_rows_and_skiprows_raises(self, tmp_path):
        """Test that conflicting skip_rows and skiprows raises ValueError."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            f.write("id,value\n1,2\n3,4\n")
        
        with pytest.raises(ValueError, match="Conflicting values"):
            list(ar.read_csv_chunked(
                str(csv_file),
                skip_rows=5,
                skiprows=10
            ))

    def test_null_values_recognized(self, tmp_path):
        """Test that null_values parameter treats specified strings as null."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        rows.append(["1", "10"])
        rows.append(["2", "NA"])
        rows.append(["3", "null"])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_file),
            null_values=["NA", "null"]
        ))
        
        assert chunks[0].shape[0] == 3

    def test_on_bad_lines_error_raises(self, tmp_path):
        """Test that on_bad_lines='error' raises CsvReadError on malformed rows."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            f.write("id,value\n")
            f.write("1,2\n")
            f.write("3,4,5\n")  # Extra field
        
        with pytest.raises(ar.CsvReadError):
            list(ar.read_csv_chunked(str(csv_file), on_bad_lines="error"))

    def test_on_bad_lines_skip_ignores_errors(self, tmp_path):
        """Test that on_bad_lines='skip' silently skips bad rows."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            f.write("id,value\n")
            f.write("1,2\n")
            f.write("3,4\n")
        
        chunks = list(ar.read_csv_chunked(str(csv_file), on_bad_lines="skip"))
        assert len(chunks) >= 1

    def test_permissive_mode_allows_missing_fields(self, tmp_path):
        """Test that mode='permissive' allows missing trailing fields."""
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            f.write("id,name,value\n")
            f.write("1,alice,10\n")
            f.write("2,bob\n")
            f.write("3,charlie,30\n")
        
        chunks = list(ar.read_csv_chunked(
            str(csv_file),
            mode="permissive",
            on_bad_lines="skip"
        ))
        
        assert len(chunks) >= 1

    def test_file_like_object_support(self, tmp_path):
        """Test reading from file-like object (io.TextIOBase)."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        for i in range(15):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        with open(csv_file, "r") as file_obj:
            chunks = list(ar.read_csv_chunked(file_obj, chunksize=5))
        
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 15

    def test_pandas_integration_via_to_pandas(self, tmp_path):
        """Test integration with pandas conversion via ar.to_pandas()."""
        csv_file = tmp_path / "data.csv"
        rows = [["id", "value"]]
        for i in range(10):
            rows.append([str(i), str(i * 1.5)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        all_dfs = []
        for chunk in ar.read_csv_chunked(str(csv_file), chunksize=5):
            df = ar.to_pandas(chunk)
            all_dfs.append(df)
        
        result_df = pd.concat(all_dfs, ignore_index=True)
        
        assert len(result_df) == 10
        assert list(result_df.columns) == ["id", "value"]

    def test_unsupported_file_format_raises(self, tmp_path):
        """Test that unsupported file formats raise ValueError."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"id": 1}')
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(str(json_file)))

    def test_pathlike_object_support(self, tmp_path):
        """Test reading from pathlib.Path object."""
        csv_file = tmp_path / "data.csv"
        rows = [["a", "b"]]
        for i in range(5):
            rows.append([str(i), str(i * 2)])
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        # Pass as pathlib.Path object
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=3))
        
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 5
