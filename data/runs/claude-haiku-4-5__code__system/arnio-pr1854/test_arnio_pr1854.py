import pytest
import io
import gzip
import os
import sys
import warnings
from pathlib import Path

import arnio as ar


class TestReadCsvChunked:
    """System tests for arnio.io.read_csv_chunked."""

    def test_basic_csv_chunking(self, tmp_path):
        """Test basic CSV reading with chunking."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,10\n2,Bob,20\n3,Charlie,30\n4,Diana,40\n5,Eve,50\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=2))

        assert len(chunks) == 3
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2
        assert chunks[2].shape[0] == 1

    def test_tsv_auto_detection(self, tmp_path):
        """Test automatic tab delimiter detection for .tsv files."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id\tname\tvalue\n1\tAlice\t10\n2\tBob\t20\n")

        chunks = list(ar.read_csv_chunked(str(tsv_file), chunksize=10))

        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2

    def test_explicit_delimiter_override(self, tmp_path):
        """Test explicit delimiter overrides auto-detection."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id,name,value\n1,Alice,10\n2,Bob,20\n")

        chunks = list(ar.read_csv_chunked(str(tsv_file), delimiter=",", chunksize=10))

        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2

    def test_skip_rows_parameter(self, tmp_path):
        """Test skip_rows skips data rows after header."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,skip1\n2,skip2\n3,Alice\n4,Bob\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), skip_rows=2, chunksize=10))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 2

    def test_nrows_limit(self, tmp_path):
        """Test nrows limits total rows read across all chunks."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie\n4,Diana\n5,Eve\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), nrows=3, chunksize=10))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 3

    def test_usecols_filtering(self, tmp_path):
        """Test usecols selects specific columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value,extra\n1,Alice,10,x\n2,Bob,20,y\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), usecols=["id", "name"], chunksize=10))

        assert chunks[0].shape[0] == 2

    def test_dtype_specification(self, tmp_path):
        """Test dtype parameter for explicit type specification."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n")

        chunks = list(ar.read_csv_chunked(
            str(csv_file),
            dtype={"id": "int64", "score": "float64"},
            chunksize=10
        ))

        assert chunks[0].shape[0] == 2

    def test_null_values_handling(self, tmp_path):
        """Test null_values parameter for custom null markers."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,NA\n2,NULL,20\n3,Bob,30\n")

        chunks = list(ar.read_csv_chunked(
            str(csv_file),
            null_values=["NA", "NULL"],
            chunksize=10
        ))

        assert chunks[0].shape[0] == 3

    def test_on_bad_lines_error(self, tmp_path):
        """Test on_bad_lines='error' raises CsvReadError on extra fields."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob,extra_field\n")

        with pytest.raises(ar.CsvReadError):
            list(ar.read_csv_chunked(str(csv_file), on_bad_lines="error"))

    def test_on_bad_lines_skip(self, tmp_path):
        """Test on_bad_lines='skip' silently skips bad rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob,extra\n3,Charlie\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), on_bad_lines="skip"))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 2

    def test_on_bad_lines_warn(self, tmp_path):
        """Test on_bad_lines='warn' emits UserWarning on bad rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob,extra\n3,Charlie\n")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            chunks = list(ar.read_csv_chunked(str(csv_file), on_bad_lines="warn"))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 2

    def test_compressed_csv_gz(self, tmp_path):
        """Test reading compressed .csv.gz files."""
        csv_file = tmp_path / "data.csv.gz"
        with gzip.open(str(csv_file), 'wt') as f:
            f.write("id,name\n1,Alice\n2,Bob\n3,Charlie\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=2))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 3

    def test_file_like_object_stringio(self):
        """Test reading from StringIO file-like objects."""
        csv_content = "id,name\n1,Alice\n2,Bob\n3,Charlie\n"
        file_obj = io.StringIO(csv_content)

        chunks = list(ar.read_csv_chunked(file_obj, chunksize=2))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 3

    def test_invalid_file_format_raises_error(self, tmp_path):
        """Test unsupported file format raises ValueError."""
        invalid_file = tmp_path / "data.json"
        invalid_file.write_text('{"id": 1}')

        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(str(invalid_file)))

    def test_pathlib_path_input(self, tmp_path):
        """Test pathlib.Path objects are accepted as input."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")

        chunks = list(ar.read_csv_chunked(csv_file, chunksize=10))

        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2
