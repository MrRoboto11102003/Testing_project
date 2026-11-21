import pytest
import tempfile
import os
from pathlib import Path
import arnio as ar


class TestReadCsvChunked:
    """Test suite for arnio.io.read_csv_chunked"""

    def create_csv_file(self, rows=25, headers=None, delimiter=",", suffix=".csv"):
        """Helper to create a CSV file for testing"""
        if headers is None:
            headers = ["name", "age", "score"]
        
        f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        f.write(delimiter.join(headers) + "\n")
        for i in range(rows):
            values = [f"person_{i}", str(20 + i), str(i * 10)]
            f.write(delimiter.join(values) + "\n")
        f.close()
        return f.name

    def test_basic_chunked_read(self):
        """Test basic chunked reading with default parameters"""
        path = self.create_csv_file(rows=25)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 3
            assert chunks[0].shape[0] == 10
            assert chunks[1].shape[0] == 10
            assert chunks[2].shape[0] == 5
        finally:
            os.unlink(path)

    def test_different_chunksizes(self):
        """Test that different chunksizes produce correct total row counts"""
        path = self.create_csv_file(rows=27)
        try:
            for chunksize in [5, 8, 10, 27, 50]:
                chunks = list(ar.read_csv_chunked(path, chunksize=chunksize))
                total_rows = sum(chunk.shape[0] for chunk in chunks)
                assert total_rows == 27
        finally:
            os.unlink(path)

    def test_nrows_parameter_limits_total_rows(self):
        """Test that nrows limits total rows read across all chunks"""
        path = self.create_csv_file(rows=50)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10, nrows=25))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 25
        finally:
            os.unlink(path)

    def test_usecols_selects_columns(self):
        """Test that usecols parameter selects specific columns"""
        path = self.create_csv_file(rows=10)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=5, usecols=["name", "age"]))
            assert set(chunks[0].columns) == {"name", "age"}
            assert "score" not in chunks[0].columns
        finally:
            os.unlink(path)

    def test_skip_rows_parameter(self):
        """Test skip_rows skips data rows after header"""
        path = self.create_csv_file(rows=25)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10, skip_rows=5))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 20
        finally:
            os.unlink(path)

    def test_skiprows_alias_parameter(self):
        """Test skiprows as alias for skip_rows"""
        path = self.create_csv_file(rows=25)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10, skiprows=5))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 20
        finally:
            os.unlink(path)

    def test_skip_rows_skiprows_agreement(self):
        """Test that agreeing skip_rows and skiprows values work together"""
        path = self.create_csv_file(rows=25)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10, skip_rows=5, skiprows=5))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 20
        finally:
            os.unlink(path)

    def test_skip_rows_skiprows_conflict_raises_error(self):
        """Test that conflicting skip_rows and skiprows raise ValueError"""
        path = self.create_csv_file(rows=25)
        try:
            with pytest.raises(ValueError, match="Conflicting"):
                list(ar.read_csv_chunked(path, skip_rows=5, skiprows=10))
        finally:
            os.unlink(path)

    def test_tsv_auto_detection(self):
        """Test automatic tab delimiter detection for .tsv files"""
        path = self.create_csv_file(rows=15, delimiter="\t", suffix=".tsv")
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 2
            assert chunks[0].shape[0] == 10
            assert set(chunks[0].columns) == {"name", "age", "score"}
        finally:
            os.unlink(path)

    def test_explicit_delimiter_overrides_extension(self):
        """Test that explicit delimiter overrides .tsv auto-detection"""
        path = self.create_csv_file(rows=10, delimiter=",", suffix=".tsv")
        try:
            chunks = list(ar.read_csv_chunked(path, delimiter=",", chunksize=10))
            assert set(chunks[0].columns) == {"name", "age", "score"}
        finally:
            os.unlink(path)

    def test_no_header_mode(self):
        """Test reading CSV without header row"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        f.write("person_0,20,0\n")
        f.write("person_1,21,10\n")
        f.close()
        try:
            chunks = list(ar.read_csv_chunked(f.name, has_header=False, chunksize=10))
            assert chunks[0].shape[0] == 2
        finally:
            os.unlink(f.name)

    def test_empty_file_with_header_only(self):
        """Test reading empty CSV with only header"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        f.write("name,age,score\n")
        f.close()
        try:
            chunks = list(ar.read_csv_chunked(f.name, chunksize=10))
            # Empty files typically yield no chunks or one empty chunk
            if len(chunks) > 0:
                assert all(c.shape[0] == 0 for c in chunks)
        finally:
            os.unlink(f.name)

    def test_large_chunksize_single_chunk(self):
        """Test chunksize larger than data yields single chunk"""
        path = self.create_csv_file(rows=25)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=1000))
            assert len(chunks) == 1
            assert chunks[0].shape[0] == 25
        finally:
            os.unlink(path)

    def test_chunksize_one(self):
        """Test chunksize=1 yields one row per chunk"""
        path = self.create_csv_file(rows=10)
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=1))
            assert len(chunks) == 10
            assert all(chunk.shape[0] == 1 for chunk in chunks)
        finally:
            os.unlink(path)

    def test_returns_iterator(self):
        """Test that function returns an iterator"""
        path = self.create_csv_file(rows=10)
        try:
            result = ar.read_csv_chunked(path, chunksize=5)
            assert hasattr(result, '__iter__')
            # Verify it can be consumed
            chunks = list(result)
            assert len(chunks) > 0
        finally:
            os.unlink(path)

    def test_pathlib_path_support(self):
        """Test that function accepts pathlib.Path objects"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        f.write("name,age\nalice,30\n")
        f.close()
        try:
            path = Path(f.name)
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 1
            assert chunks[0].shape[0] == 1
        finally:
            os.unlink(f.name)

    def test_dtype_parameter(self):
        """Test specifying column types with dtype parameter"""
        path = self.create_csv_file(rows=10)
        try:
            dtype_map = {"name": "string", "age": "int64", "score": "int64"}
            chunks = list(ar.read_csv_chunked(path, chunksize=5, dtype=dtype_map))
            assert len(chunks) == 2
            assert chunks[0].shape[0] == 5
        finally:
            os.unlink(path)
