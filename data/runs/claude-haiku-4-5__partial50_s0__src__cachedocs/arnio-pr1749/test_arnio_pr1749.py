import pytest
import pandas as pd
import arnio as ar
import os
import io
from pathlib import Path


class TestReadCsvChunked:
    """Test suite for arnio.read_csv_chunked function."""

    @pytest.fixture
    def csv_file(self, tmp_path):
        """Create a 5-row CSV file for testing."""
        content = "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n4,David,400\n5,Eve,500\n"
        path = tmp_path / "test.csv"
        path.write_text(content)
        return str(path)

    @pytest.fixture
    def tsv_file(self, tmp_path):
        """Create a TSV file with tab delimiter."""
        content = "id\tname\tvalue\n1\tAlice\t100\n2\tBob\t200\n3\tCharlie\t300\n"
        path = tmp_path / "test.tsv"
        path.write_text(content)
        return str(path)

    def test_basic_iteration_yields_arframes(self, csv_file):
        """Test that read_csv_chunked yields ArFrame objects."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, ar.ArFrame)

    def test_chunksize_respected(self, csv_file):
        """Test that each chunk respects the chunksize limit."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2
        assert chunks[2].shape[0] == 1

    def test_columns_consistent_across_chunks(self, csv_file):
        """Test that column names remain consistent across all chunks."""
        expected = ["id", "name", "value"]
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            assert chunk.columns == expected

    def test_dtypes_inferred_from_first_chunk_applied_to_all(self, csv_file):
        """Test that column dtypes inferred from first chunk apply to subsequent chunks."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        first_dtypes = chunks[0].dtypes
        for chunk in chunks[1:]:
            assert chunk.dtypes == first_dtypes

    def test_usecols_parameter_filters_columns(self, csv_file):
        """Test that usecols selects only specified columns."""
        chunks = list(ar.read_csv_chunked(csv_file, usecols=["id", "name"], chunksize=2))
        for chunk in chunks:
            assert chunk.columns == ["id", "name"]
            assert chunk.shape[1] == 2

    def test_nrows_limits_total_rows_across_chunks(self, csv_file):
        """Test that nrows limits total rows read across all chunks."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2, nrows=3))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 3

    def test_skip_rows_skips_data_rows_after_header(self, csv_file):
        """Test that skip_rows skips data rows (after header), not header itself."""
        no_skip = list(ar.read_csv_chunked(csv_file, chunksize=10))
        with_skip = list(ar.read_csv_chunked(csv_file, skip_rows=2, chunksize=10))
        
        total_no_skip = sum(c.shape[0] for c in no_skip)
        total_with_skip = sum(c.shape[0] for c in with_skip)
        assert total_with_skip == total_no_skip - 2

    def test_tsv_auto_detects_tab_delimiter(self, tsv_file):
        """Test that .tsv file extension automatically detects tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=10))
        assert chunks[0].columns == ["id", "name", "value"]
        assert chunks[0].shape[1] == 3

    def test_explicit_delimiter_overrides_extension(self, tmp_path):
        """Test that explicit delimiter parameter overrides file extension detection."""
        # Create a file with .tsv extension but comma delimiter
        content = "id,name,value\n1,Alice,100\n2,Bob,200\n"
        path = tmp_path / "data.tsv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(str(path), delimiter=",", chunksize=10))
        assert chunks[0].columns == ["id", "name", "value"]

    def test_file_like_object_support(self, csv_file):
        """Test that file-like objects (file handles) work."""
        with open(csv_file, 'r') as f:
            chunks = list(ar.read_csv_chunked(f, chunksize=2))
        
        assert len(chunks) == 3
        assert chunks[0].columns == ["id", "name", "value"]

    def test_pathlib_path_support(self, csv_file):
        """Test that pathlib.Path objects work."""
        path = Path(csv_file)
        chunks = list(ar.read_csv_chunked(path, chunksize=2))
        assert len(chunks) == 3

    def test_convert_chunks_to_pandas_dataframes(self, csv_file):
        """Test that each chunk can be converted to a pandas DataFrame."""
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            df = ar.to_pandas(chunk)
            assert isinstance(df, pd.DataFrame)
            assert df.shape[1] == 3

    def test_pipeline_applies_to_each_chunk(self, csv_file):
        """Test that pipeline operations can be applied to each chunk."""
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            cleaned = ar.pipeline(chunk, [("strip_whitespace",)])
            assert isinstance(cleaned, ar.ArFrame)

    def test_skiprows_alias_for_skip_rows(self, csv_file):
        """Test that skiprows parameter is an alias for skip_rows."""
        chunks1 = list(ar.read_csv_chunked(csv_file, skip_rows=1, chunksize=10))
        chunks2 = list(ar.read_csv_chunked(csv_file, skiprows=1, chunksize=10))
        
        total1 = sum(c.shape[0] for c in chunks1)
        total2 = sum(c.shape[0] for c in chunks2)
        assert total1 == total2

    def test_unsupported_file_format_raises_valueerror(self, tmp_path):
        """Test that unsupported file extensions raise ValueError."""
        bad_file = tmp_path / "test.xlsx"
        bad_file.write_text("dummy content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(str(bad_file)))
