import pytest
import pandas as pd
import io
import os
import arnio as ar


class TestReadCsvChunked:
    """Tests for ar.read_csv_chunked functionality."""

    @pytest.fixture
    def simple_csv(self, tmp_path):
        """Create a simple CSV file with 5 rows."""
        content = "id,name,score\n1,alice,95\n2,bob,87\n3,charlie,92\n4,diana,88\n5,eve,91\n"
        path = tmp_path / "data.csv"
        path.write_text(content)
        return str(path)

    @pytest.fixture
    def tsv_file(self, tmp_path):
        """Create a TSV file with tab delimiters."""
        content = "id\tname\tscore\n1\talice\t95\n2\tbob\t87\n3\tcharlie\t92\n"
        path = tmp_path / "data.tsv"
        path.write_text(content)
        return str(path)

    @pytest.fixture
    def csv_with_nulls(self, tmp_path):
        """Create a CSV file with missing values."""
        content = "id,name,status\n1,alice,active\n2,,inactive\n3,charlie,\n4,diana,active\n"
        path = tmp_path / "nulls.csv"
        path.write_text(content)
        return str(path)

    def test_basic_chunking_yields_correct_sizes(self, simple_csv):
        """Test that chunks are yielded with the specified chunk size."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2))
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2
        assert chunks[2].shape[0] == 1

    def test_chunk_columns_consistent(self, simple_csv):
        """Test that all chunks have the same column names."""
        expected_cols = ["id", "name", "score"]
        for chunk in ar.read_csv_chunked(simple_csv, chunksize=2):
            assert chunk.columns == expected_cols

    def test_tsv_delimiter_auto_detected(self, tsv_file):
        """Test that .tsv files automatically use tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=2))
        df = ar.to_pandas(chunks[0])
        assert df.shape[1] == 3
        assert "id" in df.columns and "name" in df.columns

    def test_explicit_delimiter_override(self, tmp_path):
        """Test that explicit delimiter parameter overrides auto-detection."""
        content = "id,name,score\n1,alice,95\n2,bob,87\n"
        path = tmp_path / "data.tsv"
        path.write_text(content)
        chunks = list(ar.read_csv_chunked(
            str(path), 
            chunksize=10, 
            delimiter=","
        ))
        assert chunks[0].columns == ["id", "name", "score"]

    def test_usecols_selects_columns(self, simple_csv):
        """Test that usecols parameter filters to selected columns."""
        chunks = list(ar.read_csv_chunked(
            simple_csv, 
            chunksize=3, 
            usecols=["id", "name"]
        ))
        for chunk in chunks:
            assert chunk.columns == ["id", "name"]
            assert len(chunk.columns) == 2

    def test_nrows_limits_total_rows(self, simple_csv):
        """Test that nrows parameter limits total rows across all chunks."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2, nrows=3))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 3

    def test_skip_rows_after_header(self, simple_csv):
        """Test that skip_rows skips data rows after the header."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=10, skip_rows=2))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 3  # 5 rows - 2 skipped

    def test_skiprows_alias_works(self, simple_csv):
        """Test that skiprows parameter works as an alias for skip_rows."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=10, skiprows=1))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 4

    def test_conflicting_skiprows_skip_rows_raises_error(self, simple_csv):
        """Test that providing conflicting skiprows and skip_rows raises ValueError."""
        with pytest.raises(ValueError, match="Conflicting values"):
            list(ar.read_csv_chunked(
                simple_csv, 
                skip_rows=1, 
                skiprows=2,
                chunksize=10
            ))

    def test_file_like_object_stringio(self, simple_csv):
        """Test that file-like objects (StringIO) are supported."""
        with open(simple_csv) as f:
            content = f.read()
        file_obj = io.StringIO(content)
        chunks = list(ar.read_csv_chunked(file_obj, chunksize=2))
        assert len(chunks) == 3
        assert all(c.shape[0] > 0 for c in chunks[:-1])

    def test_custom_null_values(self, tmp_path):
        """Test that null_values parameter treats custom strings as null."""
        content = "id,name,status\n1,alice,active\n2,bob,MISSING\n3,charlie,UNKNOWN\n"
        path = tmp_path / "custom_nulls.csv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            chunksize=10,
            null_values=["MISSING", "UNKNOWN"]
        ))
        df = ar.to_pandas(chunks[0])
        assert pd.isna(df.loc[1, "status"])
        assert pd.isna(df.loc[2, "status"])

    def test_permissive_mode_fills_missing_fields(self, tmp_path):
        """Test that permissive mode fills missing trailing fields with nulls."""
        content = "id,name,score\n1,alice,95\n2,bob\n3,charlie,92\n"
        path = tmp_path / "permissive.csv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            chunksize=10,
            mode="permissive"
        ))
        df = ar.to_pandas(chunks[0])
        assert df.shape[0] == 3
        assert pd.isna(df.loc[1, "score"])

    def test_on_bad_lines_skip(self, tmp_path):
        """Test that on_bad_lines='skip' drops rows with extra fields."""
        content = "id,name,score\n1,alice,95\n2,bob,87,extra\n3,charlie,92\n"
        path = tmp_path / "bad_lines.csv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            chunksize=10,
            mode="strict",
            on_bad_lines="skip"
        ))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 2

    def test_decimal_separator_parsing(self, tmp_path):
        """Test that decimal_separator parameter handles non-dot decimals."""
        content = "id,price\n1,19,99\n2,29,95\n3,15,50\n"
        path = tmp_path / "decimals.csv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            chunksize=10,
            decimal_separator=","
        ))
        assert len(chunks) > 0
        assert chunks[0].shape[0] == 3

    def test_empty_csv_header_only(self, tmp_path):
        """Test handling of CSV with only header row."""
        content = "id,name,score\n"
        path = tmp_path / "empty.csv"
        path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(str(path), chunksize=10))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 0
