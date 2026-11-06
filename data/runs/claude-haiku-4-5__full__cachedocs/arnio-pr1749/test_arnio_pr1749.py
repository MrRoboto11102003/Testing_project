import pytest
import io
import tempfile
from pathlib import Path
import pandas as pd
import arnio as ar


@pytest.fixture
def simple_csv():
    """Create a simple CSV file for testing."""
    csv_data = "id,name,score\n1,Alice,95.5\n2,Bob,87.3\n3,Charlie,92.1\n4,Diana,88.9\n5,Eve,91.2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write(csv_data)
        temp_path = f.name
    yield temp_path
    import os
    os.unlink(temp_path)


@pytest.fixture
def tsv_file():
    """Create a TSV file for testing."""
    tsv_data = "id\tname\tscore\n1\tAlice\t95.5\n2\tBob\t87.3\n3\tCharlie\t92.1\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
        f.write(tsv_data)
        temp_path = f.name
    yield temp_path
    import os
    os.unlink(temp_path)


class TestReadCsvChunked:
    """Tests for ar.read_csv_chunked"""

    def test_chunking_yields_correct_sizes(self, simple_csv):
        """Test that chunks are yielded with correct sizes, last can be smaller."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2))
        
        assert len(chunks) == 3  # 5 rows / 2 per chunk = 3 chunks
        assert ar.to_pandas(chunks[0]).shape[0] == 2
        assert ar.to_pandas(chunks[1]).shape[0] == 2
        assert ar.to_pandas(chunks[2]).shape[0] == 1  # last chunk smaller

    def test_type_inference_consistent_across_chunks(self, simple_csv):
        """Test that column types are inferred once and applied to all chunks."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2))
        
        # All chunks should have identical column types
        dtypes_0 = chunks[0].dtypes
        dtypes_1 = chunks[1].dtypes
        dtypes_2 = chunks[2].dtypes
        
        assert dtypes_0 == dtypes_1 == dtypes_2
        assert dtypes_0['id'] == 'int64'
        assert dtypes_0['score'] == 'float64'

    def test_single_chunk_when_chunksize_exceeds_rows(self, simple_csv):
        """Test that chunksize larger than file yields single chunk."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=100))
        
        assert len(chunks) == 1
        assert ar.to_pandas(chunks[0]).shape[0] == 5

    def test_nrows_limits_total_rows(self, simple_csv):
        """Test that nrows parameter limits total rows across all chunks."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2, nrows=3))
        
        total_rows = sum(ar.to_pandas(c).shape[0] for c in chunks)
        assert total_rows == 3

    def test_usecols_filters_columns(self, simple_csv):
        """Test that usecols selects only specified columns."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=2, usecols=['id', 'name']))
        
        for chunk in chunks:
            assert chunk.columns == ['id', 'name']
            assert 'score' not in chunk.columns

    def test_skip_rows_skips_data_rows(self, simple_csv):
        """Test that skip_rows skips data rows after the header."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=5, skip_rows=2))
        
        df = ar.to_pandas(chunks[0])
        # First row should be Charlie (Alice and Bob skipped)
        assert df['name'].iloc[0] == 'Charlie'
        assert len(df) == 3  # 5 total - 2 skipped

    def test_has_header_false_auto_generates_column_names(self):
        """Test reading CSV without header generates column names."""
        csv_data = "1,Alice,95.5\n2,Bob,87.3\n3,Charlie,92.1\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(temp_path, chunksize=2, has_header=False))
            
            assert len(chunks) == 2
            # Should have 3 auto-generated columns
            assert len(chunks[0].columns) == 3
            # Data should be preserved
            assert ar.to_pandas(chunks[0]).shape[0] == 2
        finally:
            import os
            os.unlink(temp_path)

    def test_tsv_auto_detects_tab_delimiter(self, tsv_file):
        """Test that .tsv extension auto-detects tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=2))
        
        # Should parse correctly with auto-detected tab delimiter
        assert chunks[0].columns == ['id', 'name', 'score']
        df = ar.to_pandas(chunks[0])
        assert 'Alice' in df['name'].values

    def test_explicit_delimiter_overrides_extension(self):
        """Test that explicit delimiter overrides auto-detection from extension."""
        csv_data = "id;name;score\n1;Alice;95.5\n2;Bob;87.3\n3;Charlie;92.1\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(temp_path, chunksize=10, delimiter=';'))
            
            assert chunks[0].columns == ['id', 'name', 'score']
            df = ar.to_pandas(chunks[0])
            assert len(df) == 3
        finally:
            import os
            os.unlink(temp_path)

    def test_trim_headers_strips_whitespace(self):
        """Test that trim_headers=True removes whitespace from column names."""
        csv_data = " id , name , score \n1,Alice,95.5\n2,Bob,87.3\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(temp_path, chunksize=10, trim_headers=True))
            
            # Column names should be trimmed
            assert chunks[0].columns == ['id', 'name', 'score']
        finally:
            import os
            os.unlink(temp_path)

    def test_null_values_custom_sentinels(self):
        """Test that custom null_values strings are recognized as null."""
        csv_data = "id,name,score\n1,Alice,95.5\n2,N/A,87.3\n3,Charlie,MISSING\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(
                temp_path,
                chunksize=10,
                null_values=['N/A', 'MISSING']
            ))
            
            df = ar.to_pandas(chunks[0])
            # Custom sentinels should be null
            assert pd.isna(df.loc[df['id'] == 2, 'name'].iloc[0])
            assert pd.isna(df.loc[df['id'] == 3, 'score'].iloc[0])
        finally:
            import os
            os.unlink(temp_path)

    def test_pathlib_path_input(self, simple_csv):
        """Test that pathlib.Path input works."""
        path = Path(simple_csv)
        chunks = list(ar.read_csv_chunked(path, chunksize=2))
        
        assert len(chunks) == 3

    def test_file_like_object_input(self):
        """Test that file-like object (io.StringIO) input works."""
        csv_data = "id,name,score\n1,Alice,95.5\n2,Bob,87.3\n3,Charlie,92.1\n"
        file_obj = io.StringIO(csv_data)
        
        chunks = list(ar.read_csv_chunked(file_obj, chunksize=2))
        
        assert len(chunks) == 2
        assert ar.to_pandas(chunks[0]).shape[0] == 2

    def test_iterator_protocol_lazy_evaluation(self, simple_csv):
        """Test that result is a lazy iterator that yields on demand."""
        iterator = ar.read_csv_chunked(simple_csv, chunksize=2)
        
        # Consume chunks one by one
        chunk1 = next(iterator)
        assert ar.to_pandas(chunk1).shape[0] == 2
        
        chunk2 = next(iterator)
        assert ar.to_pandas(chunk2).shape[0] == 2
        
        chunk3 = next(iterator)
        assert ar.to_pandas(chunk3).shape[0] == 1
        
        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(iterator)

    def test_skiprows_alias_for_skip_rows(self, simple_csv):
        """Test that skiprows parameter works as alias for skip_rows."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=5, skiprows=1))
        
        df = ar.to_pandas(chunks[0])
        # First row should be Bob (Alice skipped)
        assert df['name'].iloc[0] == 'Bob'
        assert len(df) == 4  # 5 total - 1 skipped
