import pytest
import io
import pathlib
import arnio as ar
from arnio.exceptions import CsvReadError


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""

    def test_basic_chunked_reading(self, tmp_path):
        """Verify chunks respect chunksize and sum to full dataset"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")

        chunks = list(ar.read_csv_chunked(p, chunksize=2))

        assert len(chunks) == 3
        shapes = [c.shape for c in chunks]
        assert shapes == [(2, 2), (2, 2), (1, 2)]

    def test_tsv_auto_delimiter_detection(self, tmp_path):
        """Tab delimiter auto-detected for .tsv extension"""
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\tx\n2\ty\n")

        chunks = list(ar.read_csv_chunked(p, chunksize=10))

        assert chunks[0].shape == (2, 2)
        assert chunks[0].columns == ["a", "b"]

    def test_explicit_delimiter_overrides_auto_detection(self, tmp_path):
        """Explicit delimiter parameter overrides .tsv auto-detection"""
        p = tmp_path / "data.tsv"
        p.write_text("a,b\n1,x\n2,y\n")

        chunks = list(ar.read_csv_chunked(p, delimiter=",", chunksize=10))

        assert chunks[0].shape == (2, 2)

    def test_stringio_file_like_object(self):
        """File-like objects (StringIO) are supported"""
        data = io.StringIO("a,b\n1,x\n2,y\n3,z\n")

        chunks = list(ar.read_csv_chunked(data, chunksize=2))

        assert len(chunks) == 2
        assert [c.shape[0] for c in chunks] == [2, 1]

    def test_nrows_limits_total_rows_across_chunks(self, tmp_path):
        """nrows parameter limits total rows across all chunks"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")

        chunks = list(ar.read_csv_chunked(p, chunksize=2, nrows=3))

        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 3

    def test_skip_rows_skips_data_rows_after_header(self, tmp_path):
        """skip_rows parameter skips data rows after the header"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n")

        chunks = list(ar.read_csv_chunked(p, skip_rows=2, chunksize=10))

        df = ar.to_pandas(chunks[0])
        assert len(df) == 2  # Rows 3 and 4 only

    def test_skiprows_alias_works(self, tmp_path):
        """skiprows is an alias for skip_rows"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n")

        chunks = list(ar.read_csv_chunked(p, skiprows=1, chunksize=10))

        df = ar.to_pandas(chunks[0])
        assert len(df) == 2  # First data row skipped

    def test_conflicting_skip_rows_and_skiprows_raises_error(self, tmp_path):
        """Conflicting skip_rows and skiprows values raise ValueError on iteration"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n")

        gen = ar.read_csv_chunked(p, skip_rows=2, skiprows=3)

        with pytest.raises(ValueError, match="Conflicting values"):
            next(gen)

    def test_invalid_chunksize_raises_error(self, tmp_path):
        """Invalid chunksize raises error on first next()"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")

        gen = ar.read_csv_chunked(p, chunksize=0)

        with pytest.raises((TypeError, ValueError)):
            next(gen)

    def test_invalid_delimiter_raises_error(self, tmp_path):
        """Invalid delimiter raises error on first next()"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")

        gen = ar.read_csv_chunked(p, delimiter="\n")

        with pytest.raises((TypeError, ValueError)):
            next(gen)

    def test_invalid_mode_raises_error(self, tmp_path):
        """Invalid mode string raises ValueError on first next()"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")

        gen = ar.read_csv_chunked(p, mode="invalid")

        with pytest.raises((TypeError, ValueError)):
            next(gen)

    def test_usecols_filters_columns(self, tmp_path):
        """usecols parameter selects only specified columns"""
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,x,foo\n2,y,bar\n")

        chunks = list(ar.read_csv_chunked(p, usecols=["a", "c"], chunksize=10))

        assert chunks[0].columns == ["a", "c"]
        assert chunks[0].shape == (2, 2)

    def test_empty_file_raises_csvreaderror(self, tmp_path):
        """Empty CSV file raises CsvReadError on iteration"""
        p = tmp_path / "empty.csv"
        p.write_text("")

        gen = ar.read_csv_chunked(p)

        with pytest.raises(CsvReadError, match="empty"):
            next(gen)

    def test_pathlib_path_input_works(self, tmp_path):
        """pathlib.Path objects are accepted as input"""
        p = pathlib.Path(tmp_path) / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n")

        chunks = list(ar.read_csv_chunked(p, chunksize=10))

        assert chunks[0].shape == (2, 2)

    def test_null_values_parameter(self, tmp_path):
        """null_values parameter configures custom null sentinels"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\nMISSING,y\n3,UNKNOWN\n")

        chunks = list(ar.read_csv_chunked(
            p,
            null_values=["", "MISSING", "UNKNOWN"],
            chunksize=10
        ))

        assert chunks[0].shape == (3, 2)

    def test_errors_raised_only_on_iteration(self, tmp_path):
        """Validation errors are raised on first next(), not on function call"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")

        # Function call succeeds (generator not evaluated)
        gen = ar.read_csv_chunked(p, chunksize=-1)

        # Error raised only when iteration begins
        with pytest.raises((TypeError, ValueError)):
            next(gen)
