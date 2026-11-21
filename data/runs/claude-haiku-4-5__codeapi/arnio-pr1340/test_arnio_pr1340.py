import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import io

import arnio as ar
from arnio.exceptions import CsvReadError


class TestArnicoCleaner:
    """Tests for arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out"""

    def test_get_feature_names_out_returns_fitted_names_when_none(self):
        """Should return fitted feature_names_out_ when input_features is None."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        result = cleaner.get_feature_names_out(input_features=None)
        assert list(result) == ["col_a", "col_b"]

    def test_get_feature_names_out_no_argument(self):
        """Should return feature names when called without arguments."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        result = cleaner.get_feature_names_out()
        assert list(result) == ["x", "y"]

    def test_get_feature_names_out_not_fitted_raises(self):
        """Should raise if not fitted."""
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        with pytest.raises(Exception):  # NotFittedError
            cleaner.get_feature_names_out()

    def test_get_feature_names_out_rejects_string_input(self):
        """Should raise TypeError if input_features is a string."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(TypeError, match="input_features must be a sequence"):
            cleaner.get_feature_names_out(input_features="ab")

    def test_get_feature_names_out_rejects_bytes_input(self):
        """Should raise TypeError if input_features is bytes."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(TypeError, match="input_features must be a sequence"):
            cleaner.get_feature_names_out(input_features=b"ab")

    def test_get_feature_names_out_rejects_non_sequence(self):
        """Should raise TypeError if input_features is not a sequence."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(TypeError, match="input_features must be a sequence"):
            cleaner.get_feature_names_out(input_features=42)

    def test_get_feature_names_out_rejects_non_string_elements(self):
        """Should raise TypeError if any element is not a string."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(TypeError, match="All input_features must be strings"):
            cleaner.get_feature_names_out(input_features=["a", 1])

    def test_get_feature_names_out_rejects_wrong_length(self):
        """Should raise ValueError if length mismatches."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(ValueError, match="input_features should have length"):
            cleaner.get_feature_names_out(input_features=["a", "b", "c"])

    def test_get_feature_names_out_rejects_mismatched_values(self):
        """Should raise ValueError if names don't match fitted columns."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        with pytest.raises(ValueError, match="input_features must match columns"):
            cleaner.get_feature_names_out(input_features=["a", "c"])

    def test_get_feature_names_out_accepts_matching_list(self):
        """Should return output names when input matches."""
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        result = cleaner.get_feature_names_out(input_features=["col1", "col2"])
        assert list(result) == ["col1", "col2"]

    def test_get_feature_names_out_accepts_numpy_array(self):
        """Should accept numpy array of matching feature names."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        cleaner = ar.integrations.sklearn.ArnioCleaner()
        cleaner.fit(df)
        result = cleaner.get_feature_names_out(
            input_features=np.array(["x", "y"])
        )
        assert list(result) == ["x", "y"]


class TestReadCsv:
    """Tests for arnio.io.read_csv"""

    @pytest.fixture
    def simple_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.csv")
            with open(path, "w") as f:
                f.write("name,age,score\nAlice,30,95.5\nBob,25,87.3\n")
            yield path

    @pytest.fixture
    def simple_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.tsv")
            with open(path, "w") as f:
                f.write("name\tage\tscore\nAlice\t30\t95.5\nBob\t25\t87.3\n")
            yield path

    def test_read_csv_basic(self, simple_csv):
        """Should read basic CSV."""
        frame = ar.read_csv(simple_csv)
        assert frame.shape == (2, 3)
        assert frame.columns == ["name", "age", "score"]

    def test_read_csv_with_usecols(self, simple_csv):
        """Should read only specified columns."""
        frame = ar.read_csv(simple_csv, usecols=["name", "score"])
        assert frame.shape == (2, 2)
        assert frame.columns == ["name", "score"]

    def test_read_csv_with_nrows(self, simple_csv):
        """Should limit rows read."""
        frame = ar.read_csv(simple_csv, nrows=1)
        assert frame.shape[0] == 1

    def test_read_csv_with_skiprows(self, simple_csv):
        """Should skip rows after header."""
        frame = ar.read_csv(simple_csv, skiprows=1)
        assert frame.shape[0] == 1

    def test_read_csv_auto_detects_tab_for_tsv(self, simple_tsv):
        """Should auto-detect tab delimiter for .tsv files."""
        frame = ar.read_csv(simple_tsv)
        assert frame.shape == (2, 3)
        assert "name" in frame.columns

    def test_read_csv_explicit_delimiter_overrides(self, simple_tsv):
        """Should use explicit delimiter over auto-detection."""
        with pytest.raises(CsvReadError):
            ar.read_csv(simple_tsv, delimiter=",")

    def test_read_csv_pathlike_object(self, simple_csv):
        """Should accept Path objects."""
        frame = ar.read_csv(Path(simple_csv))
        assert frame.shape == (2, 3)

    def test_read_csv_trim_headers_enabled(self):
        """Should trim whitespace from headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spaces.csv")
            with open(path, "w") as f:
                f.write(" name , age \nAlice,30\n")
            frame = ar.read_csv(path, trim_headers=True)
            assert frame.columns == ["name", "age"]

    def test_read_csv_no_header(self):
        """Should read without header when has_header=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_hdr.csv")
            with open(path, "w") as f:
                f.write("Alice,30\nBob,25\n")
            frame = ar.read_csv(path, has_header=False)
            assert frame.shape == (2, 2)

    def test_read_csv_with_dtype(self, simple_csv):
        """Should apply explicit dtype mapping."""
        frame = ar.read_csv(simple_csv, dtype={"age": "int64"})
        assert frame.dtypes["age"] == "int64"


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""

    @pytest.fixture
    def large_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "large.csv")
            with open(path, "w") as f:
                f.write("id,value\n")
                for i in range(100):
                    f.write(f"{i},{i*10}\n")
            yield path

    @pytest.fixture
    def tsv_for_chunking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.tsv")
            with open(path, "w") as f:
                f.write("id\tvalue\n")
                for i in range(50):
                    f.write(f"{i}\t{i*10}\n")
            yield path

    def test_read_csv_chunked_respects_chunksize(self, large_csv):
        """Should yield chunks of correct size."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=25))
        assert len(chunks) == 4
        assert all(chunk.shape[0] <= 25 for chunk in chunks)

    def test_read_csv_chunked_respects_nrows(self, large_csv):
        """Should respect nrows limit across all chunks."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=30, nrows=50))
        total = sum(chunk.shape[0] for chunk in chunks)
        assert total == 50

    def test_read_csv_chunked_auto_detects_tab(self, tsv_for_chunking):
        """Should auto-detect tab for .tsv files."""
        chunks = list(ar.read_csv_chunked(tsv_for_chunking, chunksize=10))
        assert len(chunks) >= 1
        assert chunks[0].columns == ["id", "value"]

    def test_read_csv_chunked_explicit_delimiter(self, tsv_for_chunking):
        """Should use explicit delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_for_chunking, delimiter="\t", chunksize=10))
        assert all(chunk.shape[1] == 2 for chunk in chunks)

    def test_read_csv_chunked_with_usecols(self, large_csv):
        """Should respect usecols in each chunk."""
        chunks = list(ar.read_csv_chunked(large_csv, usecols=["id"], chunksize=20))
        assert all(chunk.shape[1] == 1 for chunk in chunks)
        assert all(chunk.columns == ["id"] for chunk in chunks)

    def test_read_csv_chunked_skip_rows(self, large_csv):
        """Should skip data rows after header."""
        chunks_skipped = list(ar.read_csv_chunked(large_csv, skip_rows=10, chunksize=20))
        chunks_no_skip = list(ar.read_csv_chunked(large_csv, skip_rows=0, chunksize=20))
        total_skipped = sum(c.shape[0] for c in chunks_skipped)
        total_no_skip = sum(c.shape[0] for c in chunks_no_skip)
        assert total_skipped == 90  # 100 - 10 skipped
        assert total_no_skip == 100

    def test_read_csv_chunked_pathlike(self, large_csv):
        """Should accept Path objects."""
        chunks = list(ar.read_csv_chunked(Path(large_csv), chunksize=30))
        assert len(chunks) >= 1

    def test_read_csv_chunked_empty_data(self):
        """Should handle file with only headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.csv")
            with open(path, "w") as f:
                f.write("a,b\n")
            chunks = list(ar.read_csv_chunked(path, chunksize=10))
            assert len(chunks) == 0

    def test_read_csv_chunked_total_rows_match(self, large_csv):
        """Should read all rows across chunks."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=25))
        total = sum(c.shape[0] for c in chunks)
        assert total == 100
