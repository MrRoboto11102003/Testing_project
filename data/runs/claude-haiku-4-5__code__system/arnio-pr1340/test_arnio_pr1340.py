import pytest
import io
import tempfile
import os
from pathlib import Path

import arnio as ar
from arnio.integrations.sklearn import ArnioCleaner


class TestArnioCleaner:
    """Tests for ArnioCleaner.get_feature_names_out sklearn compatibility."""
    
    def test_get_feature_names_out_returns_fitted_names_when_none(self, tmp_path):
        """When input_features is None, return fitted output feature names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        df = ar.read_csv(str(csv_file))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out(None)
        assert result is not None
        assert len(result) == 3
        
    def test_get_feature_names_out_with_matching_features(self, tmp_path):
        """Accept matching input_features and return fitted output names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        df = ar.read_csv(str(csv_file))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out(["col1", "col2", "col3"])
        assert result is not None
        assert len(result) == 3
        
    def test_get_feature_names_out_rejects_string_input(self, tmp_path):
        """TypeError when input_features is a string instead of sequence."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        
        df = ar.read_csv(str(csv_file))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        with pytest.raises(TypeError, match="must be a sequence"):
            cleaner.get_feature_names_out("col1")
            
    def test_get_feature_names_out_rejects_wrong_length(self, tmp_path):
        """ValueError when input_features has wrong length."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n")
        
        df = ar.read_csv(str(csv_file))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        with pytest.raises(ValueError, match="should have length"):
            cleaner.get_feature_names_out(["col1", "col2"])
            
    def test_get_feature_names_out_rejects_mismatched_names(self, tmp_path):
        """ValueError when input_features don't match fitted column names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n")
        
        df = ar.read_csv(str(csv_file))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        with pytest.raises(ValueError, match="must match columns"):
            cleaner.get_feature_names_out(["other1", "other2", "other3"])


class TestReadCsv:
    """Tests for arnio.io.read_csv with complete user workflows."""
    
    def test_read_csv_basic_workflow(self, tmp_path):
        """Read basic CSV file and verify shape."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,score\nAlice,30,95.5\nBob,25,87.3\nCharlie,35,92.0\n")
        
        df = ar.read_csv(str(csv_file))
        assert df.shape == (3, 3)
        
    def test_read_csv_with_usecols_and_nrows(self, tmp_path):
        """Read subset of columns and rows using usecols and nrows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,score,dept\nAlice,30,95.5,A\nBob,25,87.3,B\nCharlie,35,92.0,A\nDave,28,78.5,C\n")
        
        df = ar.read_csv(str(csv_file), usecols=["name", "score"], nrows=2)
        assert df.shape == (2, 2)
        
    def test_read_csv_tsv_auto_detection_and_explicit_delimiter(self, tmp_path):
        """Auto-detect TSV but allow explicit delimiter override."""
        # Test 1: .tsv file with auto-detection
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("name\tage\tscope\nAlice\t30\t95\nBob\t25\t87\n")
        df1 = ar.read_csv(str(tsv_file))
        assert df1.shape == (2, 3)
        
        # Test 2: .tsv file with explicit comma delimiter
        csv_file = tmp_path / "data_comma.tsv"
        csv_file.write_text("name,age,score\nAlice,30,95\nBob,25,87\n")
        df2 = ar.read_csv(str(csv_file), delimiter=",")
        assert df2.shape == (2, 3)
        
    def test_read_csv_pathlib_and_file_like_objects(self, tmp_path):
        """Accept both pathlib.Path and file-like objects."""
        # Test 1: pathlib.Path
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        df1 = ar.read_csv(csv_file)
        assert df1.shape == (2, 2)
        
        # Test 2: StringIO file-like object
        csv_data = "name,age\nCharlie,35\nDave,40\n"
        file_obj = io.StringIO(csv_data)
        df2 = ar.read_csv(file_obj)
        assert df2.shape == (2, 2)
        
    def test_read_csv_dtype_null_values_and_decimal_separator(self, tmp_path):
        """Specify dtype, null values, and decimal separator."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,count,score\nAlice,100,95,5\nBob,200,N/A\nCharlie,150,88,2\n")
        
        df = ar.read_csv(
            str(csv_file),
            dtype={"count": "int64"},
            null_values=["N/A"],
            decimal_separator=","
        )
        assert df.shape == (3, 3)
        
    def test_read_csv_skiprows_parameter(self, tmp_path):
        """Skip initial comment lines before header."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("# Version 1.0\n# Generated 2025-01-01\nname,age\nAlice,30\nBob,25\n")
        
        df = ar.read_csv(str(csv_file), skiprows=2)
        assert df.shape == (2, 2)
        
    def test_read_csv_permissive_mode_skip_bad_lines(self, tmp_path):
        """Handle malformed rows in permissive mode by skipping them."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,score\nAlice,30,95\nBob,25\nCharlie,35,88\n")
        
        df = ar.read_csv(str(csv_file), mode="permissive", on_bad_lines="skip")
        # Should successfully read the well-formed rows
        assert df.shape[0] >= 1


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked with chunked streaming workflow."""
    
    def test_read_csv_chunked_basic_workflow(self, tmp_path):
        """Read large CSV file in chunks and process incrementally."""
        csv_file = tmp_path / "large.csv"
        rows = ["name,age,score"]
        for i in range(25):
            rows.append(f"person{i},{20+i},{80+i%20}")
        csv_file.write_text("\n".join(rows))
        
        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=10))
        assert len(chunks) >= 2
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 25
        
    def test_read_csv_chunked_chunksize_respected(self, tmp_path):
        """Each chunk respects the chunksize limit."""
        csv_file = tmp_path / "large.csv"
        rows = ["name,age"]
        for i in range(100):
            rows.append(f"person{i},{20+i}")
        csv_file.write_text("\n".join(rows))
        
        all_chunks_ok = True
        for chunk in ar.read_csv_chunked(str(csv_file), chunksize=25):
            if chunk.shape[0] > 25:
                all_chunks_ok = False
                break
        assert all_chunks_ok
        
    def test_read_csv_chunked_with_nrows_and_skip_rows(self, tmp_path):
        """Limit total rows and skip initial rows in chunked reading."""
        csv_file = tmp_path / "data.csv"
        rows = ["name,age"]
        for i in range(50):
            rows.append(f"person{i},{20+i}")
        csv_file.write_text("\n".join(rows))
        
        total_rows = 0
        for chunk in ar.read_csv_chunked(str(csv_file), chunksize=10, skip_rows=5, nrows=20):
            total_rows += chunk.shape[0]
        assert total_rows <= 20
        
    def test_read_csv_chunked_consistent_schema_across_chunks(self, tmp_path):
        """All chunks maintain consistent column schema and types."""
        csv_file = tmp_path / "large.csv"
        rows = ["name,count,score"]
        for i in range(50):
            rows.append(f"person{i},{100+i},{80.5+i}")
        csv_file.write_text("\n".join(rows))
        
        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=10))
        expected_cols = chunks[0].shape[1]
        for chunk in chunks:
            assert chunk.shape[1] == expected_cols
            
    def test_read_csv_chunked_skiprows_vs_skip_rows_conflict(self, tmp_path):
        """Conflicting skiprows and skip_rows parameters raise ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\nCharlie,35\n")
        
        with pytest.raises(ValueError, match="Conflicting values"):
            list(ar.read_csv_chunked(str(csv_file), skip_rows=1, skiprows=2))
