"""
End-to-end pytest tests for arnio.io.read_csv_chunked.

Tests exercise the full user workflow: creating test data, loading in chunks,
applying cleaning/validation/profiling, and converting results.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestReadCsvChunked:
    """End-to-end tests for arnio.io.read_csv_chunked"""
    
    def test_chunked_basic_multiple_chunks(self, tmp_path):
        """Test basic chunked reading with multiple chunks."""
        csv_path = tmp_path / "data.csv"
        rows = 25
        df = pd.DataFrame({
            "id": range(1, rows + 1),
            "name": [f"user_{i}" for i in range(1, rows + 1)],
            "value": np.random.rand(rows),
        })
        df.to_csv(csv_path, index=False)
        
        # Read in chunks of size 10
        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10))
        
        # Verify chunk count and sizes (10, 10, 5)
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5
        
        # All chunks share same column names
        for chunk in chunks:
            assert chunk.columns == ["id", "name", "value"]
        
        # Verify data integrity when concatenated
        result_df = pd.concat(
            [ar.to_pandas(c) for c in chunks], ignore_index=True
        )
        assert len(result_df) == rows
        assert result_df["id"].tolist() == list(range(1, rows + 1))
    
    def test_chunked_respects_custom_chunksize(self, tmp_path):
        """Test that chunking respects the chunksize parameter."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": range(100)}).to_csv(csv_path, index=False)
        
        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=25))
        
        assert len(chunks) == 4
        assert all(c.shape[0] == 25 for c in chunks)
    
    def test_chunked_nrows_limit(self, tmp_path):
        """Test that nrows parameter limits total rows across chunks."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": range(100)}).to_csv(csv_path, index=False)
        
        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10, nrows=35))
        
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 35
    
    def test_chunked_skip_rows(self, tmp_path):
        """Test that skip_rows parameter skips data rows after header."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": range(1, 21)}).to_csv(csv_path, index=False)
        
        # Skip first 10 data rows
        chunks = list(ar.read_csv_chunked(str(csv_path), skip_rows=10, chunksize=5))
        
        result_df = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        assert result_df["x"].tolist() == list(range(11, 21))
    
    def test_chunked_usecols_selection(self, tmp_path):
        """Test that usecols selects only specified columns."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": range(1, 11),
            "name": [f"user_{i}" for i in range(1, 11)],
            "email": [f"user{i}@example.com" for i in range(1, 11)],
            "extra": ["x"] * 10,
        })
        df.to_csv(csv_path, index=False)
        
        chunks = list(ar.read_csv_chunked(str(csv_path), usecols=["id", "name"]))
        
        assert len(chunks) == 1
        assert chunks[0].columns == ["id", "name"]
        result_df = ar.to_pandas(chunks[0])
        assert set(result_df.columns) == {"id", "name"}
    
    def test_chunked_explicit_dtype(self, tmp_path):
        """Test that dtype parameter forces column types."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5"],
            "value": ["10.5", "20.5", "30.5", "40.5", "50.5"],
        })
        df.to_csv(csv_path, index=False)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            dtype={"id": "int64", "value": "float64"}
        ))
        
        result_df = ar.to_pandas(chunks[0])
        assert result_df["id"].dtype == np.int64
        assert result_df["value"].dtype == np.float64
    
    def test_chunked_tsv_auto_delimiter(self, tmp_path):
        """Test that .tsv extension auto-detects tab delimiter."""
        tsv_path = tmp_path / "data.tsv"
        df = pd.DataFrame({
            "id": range(1, 6),
            "name": ["alice", "bob", "charlie", "diana", "eve"],
        })
        df.to_csv(tsv_path, sep="\t", index=False)
        
        # Read TSV without specifying delimiter
        chunks = list(ar.read_csv_chunked(str(tsv_path), chunksize=2))
        
        result_df = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        assert len(result_df) == 5
        assert result_df.columns.tolist() == ["id", "name"]
    
    def test_chunked_null_values_sentinel(self, tmp_path):
        """Test custom null value sentinels are recognized."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            f.write("id,name,score\n")
            f.write("1,Alice,100\n")
            f.write("2,MISSING,NA\n")
            f.write("3,Charlie,--\n")
        
        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            null_values=["MISSING", "NA", "--"]
        ))
        
        result_df = ar.to_pandas(chunks[0])
        # Custom sentinels should be treated as nulls
        assert pd.isna(result_df.loc[1, "name"])
        assert pd.isna(result_df.loc[1, "score"])
        assert pd.isna(result_df.loc[2, "score"])
    
    def test_chunked_on_bad_lines_skip(self, tmp_path):
        """Test on_bad_lines='skip' drops rows with extra fields."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            f.write("id,name\n")
            f.write("1,Alice\n")
            f.write("2,Bob,extra_field\n")  # Extra field — bad row
            f.write("3,Charlie\n")
        
        # Strict mode + skip: drop rows with extra fields
        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            mode="strict",
            on_bad_lines="skip"
        ))
        
        result_df = ar.to_pandas(chunks[0])
        # Row 2 with extra field should be skipped
        assert len(result_df) == 2
        assert 2 not in result_df["id"].values
    
    def test_chunked_pipeline_cleaning(self, tmp_path):
        """Test applying cleaning pipeline to each chunk."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "name": ["  Alice  ", " Bob ", "Charlie"],
            "score": [100, 50, 75],
        })
        df.to_csv(csv_path, index=False)
        
        all_cleaned = []
        for chunk in ar.read_csv_chunked(str(csv_path)):
            cleaned = ar.pipeline(chunk, [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
            ])
            all_cleaned.append(ar.to_pandas(cleaned))
        
        result_df = pd.concat(all_cleaned, ignore_index=True)
        # Names should be stripped and lowercased
        assert result_df["name"].tolist() == ["alice", "bob", "charlie"]
    
    def test_chunked_schema_validation(self, tmp_path):
        """Test validating each chunk against a schema with global row tracking."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "email": [
                "alice@example.com", "bob@test.org", "charlie@example.com",
                "diana@example.com", "eve@example.com"
            ],
        })
        df.to_csv(csv_path, index=False)
        
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        all_issues = []
        total_rows = 0
        
        for chunk_num, chunk in enumerate(ar.read_csv_chunked(str(csv_path))):
            result = ar.validate(chunk, schema)
            total_rows += chunk.shape[0]
            
            for issue in result.issues:
                global_row = chunk_num * 5 + issue.row_index
                all_issues.append({
                    "global_row": global_row,
                    "column": issue.column,
                })
        
        assert total_rows == 5
        # Valid emails should produce no issues
        assert len(all_issues) == 0
    
    def test_chunked_profile_aggregation(self, tmp_path):
        """Test profiling chunks and aggregating statistics."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": range(1, 101),
            "category": ["A", "B"] * 50,
            "value": np.random.rand(100),
        })
        df.to_csv(csv_path, index=False)
        
        total_rows = 0
        all_reports = []
        
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=25):
            report = ar.profile(chunk)
            all_reports.append(report)
            total_rows += report.row_count
        
        assert len(all_reports) == 4  # 100 / 25 = 4 chunks
        assert total_rows == 100
        # Each report should have the same column names
        for report in all_reports:
            assert set(report.columns.keys()) == {"id", "category", "value"}
    
    def test_chunked_edge_case_header_only(self, tmp_path):
        """Test edge case: file with only header row."""
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w") as f:
            f.write("id,name\n")
        
        chunks = list(ar.read_csv_chunked(str(csv_path)))
        # Header-only file yields no chunks or empty chunks
        assert len(chunks) == 0 or all(c.shape[0] == 0 for c in chunks)
    
    def test_chunked_error_invalid_chunksize(self, tmp_path):
        """Test error: negative chunksize raises ValueError."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError):
            list(ar.read_csv_chunked(str(csv_path), chunksize=-1))
    
    def test_chunked_error_invalid_mode(self, tmp_path):
        """Test error: invalid mode parameter raises ValueError."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError):
            list(ar.read_csv_chunked(str(csv_path), mode="invalid_mode"))
