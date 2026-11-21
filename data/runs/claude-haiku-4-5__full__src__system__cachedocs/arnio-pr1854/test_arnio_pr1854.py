"""
System (end-to-end) tests for arnio.io.read_csv_chunked.

These tests drive complete user workflows through arnio's public API,
using real files on disk and validating observable outcomes like
output files, frames, error messages, and exit codes.
"""

import arnio as ar
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestReadCsvChunked:
    """System tests for ar.read_csv_chunked streaming workflow."""

    def test_basic_chunked_reading(self, tmp_path):
        """Test basic chunked reading produces correct chunk sizes."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": range(1, 26),
            "name": [f"user_{i}" for i in range(1, 26)],
            "score": np.random.rand(25),
        })
        df.to_csv(csv_path, index=False)

        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10))

        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5

        for chunk in chunks:
            assert chunk.columns == ["id", "name", "score"]
            assert chunk.shape[1] == 3

    def test_chunked_pipeline_and_export(self, tmp_path):
        """Read chunks, apply pipeline to each, and write cleaned output file."""
        csv_path = tmp_path / "messy.csv"
        output_path = tmp_path / "clean.csv"

        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Charlie  ", "  David  "],
            "age": [25, 30, None, 35],
        })
        df.to_csv(csv_path, index=False)

        # Process chunks: strip whitespace and fill nulls
        all_clean = []
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=2):
            clean = ar.pipeline(chunk, [
                ("strip_whitespace",),
                ("fill_nulls", {"value": 0, "subset": ["age"]}),
            ])
            all_clean.append(ar.to_pandas(clean))

        result_df = pd.concat(all_clean, ignore_index=True)
        result_df.to_csv(output_path, index=False)

        # Verify output file exists and contains expected data
        assert output_path.exists()
        read_back = pd.read_csv(output_path)
        assert read_back["name"].tolist() == ["Alice", "Bob", "Charlie", "David"]
        assert read_back["age"].tolist() == [25.0, 30.0, 0.0, 35.0]

    def test_chunked_schema_validation_with_global_row_numbers(self, tmp_path):
        """Validate chunks against schema and compute global row indices."""
        csv_path = tmp_path / "users.csv"
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "email": ["alice@test.com", "bob@test.com", "invalid", "david@test.com", "eve@test.com"],
            "age": [25, 30, 35, 40, 45],
        })
        df.to_csv(csv_path, index=False)

        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=18),
        })

        chunksize = 2
        all_issues = []
        total_rows = 0

        for chunk_num, chunk in enumerate(ar.read_csv_chunked(str(csv_path), chunksize=chunksize)):
            result = ar.validate(chunk, schema)
            total_rows += chunk.shape[0]
            for issue in result.issues:
                # Convert local row_index to global row number
                global_row = chunk_num * chunksize + issue.row_index
                all_issues.append({
                    "global_row": global_row,
                    "column": issue.column,
                    "rule": issue.rule,
                })

        assert total_rows == 5
        # Should identify invalid email on row 3
        email_issues = [i for i in all_issues if i["column"] == "email"]
        assert len(email_issues) > 0
        assert any(i["global_row"] == 3 for i in email_issues)

    def test_tsv_auto_detection_and_profiling(self, tmp_path):
        """Auto-detect tab delimiter for .tsv and profile each chunk."""
        tsv_path = tmp_path / "data.tsv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "value": [100, 200, 300, 400],
        })
        df.to_csv(tsv_path, sep="\t", index=False)

        profiles = []
        for chunk in ar.read_csv_chunked(str(tsv_path), chunksize=2):
            profile = ar.profile(chunk)
            profiles.append(profile)

        assert len(profiles) == 2
        assert profiles[0].row_count == 2
        assert profiles[1].row_count == 2
        assert len(profiles[0].columns) == 2

    def test_usecols_filtering_with_pandas_accessor(self, tmp_path):
        """Read specific columns and use pandas .arnio accessor for cleaning."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["  Alice  ", "  Bob  ", "  Charlie  ", "  David  "],
            "age": [25, 30, 35, 40],
            "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com"],
        })
        df.to_csv(csv_path, index=False)

        all_cleaned = []
        for chunk in ar.read_csv_chunked(str(csv_path), usecols=["name", "age"], chunksize=2):
            assert set(chunk.columns) == {"name", "age"}
            # Convert to pandas and use accessor
            pdf = ar.to_pandas(chunk)
            cleaned_pdf = pdf.arnio.clean(strip_whitespace=True)
            all_cleaned.append(cleaned_pdf)

        result = pd.concat(all_cleaned, ignore_index=True)
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David"]
        assert result.shape[0] == 4

    def test_nrows_total_limit(self, tmp_path):
        """Stop reading after total nrows limit across all chunks."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": range(1, 31),
            "value": range(100, 130),
        })
        df.to_csv(csv_path, index=False)

        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=5, nrows=12))

        total = sum(c.shape[0] for c in chunks)
        assert total == 12

    def test_skip_rows_with_decimal_separator(self, tmp_path):
        """Skip data rows and handle European decimal separator in chunks."""
        csv_path = tmp_path / "prices.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id;price\n")
            f.write("1;10,99\n")
            f.write("2;20,50\n")
            f.write("3;15,99\n")
            f.write("4;30,00\n")
            f.write("5;25,75\n")

        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            delimiter=";",
            decimal_separator=",",
            skip_rows=1,
            chunksize=2
        ))

        result = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        # Should skip first data row (id=1), leaving 4 rows
        assert result.shape[0] == 4
        # Decimal values should be properly parsed
        assert any(v == 20.50 for v in result["price"].tolist())

    def test_custom_null_values_in_chunks(self, tmp_path):
        """Handle custom null value sentinels in chunked reading."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            f.write("id,score,notes\n")
            f.write("1,100,OK\n")
            f.write("2,N/A,Missing\n")
            f.write("3,UNKNOWN,Unknown\n")
            f.write("4,85,OK\n")

        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            null_values=["N/A", "UNKNOWN"],
            chunksize=2
        ))

        result = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        # Custom sentinels should become null
        assert pd.isna(result.iloc[1]["score"])
        assert pd.isna(result.iloc[2]["score"])
        assert result.iloc[0]["score"] == 100.0

    def test_encoding_non_utf8_latin1(self, tmp_path):
        """Read CSV with non-UTF-8 encoding (latin-1) in chunks."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["José", "François", "Müller"],
        })
        df.to_csv(csv_path, index=False, encoding="latin-1")

        chunks = list(ar.read_csv_chunked(str(csv_path), encoding="latin-1", chunksize=2))

        result = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        assert "José" in result["name"].values
        assert "François" in result["name"].values
        assert result.shape[0] == 3

    def test_permissive_mode_with_bad_lines(self, tmp_path):
        """Handle malformed rows in permissive mode with on_bad_lines='skip'."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w") as f:
            f.write("id,name,value\n")
            f.write("1,Alice,100\n")
            f.write("2,Bob,200\n")
            f.write("3,Charlie\n")  # Missing value field
            f.write("4,David,400\n")

        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            mode="permissive",
            on_bad_lines="skip",
            chunksize=2
        ))

        result = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        # Should have read valid rows
        assert result.shape[0] >= 3

    def test_global_row_indexing_validation_issues(self, tmp_path):
        """Verify row_index is local to chunk; compute global row numbers for violations."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "value": [10, 20, 999, 40, 999, 60],  # 999 exceeds max
        })
        df.to_csv(csv_path, index=False)

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "value": ar.Int64(min=0, max=500),
        })

        chunksize = 2
        violation_global_rows = []

        for chunk_num, chunk in enumerate(ar.read_csv_chunked(str(csv_path), chunksize=chunksize)):
            result = ar.validate(chunk, schema)
            for issue in result.issues:
                if "max" in issue.rule:
                    global_row = chunk_num * chunksize + issue.row_index
                    violation_global_rows.append(global_row)

        # Rows 3 and 5 have value=999 (should be caught)
        assert len(violation_global_rows) > 0

    def test_unsupported_file_format_raises_error(self, tmp_path):
        """Verify error handling for unsupported file formats."""
        json_path = tmp_path / "data.json"
        json_path.write_text('{"id": 1}')

        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(str(json_path), chunksize=10))

    def test_chunked_profiling_aggregation(self, tmp_path):
        """Profile chunks and manually aggregate quality metrics."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "score": [100, None, 85, None, 92, 88],
            "category": ["A", "A", "B", "B", "A", "B"],
        })
        df.to_csv(csv_path, index=False)

        total_rows = 0
        total_nulls_score = 0

        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=3):
            report = ar.profile(chunk)
            total_rows += report.row_count
            total_nulls_score += report.columns["score"].null_count

        assert total_rows == 6
        assert total_nulls_score == 2

    def test_empty_csv_file_with_headers(self, tmp_path):
        """Handle CSV file with only headers (no data rows)."""
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame(columns=["id", "name"])
        df.to_csv(csv_path, index=False)

        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10))

        # Empty CSV may yield one empty chunk or none
        if chunks:
            assert chunks[0].shape[0] == 0
        else:
            assert len(chunks) == 0

    def test_large_file_streaming_bounded_memory(self, tmp_path):
        """Simulate processing large file in memory-bounded chunks."""
        csv_path = tmp_path / "large.csv"
        # Create 1000-row file
        rows = 1000
        df = pd.DataFrame({
            "id": range(rows),
            "category": [f"cat_{i % 5}" for i in range(rows)],
            "amount": np.random.randint(1, 1000, rows),
        })
        df.to_csv(csv_path, index=False)

        # Process in 100-row chunks
        chunksize = 100
        total_processed = 0
        chunk_count = 0

        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=chunksize):
            chunk_count += 1
            total_processed += chunk.shape[0]

            # Verify chunk size is bounded
            assert chunk.shape[0] <= chunksize
            # Verify columns are correct
            assert chunk.columns == ["id", "category", "amount"]

        assert total_processed == rows
        assert chunk_count == 10
