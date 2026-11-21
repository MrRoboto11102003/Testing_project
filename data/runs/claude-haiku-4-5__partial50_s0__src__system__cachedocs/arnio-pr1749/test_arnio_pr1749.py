import arnio as ar
import io
import os
import pandas as pd
import pytest
from pathlib import Path


class TestReadCsvChunked:
    """End-to-end tests for ar.read_csv_chunked() streaming CSV reader"""

    def test_basic_chunked_reading_multiple_chunks(self, tmp_path):
        """Read CSV file in chunks and verify all data is recovered"""
        csv_file = tmp_path / "data.csv"
        # Create a 100-row CSV
        data = {"id": range(1, 101), "value": range(100, 200)}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        # Read in chunks of 25 rows
        all_rows = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=25):
            chunk_df = ar.to_pandas(chunk)
            all_rows.append(chunk_df)

        # Reconstruct full dataframe
        result = pd.concat(all_rows, ignore_index=True)
        assert len(result) == 100
        assert list(result.columns) == ["id", "value"]
        assert result["id"].min() == 1
        assert result["id"].max() == 100

    def test_chunk_size_respected(self, tmp_path):
        """Verify chunks respect the specified chunksize parameter"""
        csv_file = tmp_path / "large.csv"
        data = {"col": range(1, 251)}  # 250 rows
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        chunksize = 50
        chunk_sizes = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=chunksize):
            chunk_sizes.append(chunk.shape[0])

        # Should have 5 chunks of exactly 50 rows each
        assert len(chunk_sizes) == 5
        assert all(size == 50 for size in chunk_sizes)
        assert sum(chunk_sizes) == 250

    def test_tsv_auto_delimiter_detection(self, tmp_path):
        """Test automatic tab delimiter detection for .tsv files"""
        tsv_file = tmp_path / "data.tsv"
        lines = "name\tage\tcity\nAlice\t30\tParis\nBob\t25\tLondon\nCharlie\t35\tTokyo"
        tsv_file.write_text(lines)

        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=2))
        assert len(chunks) == 2  # 3 rows / 2 per chunk

        result = pd.concat([ar.to_pandas(c) for c in chunks], ignore_index=True)
        assert list(result.columns) == ["name", "age", "city"]
        assert len(result) == 3
        assert "Alice" in result["name"].values

    def test_explicit_delimiter_override(self, tmp_path):
        """Override delimiter explicitly even for .tsv files"""
        csv_file = tmp_path / "data.tsv"
        # Actually pipe-delimited but with .tsv extension
        lines = "name|age|city\nAlice|30|Paris\nBob|25|London"
        csv_file.write_text(lines)

        chunks = list(ar.read_csv_chunked(csv_file, delimiter="|", chunksize=5))
        result = ar.to_pandas(chunks[0])
        assert list(result.columns) == ["name", "age", "city"]
        assert len(result) == 2

    def test_usecols_select_subset(self, tmp_path):
        """Select specific columns with usecols parameter"""
        csv_file = tmp_path / "data.csv"
        data = {
            "id": range(1, 51),
            "name": ["user"] * 50,
            "score": range(50),
            "timestamp": ["2024-01-01"] * 50,
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        all_rows = []
        for chunk in ar.read_csv_chunked(csv_file, usecols=["id", "score"], chunksize=20):
            all_rows.append(ar.to_pandas(chunk))

        result = pd.concat(all_rows, ignore_index=True)
        assert list(result.columns) == ["id", "score"]
        assert len(result) == 50
        assert "name" not in result.columns

    def test_nrows_limit_total_read(self, tmp_path):
        """Limit total rows read across all chunks with nrows parameter"""
        csv_file = tmp_path / "data.csv"
        data = {"id": range(1, 1001)}  # 1000 rows total
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        all_rows = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=100, nrows=250):
            all_rows.append(ar.to_pandas(chunk))

        result = pd.concat(all_rows, ignore_index=True)
        assert len(result) == 250
        assert result["id"].max() == 250

    def test_skip_rows_parameter(self, tmp_path):
        """Skip rows after the header with skip_rows parameter"""
        csv_file = tmp_path / "data.csv"
        data = {"id": range(1, 101), "value": range(100)}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        all_rows = []
        for chunk in ar.read_csv_chunked(csv_file, skip_rows=20, chunksize=25):
            all_rows.append(ar.to_pandas(chunk))

        result = pd.concat(all_rows, ignore_index=True)
        assert len(result) == 80
        assert result["id"].min() == 21  # First 20 rows were skipped

    def test_cleaning_pipeline_on_each_chunk(self, tmp_path):
        """Apply cleaning pipeline to each chunk independently"""
        csv_file = tmp_path / "messy.csv"
        # Create messy data with leading/trailing whitespace
        data = {
            "name": [" alice ", " bob ", " charlie "] * 10,
            "score": [90, 80, 70] * 10,
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        cleaned_rows = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=10):
            # Strip whitespace from each chunk
            clean = ar.pipeline(chunk, [("strip_whitespace",)])
            cleaned_rows.append(ar.to_pandas(clean))

        result = pd.concat(cleaned_rows, ignore_index=True)
        # Verify whitespace was stripped
        assert all(name == name.strip() for name in result["name"])
        assert "alice" in result["name"].values

    def test_schema_validation_across_chunks(self, tmp_path):
        """Validate data schema and collect issues across multiple chunks"""
        csv_file = tmp_path / "data.csv"
        data = {
            "id": range(1, 101),
            "email": [f"user{i}@example.com" for i in range(100)],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
            }
        )

        all_issues = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=25):
            result = ar.validate(chunk, schema)
            if not result.passed:
                all_issues.extend(result.issues)

        # All rows should pass validation
        assert len(all_issues) == 0

    def test_read_from_stringio_file_like(self, tmp_path):
        """Read CSV from file-like object (StringIO) in chunks"""
        csv_data = "id,value\n1,100\n2,200\n3,300\n4,400\n5,500"
        file_obj = io.StringIO(csv_data)

        all_rows = []
        for chunk in ar.read_csv_chunked(file_obj, chunksize=2):
            all_rows.append(ar.to_pandas(chunk))

        result = pd.concat(all_rows, ignore_index=True)
        assert len(result) == 5
        assert result["id"].tolist() == [1, 2, 3, 4, 5]
        assert len(all_rows) == 3  # 5 rows / 2 per chunk -> 3 chunks

    def test_pathlib_path_object(self, tmp_path):
        """Read from pathlib.Path object instead of string path"""
        csv_file = tmp_path / "data.csv"
        data = {"value": range(1, 21)}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        all_rows = []
        for chunk in ar.read_csv_chunked(Path(csv_file), chunksize=5):
            all_rows.append(ar.to_pandas(chunk))

        result = pd.concat(all_rows, ignore_index=True)
        assert len(result) == 20
        assert len(all_rows) == 4  # 4 chunks of 5 rows

    def test_profile_quality_per_chunk(self, tmp_path):
        """Profile data quality metrics within each chunk"""
        csv_file = tmp_path / "data.csv"
        # Create data with some null values
        data = {
            "id": range(1, 51),
            "optional": [None if i % 5 == 0 else f"val{i}" for i in range(50)],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        chunk_count = 0
        for chunk in ar.read_csv_chunked(csv_file, chunksize=10):
            # Profile each chunk
            report = ar.profile(chunk)
            # Verify report structure
            assert "id" in report.columns
            assert "optional" in report.columns
            assert report.row_count <= 10
            chunk_count += 1

        assert chunk_count == 5  # 50 rows / 10 per chunk

    def test_write_chunk_to_new_file(self, tmp_path):
        """Read chunk and write it to a new CSV file"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        data = {"id": range(1, 26), "value": range(25)}
        df = pd.DataFrame(data)
        df.to_csv(input_file, index=False)

        # Read first chunk and write it
        for i, chunk in enumerate(ar.read_csv_chunked(input_file, chunksize=25)):
            if i == 0:
                ar.write_csv(chunk, output_file)
            break

        # Verify output file was created and contains the chunk
        assert output_file.exists()
        result = pd.read_csv(output_file)
        assert len(result) == 25
        assert list(result.columns) == ["id", "value"]
        assert result["id"].min() == 1
        assert result["id"].max() == 25

    def test_encoding_latin1(self, tmp_path):
        """Read CSV with non-UTF8 encoding (latin-1)"""
        csv_file = tmp_path / "data.csv"
        # Create data with accented characters
        data = {"name": ["José", "María", "François", "Müller"]}
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding="latin-1")

        chunks = list(ar.read_csv_chunked(csv_file, encoding="latin-1"))
        result = ar.to_pandas(chunks[0])
        # Verify accented characters were read correctly
        assert "José" in result["name"].values
        assert "François" in result["name"].values

    def test_decimal_separator_european(self, tmp_path):
        """Handle European decimal separator (comma) in numeric columns"""
        csv_file = tmp_path / "data.csv"
        lines = [
            "id,amount",
            "1,10.50",
            "2,20.75",
            "3,30.25",
        ]
        csv_file.write_text("\n".join(lines))

        chunks = list(ar.read_csv_chunked(csv_file, decimal_separator="."))
        result = ar.to_pandas(chunks[0])
        assert len(result) == 3
        assert "amount" in result.columns
        assert result["amount"].dtype in ["float64", "Float64"]
