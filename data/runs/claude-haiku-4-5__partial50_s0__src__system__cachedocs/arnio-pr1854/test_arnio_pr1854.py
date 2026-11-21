import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestReadCsvChunked:
    """System end-to-end tests for ar.read_csv_chunked()"""

    def test_basic_chunked_reading(self, tmp_path):
        """Read a CSV in chunks and verify all data is yielded correctly"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": range(1, 26),
            "value": [f"val_{i}" for i in range(1, 26)]
        })
        df.to_csv(csv_file, index=False)

        chunks = list(ar.read_csv_chunked(csv_file, chunksize=10))

        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5

        all_frames = [ar.to_pandas(chunk) for chunk in chunks]
        combined = pd.concat(all_frames, ignore_index=True)
        pd.testing.assert_frame_equal(combined, df)

    def test_chunksize_parameter(self, tmp_path):
        """Verify chunks respect the specified chunksize parameter"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"x": range(100)})
        df.to_csv(csv_file, index=False)

        chunks_5 = list(ar.read_csv_chunked(csv_file, chunksize=5))
        chunks_25 = list(ar.read_csv_chunked(csv_file, chunksize=25))

        assert len(chunks_5) == 20  # 100 / 5
        assert len(chunks_25) == 4  # 100 / 25
        
        # Verify non-final chunks respect size
        for chunk in chunks_5[:-1]:
            assert chunk.shape[0] == 5
        for chunk in chunks_25[:-1]:
            assert chunk.shape[0] == 25

    def test_tsv_auto_detection(self, tmp_path):
        """Verify .tsv files automatically use tab delimiter"""
        tsv_file = tmp_path / "test.tsv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        df.to_csv(tsv_file, index=False, sep="\t")

        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=1))

        assert len(chunks) == 3
        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        pd.testing.assert_frame_equal(combined, df)

    def test_delimiter_override(self, tmp_path):
        """Explicit delimiter overrides extension-based auto-detection"""
        tsv_file = tmp_path / "test.tsv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(tsv_file, index=False, sep=",")  # comma, not tab

        chunks = list(ar.read_csv_chunked(tsv_file, delimiter=",", chunksize=1))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        pd.testing.assert_frame_equal(combined, df)

    def test_usecols_filtering(self, tmp_path):
        """Read only specified columns via usecols parameter"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "city": ["NYC", "LA", "Chicago", "Boston"]
        })
        df.to_csv(csv_file, index=False)

        chunks = list(ar.read_csv_chunked(csv_file, usecols=["id", "name"], chunksize=2))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        expected = df[["id", "name"]]
        pd.testing.assert_frame_equal(combined, expected)

    def test_nrows_limit(self, tmp_path):
        """nrows parameter limits total rows read across all chunks"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"x": range(100)})
        df.to_csv(csv_file, index=False)

        chunks = list(ar.read_csv_chunked(csv_file, chunksize=10, nrows=25))

        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 25

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        expected = df.head(25)
        pd.testing.assert_frame_equal(combined, expected)

    def test_skip_rows(self, tmp_path):
        """skip_rows parameter skips data rows after the header"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"id": range(1, 11), "value": range(10, 20)})
        df.to_csv(csv_file, index=False)

        chunks = list(ar.read_csv_chunked(csv_file, skip_rows=3, chunksize=5))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        expected = df.iloc[3:].reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, expected)

    def test_pipeline_on_chunks(self, tmp_path):
        """Apply cleaning pipeline to each chunk independently"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["  alice  ", "  bob  ", "  charlie  ", "  david  "],
            "age": [25, 30, 35, 40]
        })
        df.to_csv(csv_file, index=False)

        cleaned_chunks = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            clean = ar.pipeline(chunk, [("strip_whitespace",)])
            cleaned_chunks.append(ar.to_pandas(clean))

        combined = pd.concat(cleaned_chunks, ignore_index=True)

        assert combined["name"].iloc[0] == "alice"
        assert combined["name"].iloc[2] == "charlie"

    def test_validation_on_chunks(self, tmp_path):
        """Validate schema on each chunk and aggregate validation results"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": ["1", "2", "not_a_number", "4"],
            "email": ["alice@example.com", "bob@example.com", "invalid_email", "david@example.com"]
        })
        df.to_csv(csv_file, index=False)

        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email()
        })

        total_issues = 0
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            result = ar.validate(chunk, schema)
            total_issues += result.issue_count

        # Both invalid id and invalid email should produce issues
        assert total_issues > 0

    def test_profile_on_chunks(self, tmp_path):
        """Profile each chunk to generate data quality reports"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, None, 4, 5],
            "status": ["active", "inactive", "active", None, "active"]
        })
        df.to_csv(csv_file, index=False)

        chunk_reports = []
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            report = ar.profile(chunk)
            chunk_reports.append(report)

        # Verify we got profiles for each chunk
        assert len(chunk_reports) == 3
        for report in chunk_reports:
            assert "id" in report.columns
            assert "status" in report.columns

    def test_permissive_mode_missing_fields(self, tmp_path):
        """Permissive mode fills missing trailing fields with nulls"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("id,name,age\n")
            f.write("1,Alice,25\n")
            f.write("2,Bob\n")
            f.write("3,Charlie,30\n")

        chunks = list(ar.read_csv_chunked(csv_file, mode="permissive", chunksize=2))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        
        # All rows present with correct shape
        assert combined.shape == (3, 3)
        # Missing field filled with null
        assert pd.isna(combined.iloc[1]["age"])

    def test_on_bad_lines_skip(self, tmp_path):
        """on_bad_lines='skip' drops malformed rows silently"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("id,name\n")
            f.write("1,Alice\n")
            f.write("2,Bob,extra_field\n")
            f.write("3,Charlie\n")

        chunks = list(ar.read_csv_chunked(
            csv_file,
            mode="permissive",
            on_bad_lines="skip",
            chunksize=2
        ))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        
        # Only 2 rows (extra_field row dropped)
        assert combined.shape[0] == 2
        assert list(combined["id"]) == [1, 3]

    def test_null_values_custom(self, tmp_path):
        """Custom null_values parameter treats specified strings as null"""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "status": ["active", "MISSING", "unknown", "active"]
        })
        df.to_csv(csv_file, index=False)

        chunks = list(ar.read_csv_chunked(
            csv_file,
            null_values=["MISSING", "unknown"],
            chunksize=2
        ))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        
        # Custom null values should become NaN
        assert pd.isna(combined.iloc[1]["status"])
        assert pd.isna(combined.iloc[2]["status"])
        assert combined.iloc[0]["status"] == "active"

    def test_chunked_clean_and_persist(self, tmp_path):
        """Read chunks, clean each, and persist results to new CSV"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        df_input = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": ["  alice  ", "  bob  ", "  bob  ", "  charlie  "]
        })
        df_input.to_csv(input_file, index=False)

        # Read, clean, and collect chunks
        all_chunks = []
        for chunk in ar.read_csv_chunked(input_file, chunksize=2):
            clean = ar.pipeline(chunk, [
                ("strip_whitespace",),
                ("drop_duplicates",)
            ])
            all_chunks.append(ar.to_pandas(clean))

        df_output = pd.concat(all_chunks, ignore_index=True)
        df_output.to_csv(output_file, index=False)

        # Verify output file
        df_read_back = pd.read_csv(output_file)
        assert df_read_back.shape[0] == 3  # one duplicate removed
        assert all(df_read_back["name"].str.strip() == df_read_back["name"])

    def test_decimal_separator(self, tmp_path):
        """Handle non-standard decimal separators in numeric columns"""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("id,price\n")
            f.write("1,19,99\n")
            f.write("2,29,95\n")

        chunks = list(ar.read_csv_chunked(
            csv_file,
            decimal_separator=",",
            chunksize=1
        ))

        combined = pd.concat([ar.to_pandas(chunk) for chunk in chunks], ignore_index=True)
        
        # Decimal separator correctly parsed
        assert float(combined.iloc[0]["price"]) == 19.99
        assert float(combined.iloc[1]["price"]) == 29.95
