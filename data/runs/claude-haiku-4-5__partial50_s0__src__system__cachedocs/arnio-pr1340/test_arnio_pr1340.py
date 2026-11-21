"""
End-to-end pytest tests for arnio components:
- arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out
- arnio.io.read_csv
- arnio.io.read_csv_chunked
"""

import pytest
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path
import arnio as ar


class TestArnioCleaner:
    """Test ArnioCleaner.get_feature_names_out in sklearn integration."""

    def test_get_feature_names_out_returns_fitted_names(self, tmp_path):
        """Test get_feature_names_out returns the feature names from fit."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,customer,revenue\n1,alice,100\n2,bob,200")

        # Load, fit cleaner, then check feature names
        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        cleaner = ar.ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)

        output = cleaner.get_feature_names_out()
        assert isinstance(output, np.ndarray)
        assert list(output) == ["id", "customer", "revenue"]

    def test_get_feature_names_out_with_matching_input_features(self, tmp_path):
        """Test get_feature_names_out accepts matching input_features."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,alice\n2,bob")

        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        cleaner = ar.ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)

        # Pass matching features; should return them unchanged
        result = cleaner.get_feature_names_out(["id", "name"])
        assert list(result) == ["id", "name"]

    def test_get_feature_names_out_raises_on_wrong_length(self, tmp_path):
        """Test get_feature_names_out raises ValueError on wrong feature count."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,age\n1,alice,25")

        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        cleaner = ar.ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)

        # Pass wrong number of features
        with pytest.raises(ValueError, match="should have length"):
            cleaner.get_feature_names_out(["id", "name"])

    def test_get_feature_names_out_raises_on_mismatched_names(self, tmp_path):
        """Test get_feature_names_out raises ValueError on mismatched feature names."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,age\n1,alice,25")

        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        cleaner = ar.ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)

        # Pass mismatched names
        with pytest.raises(ValueError, match="must match columns"):
            cleaner.get_feature_names_out(["id", "name", "salary"])

    def test_get_feature_names_out_raises_on_string_input(self, tmp_path):
        """Test get_feature_names_out raises TypeError if input_features is a string."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,alice")

        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        cleaner = ar.ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)

        # Pass string instead of sequence
        with pytest.raises(TypeError, match="must be a sequence"):
            cleaner.get_feature_names_out("id,name")


class TestReadCsv:
    """Test read_csv end-to-end workflows."""

    def test_read_csv_basic_workflow(self, tmp_path):
        """Test basic read_csv: load, verify shape, convert to pandas."""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text(
            "order_id,customer,amount\n"
            "1,alice,100.50\n"
            "2,bob,200.75\n"
            "3,charlie,150.25"
        )

        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        assert frame.shape == (3, 3)
        assert frame.columns == ["order_id", "customer", "amount"]
        assert list(df["customer"]) == ["alice", "bob", "charlie"]

    def test_read_csv_tsv_auto_detection(self, tmp_path):
        """Test read_csv auto-detects TSV delimiter from .tsv extension."""
        tsv_path = tmp_path / "data.tsv"
        tsv_path.write_text("id\tname\tvalue\n1\talice\t100\n2\tbob\t200")

        frame = ar.read_csv(str(tsv_path))

        assert frame.shape == (2, 3)
        assert frame.columns == ["id", "name", "value"]

    def test_read_csv_explicit_delimiter_override(self, tmp_path):
        """Test read_csv respects explicit delimiter over file extension."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id;name;value\n1;alice;100\n2;bob;200")

        # Override to semicolon even though extension is .csv
        frame = ar.read_csv(str(csv_path), delimiter=";")

        assert frame.shape == (2, 3)
        assert list(ar.to_pandas(frame)["name"]) == ["alice", "bob"]

    def test_read_csv_usecols_filtering(self, tmp_path):
        """Test read_csv with usecols selects only specified columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,age,city,salary\n1,alice,25,paris,5000\n2,bob,30,london,6000")

        frame = ar.read_csv(str(csv_path), usecols=["name", "city"])
        df = ar.to_pandas(frame)

        assert frame.shape == (2, 2)
        assert frame.columns == ["name", "city"]
        assert list(df["name"]) == ["alice", "bob"]

    def test_read_csv_nrows_limit(self, tmp_path):
        """Test read_csv with nrows parameter limits rows read."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name\n"
            "1,alice\n"
            "2,bob\n"
            "3,charlie\n"
            "4,diana\n"
            "5,eve"
        )

        frame = ar.read_csv(str(csv_path), nrows=2)

        assert frame.shape == (2, 2)

    def test_read_csv_custom_null_values(self, tmp_path):
        """Test read_csv with custom null_values treats specified strings as null."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,alice,95\n2,MISSING,88\n3,bob,UNKNOWN")

        frame = ar.read_csv(str(csv_path), null_values=["MISSING", "UNKNOWN"])
        df = ar.to_pandas(frame)

        assert pd.isna(df.loc[1, "name"])
        assert pd.isna(df.loc[2, "score"])

    def test_read_csv_decimal_separator(self, tmp_path):
        """Test read_csv with European decimal separator (comma)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,price\n1,19,99\n2,29,95")

        frame = ar.read_csv(str(csv_path), decimal_separator=",")
        df = ar.to_pandas(frame)

        # Prices should parse as floats with decimal point normalized
        assert isinstance(df.loc[0, "price"], (float, np.floating))

    def test_read_csv_file_not_found_error(self, tmp_path):
        """Test read_csv raises CsvReadError when file does not exist."""
        with pytest.raises(ar.CsvReadError, match="Cannot open"):
            ar.read_csv(str(tmp_path / "nonexistent.csv"))

    def test_read_csv_with_pipeline_cleaning(self, tmp_path):
        """Test full workflow: read CSV → pipeline → verify output."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "id,  customer  ,active\n"
            "1, alice ,yes\n"
            "2, bob ,no\n"
            "1, alice ,yes"
        )

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
        )
        df = ar.to_pandas(clean)

        assert clean.shape == (2, 3)
        assert df.loc[0, "customer"] == "alice"


class TestReadCsvChunked:
    """Test read_csv_chunked end-to-end workflows."""

    def test_read_csv_chunked_basic(self, tmp_path):
        """Test read_csv_chunked yields multiple chunks of correct sizes."""
        csv_path = tmp_path / "data.csv"
        lines = ["id,name,value\n"]
        for i in range(25):
            lines.append(f"{i},user{i},{i * 10}\n")
        csv_path.write_text("".join(lines))

        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=10))

        # Should yield 3 chunks: 10, 10, 5
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5
        
        # All chunks should have same columns
        for chunk in chunks:
            assert chunk.columns == ["id", "name", "value"]

    def test_read_csv_chunked_accumulate_data(self, tmp_path):
        """Test accumulating pandas DataFrames from all chunks."""
        csv_path = tmp_path / "data.csv"
        lines = ["id,revenue\n"]
        for i in range(15):
            lines.append(f"{i},{i * 100}\n")
        csv_path.write_text("".join(lines))

        all_dfs = []
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=5):
            df = ar.to_pandas(chunk)
            all_dfs.append(df)

        result = pd.concat(all_dfs, ignore_index=True)
        assert len(result) == 15
        assert list(result.columns) == ["id", "revenue"]

    def test_read_csv_chunked_with_nrows(self, tmp_path):
        """Test read_csv_chunked respects nrows limit across chunks."""
        csv_path = tmp_path / "data.csv"
        lines = ["id,name\n"]
        for i in range(30):
            lines.append(f"{i},user{i}\n")
        csv_path.write_text("".join(lines))

        total_rows = 0
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=10, nrows=25):
            total_rows += chunk.shape[0]

        assert total_rows == 25

    def test_read_csv_chunked_per_chunk_pipeline(self, tmp_path):
        """Test applying cleaning pipeline to each chunk independently."""
        csv_path = tmp_path / "data.csv"
        lines = ["id,  name  \n"]
        for i in range(12):
            lines.append(f"{i},  user{i}  \n")
        csv_path.write_text("".join(lines))

        all_cleaned_names = []
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=5):
            clean = ar.pipeline(chunk, [("strip_whitespace",)])
            df = ar.to_pandas(clean)
            all_cleaned_names.extend(df["name"].tolist())

        # Verify whitespace was stripped from all names
        assert all(not name.startswith(" ") and not name.endswith(" ") for name in all_cleaned_names)
        assert len(all_cleaned_names) == 12

    def test_read_csv_chunked_per_chunk_validation(self, tmp_path):
        """Test validating each chunk against schema and aggregating issues."""
        csv_path = tmp_path / "emails.csv"
        lines = ["id,email\n"]
        lines.append("1,alice@example.com\n")
        lines.append("2,bob@example.com\n")
        lines.append("3,not_an_email\n")
        for i in range(4, 12):
            lines.append(f"{i},user{i}@example.com\n")
        csv_path.write_text("".join(lines))

        schema = ar.Schema({"id": ar.Int64(), "email": ar.Email()})

        all_issues = []
        for chunk in ar.read_csv_chunked(str(csv_path), chunksize=5):
            result = ar.validate(chunk, schema)
            all_issues.extend(result.issues)

        # Should have found invalid email
        assert len(all_issues) > 0
        assert any(issue.column == "email" for issue in all_issues)

    def test_read_csv_chunked_tsv_format(self, tmp_path):
        """Test read_csv_chunked with TSV format."""
        tsv_path = tmp_path / "data.tsv"
        lines = ["id\tname\tvalue\n"]
        for i in range(12):
            lines.append(f"{i}\titem{i}\t{i * 100}\n")
        tsv_path.write_text("".join(lines))

        chunks = list(ar.read_csv_chunked(str(tsv_path), chunksize=5))

        assert len(chunks) == 3
        total = sum(c.shape[0] for c in chunks)
        assert total == 12
        assert chunks[0].columns == ["id", "name", "value"]


class TestIntegrationWorkflows:
    """Test cross-cutting end-to-end workflows combining multiple operations."""

    def test_read_clean_write_roundtrip(self, tmp_path):
        """Test complete roundtrip: read → clean → write → verify."""
        # Create messy input
        input_path = tmp_path / "messy.csv"
        input_path.write_text(
            "id,  customer  ,score\n"
            "1, alice ,95\n"
            "2, bob ,87\n"
            "1, alice ,95"
        )

        # Read and clean
        frame = ar.read_csv(str(input_path))
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
        )

        # Write to output
        output_path = tmp_path / "clean.csv"
        ar.write_csv(clean, str(output_path))

        # Read back and verify
        result = ar.read_csv(str(output_path))
        df = ar.to_pandas(result)

        assert result.shape == (2, 3)
        assert list(df["customer"]) == ["alice", "bob"]

    def test_pandas_accessor_workflow(self, tmp_path):
        """Test pandas .arnio accessor integration with CSV data."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,  name  ,score\n"
            "1, alice ,95\n"
            "2, bob ,87\n"
            "2, bob ,87"
        )

        # Read with arnio
        frame = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame)

        # Use accessor for cleaning
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True,
        )

        assert len(clean_df) == 2
        assert clean_df.loc[0, "name"] == "alice"

    def test_profile_suggest_clean_workflow(self, tmp_path):
        """Test profiling, suggesting cleanup, and applying suggestions."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "customer,  revenue  \n"
            "alice, 1000 \n"
            "bob, 2000 \n"
            "alice, 1000 "
        )

        frame = ar.read_csv(str(csv_path))

        # Profile the data
        report = ar.profile(frame)
        assert report.row_count == 3

        # Get cleanup suggestions
        suggestions = ar.suggest_cleaning(frame)
        assert len(suggestions) > 0

        # Apply suggestions to clean the data
        clean = ar.pipeline(frame, suggestions)
        df = ar.to_pandas(clean)

        # Result should have fewer rows (deduped)
        assert len(df) < 3
