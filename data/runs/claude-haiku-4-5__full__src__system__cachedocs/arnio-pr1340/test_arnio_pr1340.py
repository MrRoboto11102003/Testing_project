import pytest
import io
import os
import sys
import tempfile
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestReadCsvBasicWorkflow:
    """End-to-end tests for ar.read_csv basic workflows."""

    def test_read_csv_local_file_clean_and_convert(self, tmp_path):
        """Test complete workflow: load CSV, clean, convert to pandas, verify."""
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("product,revenue,date\n Alice Widget , 1500.50 , 2024-01-01\n  Bob Gadget ,2000.00, 2024-01-02\n")
        
        # Load and clean
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"subset": ["product"], "case_type": "lower"}),
        ])
        
        # Convert to pandas
        df = ar.to_pandas(clean)
        
        # Verify
        assert df.shape == (2, 3)
        assert df["product"].tolist() == ["alice widget", "bob gadget"]
        assert float(df["revenue"].iloc[0]) == pytest.approx(1500.5)

    def test_read_csv_tsv_auto_detection_and_pipeline(self, tmp_path):
        """Test TSV auto-detection with pipeline and write output."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id\tname\tscore\n1\tAlice\t95.5\n2\tBob\t88.0\n3\tCharlie\t92.3\n")
        
        # Load (tab should auto-detect)
        frame = ar.read_csv(str(tsv_file))
        
        # Apply pipeline
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "score", "op": ">=", "value": "90"}),
            ("drop_nulls",),
        ])
        
        # Write to new CSV
        output_csv = tmp_path / "output.csv"
        ar.write_csv(clean, str(output_csv))
        
        # Verify output file exists and has correct content
        assert output_csv.exists()
        output_text = output_csv.read_text()
        assert "Alice" in output_text
        assert "Charlie" in output_text
        assert "Bob" not in output_text  # Score 88 < 90

    def test_read_csv_delimiter_override_and_dtype_mapping(self, tmp_path):
        """Test explicit delimiter override and dtype mapping."""
        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("id;name;value\n001;Product A;99.99\n002;Product B;149.50\n")
        
        # Override with semicolon, map dtypes
        frame = ar.read_csv(
            str(csv_file),
            delimiter=";",
            dtype={"id": "string", "value": "float64"}
        )
        
        assert frame.dtypes["id"] == "string"
        assert frame.dtypes["value"] == "float64"
        
        df = ar.to_pandas(frame)
        assert df["id"].tolist() == ["001", "002"]  # Preserved as strings

    def test_read_csv_usecols_subset_workflow(self, tmp_path):
        """Test selective column loading with cleaning."""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text("id,first_name,last_name,email,phone\n1,Alice,Smith,alice@test.com,555-0001\n2,Bob,Jones,bob@test.com,555-0002\n")
        
        # Load only specific columns
        frame = ar.read_csv(str(csv_file), usecols=["first_name", "email"])
        
        assert frame.shape == (2, 2)
        assert list(frame.columns) == ["first_name", "email"]

    def test_read_csv_nrows_limit_workflow(self, tmp_path):
        """Test row limit and profile the result."""
        csv_file = tmp_path / "large.csv"
        lines = ["id,value\n"]
        for i in range(100):
            lines.append(f"{i},{i*10}\n")
        csv_file.write_text("".join(lines))
        
        # Load only first 10 rows
        frame = ar.read_csv(str(csv_file), nrows=10)
        
        assert frame.shape[0] == 10
        df = ar.to_pandas(frame)
        assert df["id"].max() == 9

    def test_read_csv_null_values_and_fill_pipeline(self, tmp_path):
        """Test custom null sentinels and fill nulls in pipeline."""
        csv_file = tmp_path / "incomplete.csv"
        csv_file.write_text("id,status,amount\n1,active,100.0\n2,N/A,200.0\n3,MISSING,300.0\n4,inactive,\n")
        
        frame = ar.read_csv(
            str(csv_file),
            null_values=["N/A", "MISSING"]
        )
        
        # Fill nulls
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": "unknown", "subset": ["status"]}),
            ("fill_nulls", {"value": 0.0, "subset": ["amount"]}),
        ])
        
        df = ar.to_pandas(clean)
        assert df["status"].isna().sum() == 0
        assert df["amount"].isna().sum() == 0

    def test_read_csv_decimal_separator_european_workflow(self, tmp_path):
        """Test European decimal format with thousands separator."""
        csv_file = tmp_path / "prices_eu.csv"
        csv_file.write_text('product,price\nWidget,"1.234,56"\nGadget,"2.999,99"\n')
        
        frame = ar.read_csv(
            str(csv_file),
            decimal_separator=",",
            thousands_separator="."
        )
        
        df = ar.to_pandas(frame)
        assert float(df.loc[0, "price"]) == pytest.approx(1234.56)
        assert float(df.loc[1, "price"]) == pytest.approx(2999.99)

    def test_read_csv_stringio_workflow(self, tmp_path):
        """Test reading from file-like object and profile."""
        csv_text = "id,name,score\n1,Alice,95\n2,Bob,88\n3,Charlie,92\n"
        csv_io = io.StringIO(csv_text)
        
        frame = ar.read_csv(csv_io)
        
        # Profile the frame
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert "id" in report.columns
        
        df = ar.to_pandas(frame)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_read_csv_pathlib_and_pandas_accessor(self, tmp_path):
        """Test reading with pathlib.Path and using pandas accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,city\n Alice ,25, New York \n  Bob  ,30,  London  \n")
        
        # Use pathlib.Path
        frame = ar.read_csv(csv_file)
        df = ar.to_pandas(frame)
        
        # Use pandas accessor for cleaning
        clean_df = df.arnio.clean(strip_whitespace=True)
        
        assert clean_df["name"].tolist() == ["Alice", "Bob"]
        assert clean_df["city"].tolist() == ["New York", "London"]

    def test_read_csv_mode_permissive_bad_rows_skip(self, tmp_path):
        """Test permissive mode with bad rows handling."""
        csv_file = tmp_path / "incomplete_rows.csv"
        csv_file.write_text("id,name,age\n1,Alice,25\n2,Bob\n3,Charlie,35\n")
        
        frame = ar.read_csv(
            str(csv_file),
            mode="permissive",
            on_bad_lines="skip"
        )
        
        df = ar.to_pandas(frame)
        # Row with missing field should be skipped
        assert df.shape[0] >= 1


class TestReadCsvChunked:
    """End-to-end tests for ar.read_csv_chunked."""

    def test_read_csv_chunked_basic_aggregation(self, tmp_path):
        """Test basic chunked reading with aggregation."""
        csv_file = tmp_path / "large.csv"
        lines = ["id,value\n"]
        for i in range(35):
            lines.append(f"{i},{i*100}\n")
        csv_file.write_text("".join(lines))
        
        # Read in chunks and aggregate
        all_values = []
        chunk_count = 0
        for chunk in ar.read_csv_chunked(str(csv_file), chunksize=10):
            chunk_count += 1
            df = ar.to_pandas(chunk)
            all_values.extend(df["value"].tolist())
        
        # Verify chunking: 35 rows with chunksize=10 → 4 chunks (10, 10, 10, 5)
        assert chunk_count == 4
        assert len(all_values) == 35

    def test_read_csv_chunked_pipeline_on_each_chunk(self, tmp_path):
        """Test applying pipeline to each chunk separately."""
        csv_file = tmp_path / "data.csv"
        lines = ["name,score\n"]
        for i in range(20):
            lines.append(f"  Person{i}  ,{80 + i}\n")
        csv_file.write_text("".join(lines))
        
        cleaned_data = []
        for chunk in ar.read_csv_chunked(str(csv_file), chunksize=7):
            # Clean each chunk
            clean = ar.pipeline(chunk, [
                ("strip_whitespace",),
                ("filter_rows", {"column": "score", "op": ">=", "value": "85"}),
            ])
            df = ar.to_pandas(clean)
            cleaned_data.append(df)
        
        result = pd.concat(cleaned_data, ignore_index=True)
        assert len(result) > 0
        assert all(result["score"] >= 85)
        # All names should be stripped
        assert all(result["name"].str.strip() == result["name"])

    def test_read_csv_chunked_nrows_global_limit(self, tmp_path):
        """Test that nrows limit applies across all chunks."""
        csv_file = tmp_path / "data.csv"
        lines = ["id\n"]
        for i in range(100):
            lines.append(f"{i}\n")
        csv_file.write_text("".join(lines))
        
        total_rows = 0
        for chunk in ar.read_csv_chunked(
            str(csv_file),
            chunksize=15,
            nrows=40
        ):
            total_rows += chunk.shape[0]
        
        assert total_rows == 40

    def test_read_csv_chunked_tsv_with_dtype(self, tmp_path):
        """Test TSV auto-detection with dtype mapping in chunked mode."""
        tsv_file = tmp_path / "data.tsv"
        lines = ["id\tname\tvalue\n"]
        for i in range(25):
            lines.append(f"{i:03d}\tItem{i}\t{i*1.5}\n")
        tsv_file.write_text("".join(lines))
        
        chunk_dtypes = []
        for chunk in ar.read_csv_chunked(
            str(tsv_file),
            chunksize=10,
            dtype={"id": "string", "value": "float64"}
        ):
            chunk_dtypes.append(chunk.dtypes)
            assert chunk.dtypes["id"] == "string"
            assert chunk.dtypes["value"] == "float64"
        
        assert len(chunk_dtypes) >= 2


class TestArnioCleaner:
    """End-to-end tests for ArnioCleaner.get_feature_names_out."""

    def test_get_feature_names_out_basic_workflow(self, tmp_path):
        """Test ArnioCleaner fit and get_feature_names_out."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("scikit-learn integration not available")
        
        csv_file = tmp_path / "features.csv"
        csv_file.write_text("age,income,score\n25,50000,85\n30,60000,90\n35,75000,92\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        cleaner = ArnioCleaner(steps=[("normalize_case",)])
        cleaner.fit(df)
        
        # Get feature names
        names = cleaner.get_feature_names_out()
        
        assert list(names) == ["age", "income", "score"]

    def test_get_feature_names_out_with_matching_input_features(self, tmp_path):
        """Test get_feature_names_out with matching input features."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("scikit-learn integration not available")
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)
        
        # Pass matching input features
        names = cleaner.get_feature_names_out(
            input_features=["col1", "col2", "col3"]
        )
        
        assert list(names) == ["col1", "col2", "col3"]

    def test_get_feature_names_out_error_on_mismatched_features(self, tmp_path):
        """Test get_feature_names_out raises on mismatched feature names."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("scikit-learn integration not available")
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,city\nAlice,25,NYC\nBob,30,LA\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)
        
        # Try to pass mismatched feature names
        with pytest.raises(ValueError, match="must match columns"):
            cleaner.get_feature_names_out(
                input_features=["name", "age", "country"]
            )

    def test_get_feature_names_out_error_on_wrong_length(self, tmp_path):
        """Test get_feature_names_out raises on wrong number of features."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("scikit-learn integration not available")
        
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(df)
        
        # Try with wrong number of features
        with pytest.raises(ValueError, match="length"):
            cleaner.get_feature_names_out(input_features=["a", "b"])

    def test_get_feature_names_out_error_if_not_fitted(self):
        """Test get_feature_names_out raises if cleaner not fitted."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
        except ImportError:
            pytest.skip("scikit-learn integration not available")
        
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        
        # Should raise because not fitted
        with pytest.raises(Exception):  # NotFittedError or sklearn exception
            cleaner.get_feature_names_out()
