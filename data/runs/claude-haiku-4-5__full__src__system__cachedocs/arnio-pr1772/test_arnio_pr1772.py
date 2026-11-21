import pytest
import pandas as pd
import arnio as ar
import tempfile
from pathlib import Path


class TestFromRecordsBasicWorkflows:
    """Core from_records workflows with file I/O and conversions."""

    def test_from_records_dicts_to_csv_round_trip(self, tmp_path):
        """Create ArFrame from dicts, write to CSV, read back and verify."""
        records = [
            {"id": 1, "name": "Alice", "age": 30, "city": "Paris"},
            {"id": 2, "name": "Bob", "age": 25, "city": "London"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Tokyo"},
        ]
        
        # Create frame from records
        frame = ar.from_records(records)
        assert frame.shape == (3, 4)
        assert set(frame.columns) == {"id", "name", "age", "city"}
        
        # Write to CSV
        csv_path = tmp_path / "data.csv"
        ar.write_csv(frame, str(csv_path))
        assert csv_path.exists()
        
        # Read back and verify
        frame_read = ar.read_csv(str(csv_path))
        assert frame_read.shape == frame.shape
        
        df = ar.to_pandas(frame_read)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_from_records_tuples_with_columns_pipeline(self):
        """Create ArFrame from tuples, apply cleaning pipeline, convert to pandas."""
        records = [
            (1, "  alice  ", 30),
            (2, "  bob  ", 25),
            (1, "  alice  ", 30),  # duplicate
        ]
        columns = ["id", "name", "age"]
        
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (3, 3)
        
        # Apply cleaning pipeline
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(clean)
        assert len(df) == 2  # Duplicate removed
        assert df["name"].tolist() == ["alice", "bob"]

    def test_from_records_lists_with_profiling(self):
        """Create ArFrame from lists, profile data quality, verify report."""
        records = [
            [1, "Alice", 85.5],
            [2, "Bob", None],
            [3, "Charlie", 92.0],
        ]
        columns = ["id", "name", "score"]
        
        frame = ar.from_records(records, columns=columns)
        
        # Profile the data
        report = ar.profile(frame)
        assert report.row_count == 3
        assert report.column_count == 3
        assert report.columns["score"].null_count == 1
        assert report.columns["score"].mean == 88.75  # (85.5 + 92.0) / 2
        
        # Verify markdown output works
        markdown = report.to_markdown()
        assert "score" in markdown
        assert "null" in markdown.lower()


class TestFromRecordsWithValidation:
    """Schema validation workflows with from_records."""

    def test_from_records_valid_schema(self):
        """Create frame from records and validate against schema."""
        records = [
            {"user_id": 101, "email": "alice@example.com", "age": 30},
            {"user_id": 102, "email": "bob@example.com", "age": 25},
        ]
        
        frame = ar.from_records(records)
        
        # Define schema
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, min=0, max=150),
        })
        
        # Validate
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0

    def test_from_records_invalid_schema(self):
        """Create frame with data that fails schema validation."""
        records = [
            {"user_id": 1, "email": "alice@example.com", "age": 30},
            {"user_id": 2, "email": "not-an-email", "age": 200},  # Invalid email & age
            {"user_id": 2, "email": "charlie@example.com", "age": 25},  # Duplicate ID
        ]
        
        frame = ar.from_records(records)
        
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, max=150),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert result.issue_count >= 3  # Email format, age max, uniqueness


class TestFromRecordsCleaningWorkflows:
    """Data cleaning workflows starting from from_records."""

    def test_from_records_whitespace_cleanup_to_file(self, tmp_path):
        """Create messy frame from records, clean, save to CSV."""
        records = [
            {"id": 1, "customer": "  Alice Smith  ", "city": " Paris ", "status": "ACTIVE"},
            {"id": 2, "customer": "  Bob Jones  ", "city": "London", "status": "INACTIVE"},
            {"id": 1, "customer": "  Alice Smith  ", "city": " Paris ", "status": "ACTIVE"},
        ]
        
        frame = ar.from_records(records)
        
        # Clean the data
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower", "subset": ["status"]}),
            ("drop_duplicates",),
        ])
        
        # Write cleaned data
        output_path = tmp_path / "clean_customers.csv"
        ar.write_csv(clean, str(output_path))
        
        # Verify output
        frame_verify = ar.read_csv(str(output_path))
        df = ar.to_pandas(frame_verify)
        
        assert len(df) == 2  # Duplicates removed
        assert df["customer"].tolist() == ["Alice Smith", "Bob Jones"]
        assert df["status"].tolist() == ["active", "inactive"]

    def test_from_records_fill_nulls_pipeline(self):
        """Create frame with nulls, fill them, verify results."""
        records = [
            {"id": 1, "name": "Alice", "score": 85.0},
            {"id": 2, "name": "Bob", "score": None},
            {"id": 3, "name": None, "score": 92.0},
        ]
        
        frame = ar.from_records(records)
        
        # Fill nulls
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["score"]}),
            ("fill_nulls", {"value": "Unknown", "subset": ["name"]}),
        ])
        
        df = ar.to_pandas(clean)
        assert df["score"].tolist() == [85.0, 0.0, 92.0]
        assert df["name"].tolist() == ["Alice", "Bob", "Unknown"]


class TestFromRecordsPandasIntegration:
    """Test integration with pandas DataFrame and .arnio accessor."""

    def test_from_records_to_pandas_accessor_workflow(self):
        """Create frame, convert to pandas, use .arnio accessor for cleaning."""
        records = [
            {"name": "  Alice  ", "email": "ALICE@EXAMPLE.COM", "age": 30},
            {"name": "  Bob  ", "email": "BOB@EXAMPLE.COM", "age": 25},
            {"name": "  Alice  ", "email": "ALICE@EXAMPLE.COM", "age": 30},
        ]
        
        # Create frame and convert to pandas
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        
        # Use pandas accessor for cleaning
        clean_df = df.arnio.clean(
            strip_whitespace={"subset": ["name"]},
            drop_duplicates=True,
        )
        
        assert len(clean_df) == 2
        assert clean_df["name"].tolist() == ["Alice", "Bob"]
        
        # Also verify profiling via accessor
        report = clean_df.arnio.profile()
        assert report.row_count == 2


class TestFromRecordsErrorHandling:
    """Error handling and validation of from_records inputs."""

    def test_from_records_empty_raises_value_error(self):
        """Empty records list should raise ValueError."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.from_records([])

    def test_from_records_not_list_raises_type_error(self):
        """Non-list records parameter should raise TypeError."""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.from_records({"id": 1, "name": "Alice"})

    def test_from_records_mixed_types_raises_type_error(self):
        """Mixing dicts and other types should raise TypeError."""
        records = [
            {"id": 1, "name": "Alice"},
            [2, "Bob"],  # Mixed types
        ]
        with pytest.raises(TypeError, match="all records must be dicts"):
            ar.from_records(records)

    def test_from_records_nested_values_raises_type_error(self):
        """Nested list or dict values in cells should raise TypeError."""
        records = [
            {"id": 1, "tags": ["python", "data"]},  # Nested list
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)
        
        records = [
            {"id": 1, "metadata": {"type": "user"}},  # Nested dict
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)

    def test_from_records_tuples_without_columns_raises_error(self):
        """Tuples without columns parameter should raise ValueError."""
        records = [(1, "Alice", 30), (2, "Bob", 25)]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.from_records(records)

    def test_from_records_row_column_mismatch_raises_error(self):
        """Row length mismatched with columns should raise ValueError."""
        records = [(1, "Alice"), (2, "Bob", "extra")]
        columns = ["id", "name"]
        with pytest.raises(ValueError, match="has .* value"):
            ar.from_records(records, columns=columns)

    def test_from_records_columns_parameter_validation(self):
        """Columns parameter must be list/tuple of strings."""
        records = [{"id": 1}]
        
        # Pass columns as string instead of list
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.from_records(records, columns="id,name")
        
        # Pass columns with non-string elements
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.from_records([(1, "Alice")], columns=[1, "name"])


class TestFromRecordsEdgeCases:
    """Edge cases and special scenarios."""

    def test_from_records_single_row(self):
        """Single row record should work correctly."""
        records = [{"id": 1, "name": "Alice"}]
        frame = ar.from_records(records)
        
        assert frame.shape == (1, 2)
        df = ar.to_pandas(frame)
        assert len(df) == 1
        assert df["name"].iloc[0] == "Alice"

    def test_from_records_many_columns(self):
        """Record with many columns should work."""
        record = {f"col_{i:03d}": i for i in range(50)}
        frame = ar.from_records([record])
        
        assert frame.shape == (1, 50)
        assert len(frame.columns) == 50

    def test_from_records_with_nulls_and_missing_keys(self):
        """Dicts with None values and missing keys."""
        records = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": None, "age": None},
            {"id": 3, "age": 28},  # Missing 'name'
        ]
        
        frame = ar.from_records(records)
        
        # Check nulls are handled correctly
        report = ar.profile(frame)
        assert report.columns["name"].null_count == 2
        assert report.columns["age"].null_count == 1
        
        df = ar.to_pandas(frame)
        assert pd.isna(df.loc[2, "name"])

    def test_from_records_with_mixed_numeric_types(self):
        """Records with int, float, bool, and string columns."""
        records = [
            {"id": 1, "value": 100.5, "active": True, "label": "A"},
            {"id": 2, "value": 200.0, "active": False, "label": "B"},
        ]
        
        frame = ar.from_records(records)
        
        # Verify dtypes are correct
        dtypes = frame.dtypes
        assert "int64" in str(dtypes["id"]).lower()
        assert "float64" in str(dtypes["value"]).lower()
        assert "bool" in str(dtypes["active"]).lower()
        
        # Validate with schema
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "value": ar.Float64(nullable=False),
            "active": ar.Bool(nullable=False),
            "label": ar.String(nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
