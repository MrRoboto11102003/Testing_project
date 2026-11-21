import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestFromRecordsBasicWorkflows:
    """Test basic from_records functionality with realistic data workflows."""

    def test_from_dict_records_auto_columns(self):
        """Create ArFrame from dict records with auto-inferred columns."""
        records = [
            {"name": "alice", "age": 30},
            {"name": "bob", "age": 25},
        ]
        frame = ar.from_records(records)

        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}

        df = ar.to_pandas(frame)
        assert len(df) == 2
        assert df["name"].tolist() == ["alice", "bob"]
        assert df["age"].tolist() == [30, 25]

    def test_from_list_records_with_columns(self):
        """Create ArFrame from list records with explicit column names."""
        records = [
            ["alice", 30, "paris"],
            ["bob", 25, "london"],
            ["carol", 28, "berlin"],
        ]
        columns = ["name", "age", "city"]
        frame = ar.from_records(records, columns=columns)

        assert frame.shape == (3, 3)
        assert frame.columns == columns

        df = ar.to_pandas(frame)
        assert df["city"].tolist() == ["paris", "london", "berlin"]

    def test_from_tuple_records_with_columns(self):
        """Create ArFrame from tuple records."""
        records = [
            ("alice", 30),
            ("bob", 25),
        ]
        columns = ["name", "age"]
        frame = ar.from_records(records, columns=columns)

        assert frame.shape == (2, 2)
        df = ar.to_pandas(frame)
        assert df["name"].tolist() == ["alice", "bob"]


class TestFromRecordsErrorHandling:
    """Test error cases and validation."""

    def test_empty_records_raises_valueerror(self):
        """Empty records list should raise ValueError."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.from_records([])

    def test_non_list_input_raises_typeerror(self):
        """Non-list input should raise TypeError."""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.from_records({"name": "alice"})

    def test_mixed_record_types_raises_typeerror(self):
        """Mixed dict and list records should raise TypeError."""
        records = [
            {"name": "alice", "age": 30},
            ["bob", 25],
        ]
        with pytest.raises(TypeError, match="all records must be dicts"):
            ar.from_records(records)

    def test_nested_list_in_values_raises_typeerror(self):
        """Nested list in record values should raise TypeError."""
        records = [
            {"name": "alice", "skills": ["python", "rust"]},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)

    def test_nested_dict_in_values_raises_typeerror(self):
        """Nested dict in record values should raise TypeError."""
        records = [
            {"name": "alice", "metadata": {"role": "engineer"}},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)

    def test_missing_columns_for_list_records_raises_valueerror(self):
        """List records without columns parameter should raise ValueError."""
        records = [
            ["alice", 30],
            ["bob", 25],
        ]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.from_records(records)

    def test_row_length_mismatch_raises_valueerror(self):
        """Row with mismatched length should raise ValueError."""
        records = [
            ["alice", 30],
            ["bob", 25, "extra"],
        ]
        columns = ["name", "age"]
        with pytest.raises(ValueError, match="has 3 value\\(s\\) but 2 column\\(s\\)"):
            ar.from_records(records, columns=columns)

    def test_empty_columns_with_dict_records_raises_valueerror(self):
        """Empty columns list with dict records should raise ValueError."""
        records = [{"name": "alice", "age": 30}]
        with pytest.raises(ValueError, match="columns must not be empty"):
            ar.from_records(records, columns=[])

    def test_non_string_column_names_raises_typeerror(self):
        """Non-string column names should raise TypeError."""
        records = [["alice", 30]]
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.from_records(records, columns=["name", 42])


class TestFromRecordsWithMissingValues:
    """Test from_records with null and missing values."""

    def test_dict_records_with_missing_keys(self):
        """Dict records with missing keys should fill with None/null."""
        records = [
            {"name": "alice", "age": 30, "city": "paris"},
            {"name": "bob", "age": 25},
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)

        assert df.loc[1, "city"] is None or pd.isna(df.loc[1, "city"])

    def test_list_records_preserve_none_values(self):
        """List records with None values should be preserved."""
        records = [
            ["alice", 30, "paris"],
            ["bob", None, "london"],
        ]
        columns = ["name", "age", "city"]
        frame = ar.from_records(records, columns=columns)
        df = ar.to_pandas(frame)

        assert pd.isna(df.loc[1, "age"])


class TestFromRecordsWithPipeline:
    """Test from_records integrated with cleaning pipelines."""

    def test_from_records_strip_whitespace_pipeline(self):
        """Create frame, then apply whitespace stripping."""
        records = [
            {"name": " alice ", "city": " paris "},
            {"name": " bob ", "city": " london "},
        ]
        frame = ar.from_records(records)

        clean = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(clean)

        assert df["name"].tolist() == ["alice", "bob"]
        assert df["city"].tolist() == ["paris", "london"]

    def test_from_records_drop_duplicates_pipeline(self):
        """Create frame with duplicates, apply deduplication."""
        records = [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
            {"id": 1, "value": 100},
        ]
        frame = ar.from_records(records)

        clean = ar.pipeline(frame, [("drop_duplicates",)])
        df = ar.to_pandas(clean)

        assert len(df) == 2

    def test_from_records_multi_step_pipeline(self):
        """Create frame and apply multiple cleaning steps."""
        records = [
            {"name": " ALICE ", "score": 95},
            {"name": " BOB ", "score": 88},
            {"name": " ALICE ", "score": 95},
        ]
        frame = ar.from_records(records)

        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(clean)

        assert len(df) == 2
        assert df["name"].tolist() == ["alice", "bob"]


class TestFromRecordsIOWorkflows:
    """Test from_records with file I/O workflows."""

    def test_from_records_write_read_csv(self, tmp_path):
        """Create frame from records, write to CSV, read back."""
        records = [
            {"id": 1, "email": "alice@example.com", "score": 95},
            {"id": 2, "email": "bob@example.com", "score": 88},
        ]
        frame = ar.from_records(records)

        csv_path = tmp_path / "data.csv"
        ar.write_csv(frame, str(csv_path))

        frame_loaded = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame_loaded)

        assert len(df) == 2
        assert df["email"].tolist() == ["alice@example.com", "bob@example.com"]
        assert csv_path.exists()

    def test_from_records_write_jsonl(self, tmp_path):
        """Create frame from records, write to JSONL."""
        records = [
            {"name": "alice", "age": 30},
            {"name": "bob", "age": 25},
        ]
        frame = ar.from_records(records)

        jsonl_path = tmp_path / "data.jsonl"
        ar.write_jsonl(frame, str(jsonl_path))

        assert jsonl_path.exists()
        frame_loaded = ar.read_jsonl(str(jsonl_path))
        df = ar.to_pandas(frame_loaded)
        assert len(df) == 2


class TestFromRecordsValidation:
    """Test from_records with schema validation."""

    def test_from_records_validate_schema(self):
        """Create frame from records and validate against schema."""
        records = [
            {"id": 1, "email": "alice@example.com", "age": 30},
            {"id": 2, "email": "bob@example.com", "age": 25},
        ]
        frame = ar.from_records(records)

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0),
        })

        result = ar.validate(frame, schema)
        assert result.passed

    def test_from_records_validation_with_failures(self):
        """Create frame that fails validation."""
        records = [
            {"id": 1, "email": "not-an-email", "age": 30},
        ]
        frame = ar.from_records(records)

        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) > 0


class TestFromRecordsProfiling:
    """Test from_records with data profiling."""

    def test_from_records_profile_and_suggest(self):
        """Create frame from records, profile, and get suggestions."""
        records = [
            {"name": " ALICE ", "score": 95},
            {"name": " BOB ", "score": 88},
            {"name": " ALICE ", "score": 95},
        ]
        frame = ar.from_records(records)

        report = ar.profile(frame)
        summary = report.summary()

        assert summary["row_count"] == 3
        assert "duplicate_ratio" in summary

        suggestions = ar.suggest_cleaning(frame)
        assert isinstance(suggestions, list)

    def test_from_records_auto_clean(self):
        """Create frame from records and apply auto-clean."""
        records = [
            {"name": " alice ", "value": 100},
            {"name": " bob ", "value": 200},
            {"name": " alice ", "value": 100},
        ]
        frame = ar.from_records(records)

        clean = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean)

        assert df["name"].tolist() == ["alice", "bob", "alice"]


class TestFromRecordsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_record(self):
        """Single record should work."""
        records = [{"name": "alice", "age": 30}]
        frame = ar.from_records(records)

        assert frame.shape == (1, 2)
        df = ar.to_pandas(frame)
        assert len(df) == 1

    def test_many_columns(self):
        """Many columns should work."""
        columns = [f"col_{i}" for i in range(50)]
        records = [[i] * 50 for i in range(5)]

        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (5, 50)

    def test_special_characters_in_strings(self):
        """String values with special characters should be preserved."""
        records = [
            {"text": "hello,world", "special": "foo\nbar", "quoted": 'foo"bar'},
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)

        assert df["text"][0] == "hello,world"
        assert df["special"][0] == "foo\nbar"
        assert df["quoted"][0] == 'foo"bar'

    def test_numeric_types_preserved(self):
        """Numeric types should be inferred correctly."""
        records = [
            {"int_col": 42, "float_col": 3.14},
            {"int_col": 100, "float_col": 2.71},
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)

        assert len(df) == 2
        assert df["int_col"].sum() == 142

    def test_from_records_then_select_columns(self):
        """Create frame and use select_columns."""
        records = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
        ]
        frame = ar.from_records(records)

        selected = ar.select_columns(frame, ["a", "c"])
        assert selected.shape == (2, 2)
        assert selected.columns == ["a", "c"]
