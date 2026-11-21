import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path


class TestDropColumnsBasic:
    """Basic drop_columns functionality tests."""

    def test_drop_single_column_csv_workflow(self, tmp_path):
        """Test dropping a single column from a CSV file."""
        # Create test CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,debug_col\n1,Alice,xyz\n2,Bob,abc\n")

        # Load and process
        frame = ar.read_csv(str(csv_file))
        assert frame.columns == ["id", "name", "debug_col"]
        assert frame.shape == (2, 3)

        # Drop column
        clean = ar.drop_columns(frame, ["debug_col"])
        assert clean.columns == ["id", "name"]
        assert clean.shape == (2, 2)

        # Verify data integrity
        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert list(df["id"]) == [1, 2]
        assert list(df["name"]) == ["Alice", "Bob"]

    def test_drop_multiple_columns(self, tmp_path):
        """Test dropping multiple columns in one operation."""
        csv_file = tmp_path / "multi.csv"
        csv_file.write_text("a,b,c,d,e\n1,2,3,4,5\n6,7,8,9,10\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["b", "d"])

        assert clean.columns == ["a", "c", "e"]
        df = ar.to_pandas(clean)
        assert df.shape == (2, 3)
        assert list(df["a"]) == [1, 6]

    def test_drop_columns_to_csv_roundtrip(self, tmp_path):
        """Test dropping columns and writing result to CSV."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name,revenue,temp_col\n1,Alice,1000,x\n2,Bob,2000,y\n")
        output_csv = tmp_path / "output.csv"

        frame = ar.read_csv(str(input_csv))
        clean = ar.drop_columns(frame, ["temp_col"])
        ar.write_csv(clean, str(output_csv))

        # Verify output file
        assert output_csv.exists()
        output_text = output_csv.read_text()
        assert "temp_col" not in output_text
        assert "id,name,revenue" in output_text or "id , name , revenue" in output_text

        # Verify by reading back
        result_frame = ar.read_csv(str(output_csv))
        assert result_frame.columns == ["id", "name", "revenue"]

    def test_drop_columns_to_pandas_defensive_copy(self, tmp_path):
        """Test converting dropped frame to pandas with defensive copy."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n4,5,6\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["y"])

        df = ar.to_pandas(clean, copy=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "z"]
        assert len(df) == 2
        assert df.iloc[0, 0] == 1


class TestDropColumnsInPipeline:
    """Test drop_columns as a pipeline step."""

    def test_drop_columns_as_pipeline_step(self, tmp_path):
        """Test drop_columns working within a cleaning pipeline."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id, name , email,debug_field\n1, Alice , a@ex.com,xyz\n2, Bob , b@ex.com,abc\n")

        frame = ar.read_csv(str(csv_file))

        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_columns", {"columns": ["debug_field"]}),
        ])

        df = ar.to_pandas(clean)
        assert "debug_field" not in df.columns
        assert set(df.columns) == {"id", "name", "email"}
        assert len(df) == 2

    def test_drop_columns_with_drop_duplicates(self, tmp_path):
        """Test combining drop_columns with deduplication."""
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text("id,name,temp_col\n1,Alice,x\n1,Alice,y\n2,Bob,z\n")

        frame = ar.read_csv(str(csv_file))

        clean = ar.pipeline(frame, [
            ("drop_columns", {"columns": ["temp_col"]}),
            ("drop_duplicates",),
        ])

        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert list(df["id"]) == [1, 2]

    def test_drop_columns_with_profiling(self, tmp_path):
        """Test profiling after dropping columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,debug_xyz\n1,100,unused\n2,95,unused\n3,88,unused\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["debug_xyz"])

        report = ar.profile(clean)
        summary = report.summary()

        assert summary["row_count"] == 3
        assert "id" in summary["columns"]
        assert "score" in summary["columns"]
        assert "debug_xyz" not in summary["columns"]


class TestDropColumnsPandasAccessor:
    """Test drop_columns via pandas .arnio accessor."""

    def test_pandas_accessor_drop_columns_workflow(self, tmp_path):
        """Test dropping columns using pandas .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,temporary_col\n1,Alice,x\n2,Bob,y\n")

        df = pd.read_csv(str(csv_file))

        # Use arnio pipeline through pandas accessor
        clean_df = df.arnio.clean(steps=[
            ("drop_columns", {"columns": ["temporary_col"]}),
        ])

        assert list(clean_df.columns) == ["id", "name"]
        assert len(clean_df) == 2


class TestDropColumnsErrors:
    """Test error handling and validation."""

    def test_error_drop_nonexistent_column(self, tmp_path):
        """Test ValueError when dropping a column that doesn't exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(frame, ["nonexistent"])

    def test_error_string_instead_of_list(self, tmp_path):
        """Test TypeError when columns parameter is a string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError):
            ar.drop_columns(frame, "id")

    def test_error_non_string_in_columns_list(self, tmp_path):
        """Test TypeError when columns sequence contains non-string items."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,10\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError):
            ar.drop_columns(frame, ["id", 123])

    def test_error_drop_all_columns(self, tmp_path):
        """Test ValueError when attempting to drop all columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(frame, ["id", "name"])

    def test_error_mixed_valid_and_invalid_columns(self, tmp_path):
        """Test ValueError with both valid and invalid column names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,10\n")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(frame, ["id", "missing"])


class TestDropColumnsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_drop_columns_list(self, tmp_path):
        """Test that empty drop list returns unchanged frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, [])

        assert clean.columns == frame.columns
        assert clean.shape == frame.shape
        df = ar.to_pandas(clean)
        assert len(df) == 2

    def test_column_order_preserved_after_drop(self, tmp_path):
        """Test that remaining columns maintain original order."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d,e\n1,2,3,4,5\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["b", "d"])

        assert clean.columns == ["a", "c", "e"]

    def test_single_row_dataset(self, tmp_path):
        """Test dropping columns from single-row dataset."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("id,name,extra\n42,Alice,xyz\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["extra"])

        df = ar.to_pandas(clean)
        assert len(df) == 1
        assert list(df.columns) == ["id", "name"]
        assert df.iloc[0]["id"] == 42


class TestDropColumnsFromDict:
    """Test drop_columns with frames created from dictionaries."""

    def test_drop_columns_from_dict_created_frame(self):
        """Test dropping columns from frame created with from_dict."""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "debug": ["x", "y", "z"]
        }
        frame = ar.from_dict(data)

        clean = ar.drop_columns(frame, ["debug"])

        df = ar.to_pandas(clean)
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 3
        assert list(df["id"]) == [1, 2, 3]


class TestDropColumnsComplexWorkflows:
    """Test complex, realistic multi-step workflows."""

    def test_full_data_cleaning_and_export_workflow(self, tmp_path):
        """Test complete workflow: load, clean, drop cols, dedupe, export."""
        input_csv = tmp_path / "messy.csv"
        input_csv.write_text(
            "id,  name  ,email,  temp_col  ,debug\n"
            "1,  Alice  ,alice@ex.com,  x  ,y\n"
            "1,  Alice  ,alice@ex.com,  x  ,y\n"
            "2,  Bob  ,bob@ex.com,  a  ,b\n"
        )
        output_csv = tmp_path / "cleaned.csv"

        # Load and process
        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_columns", {"columns": ["temp_col", "debug"]}),
            ("drop_duplicates",),
        ])

        # Write result
        ar.write_csv(clean, str(output_csv))

        # Verify
        df = ar.to_pandas(clean)
        assert list(df.columns) == ["id", "name", "email"]
        assert len(df) == 2

        # Check output file
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "temp_col" not in content
        assert "debug" not in content
        assert "alice@ex.com" in content

    def test_drop_columns_with_schema_validation(self, tmp_path):
        """Test dropping unwanted columns then validating schema."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("id,name,age,internal_id\n1,Alice,30,x\n2,Bob,25,y\n")

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["internal_id"])

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "age": ar.Int64(nullable=False),
        })

        result = ar.validate(clean, schema)
        assert result.passed is True
        assert result.issue_count == 0

    def test_drop_duplicate_columns_scenario(self, tmp_path):
        """Test dropping redundant columns from dataset."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,name_copy,email,email_copy,debug\n"
            "1,Alice,Alice,a@ex.com,a@ex.com,x\n"
            "2,Bob,Bob,b@ex.com,b@ex.com,y\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.drop_columns(frame, ["name_copy", "email_copy", "debug"])

        df = ar.to_pandas(clean)
        assert list(df.columns) == ["id", "name", "email"]
        assert len(df) == 2
