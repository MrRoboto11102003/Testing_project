import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import arnio as ar


class TestDropColumnsBasic:
    """Basic drop_columns functionality."""

    def test_drop_single_column_from_csv(self, tmp_path):
        """Drop a single column from loaded CSV."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")

        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns(frame, ["age"])

        assert result.columns == ["name", "city"]
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert "age" not in df.columns
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_drop_multiple_columns(self, tmp_path):
        """Drop multiple columns at once."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,email,phone,city\n1,Alice,a@ex.com,555-1234,NYC\n2,Bob,b@ex.com,555-5678,LA\n")

        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns(frame, ["email", "phone"])

        assert result.columns == ["id", "name", "city"]
        assert len(ar.to_pandas(result).columns) == 3

    def test_drop_columns_preserves_order(self, tmp_path):
        """Remaining columns maintain their original order."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c,d,e\n1,2,3,4,5\n")

        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns(frame, ["b", "d"])

        # Order should be preserved: a, c, e
        assert result.columns == ["a", "c", "e"]

    def test_drop_columns_empty_list_no_change(self, tmp_path):
        """Dropping empty list returns frame unchanged."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y,z\n1,2,3\n2,4,6\n")

        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns(frame, [])

        assert result.columns == ["x", "y", "z"]
        pd.testing.assert_frame_equal(ar.to_pandas(frame), ar.to_pandas(result))

    def test_drop_columns_write_csv(self, tmp_path):
        """Drop columns and write result to CSV."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_csv.write_text("id,name,internal_col,score\n1,Alice,secret,95\n2,Bob,private,88\n")

        frame = ar.read_csv(str(input_csv))
        clean = ar.drop_columns(frame, ["internal_col"])
        ar.write_csv(clean, str(output_csv))

        # Verify the output file
        output_content = output_csv.read_text()
        assert "internal_col" not in output_content
        assert "id,name,score" in output_content

        # Verify by reading back
        frame2 = ar.read_csv(str(output_csv))
        assert frame2.columns == ["id", "name", "score"]


class TestDropColumnsErrors:
    """Error handling for drop_columns."""

    def test_column_not_found_raises_valueerror(self, tmp_path):
        """Missing column raises ValueError with descriptive message."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(frame, ["nonexistent"])

    def test_multiple_missing_columns_raises_valueerror(self, tmp_path):
        """Multiple missing columns all raise ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(frame, ["x", "y", "z"])

    def test_string_instead_of_list_raises_typeerror(self, tmp_path):
        """Passing string instead of list raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age\nAlice,30\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError):
            ar.drop_columns(frame, "name")

    def test_non_string_in_list_raises_typeerror(self, tmp_path):
        """List with non-string column names raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError):
            ar.drop_columns(frame, ["a", 123, "c"])

    def test_drop_all_columns_raises_valueerror(self, tmp_path):
        """Trying to drop all columns raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y,z\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(frame, ["x", "y", "z"])

    def test_case_sensitive_column_names(self, tmp_path):
        """Column names are case-sensitive."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("Name,age\nAlice,30\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(frame, ["name"])  # lowercase, not "Name"


class TestDropColumnsPipeline:
    """drop_columns in cleaning pipelines and workflows."""

    def test_drop_columns_in_pipeline(self, tmp_path):
        """Use drop_columns as a pipeline step."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,temp_col,email\n1,Alice,xxx,a@ex.com\n2,Bob,yyy,b@ex.com\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("drop_columns", {"columns": ["temp_col"]}),
        ])

        assert clean.columns == ["id", "name", "email"]
        df = ar.to_pandas(clean)
        assert len(df) == 2

    def test_drop_columns_with_other_steps(self, tmp_path):
        """Combine drop_columns with other cleaning steps."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,debug,age\n Alice ,x,30\n Bob ,y,25\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace", {"subset": ["name"]}),
            ("drop_columns", {"columns": ["debug"]}),
        ])

        assert clean.columns == ["name", "age"]
        df = ar.to_pandas(clean)
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_pandas_accessor_with_drop_columns_in_pipeline(self, tmp_path):
        """Use pandas .arnio accessor with drop_columns in pipeline."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "debug_col": ["x", "y", "z"]
        })

        clean_df = df.arnio.clean(steps=[
            ("drop_columns", {"columns": ["debug_col"]})
        ])

        assert list(clean_df.columns) == ["id", "name"]
        assert len(clean_df) == 3


class TestDropColumnsDataIntegrity:
    """Data integrity after dropping columns."""

    def test_drop_columns_profile_then_write(self, tmp_path):
        """Drop columns, profile, then write to CSV."""
        csv_path = tmp_path / "data.csv"
        output_path = tmp_path / "clean.csv"
        csv_path.write_text("user_id,name,temp_col,score\n1,Alice,x,95\n2,Bob,y,88\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns(frame, ["temp_col"])

        # Profile the cleaned data
        report = ar.profile(clean)
        assert len(report.columns) == 3  # user_id, name, score
        assert "temp_col" not in report.columns

        # Write to file
        ar.write_csv(clean, str(output_path))
        assert output_path.exists()

        # Read back and verify
        frame_check = ar.read_csv(str(output_path))
        assert frame_check.columns == ["user_id", "name", "score"]

    def test_drop_columns_then_validate(self, tmp_path):
        """Drop columns then validate against schema."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,email,debug_col\n1,alice@ex.com,x\n2,bob@ex.com,y\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns(frame, ["debug_col"])

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(clean, schema)
        assert result.passed
        assert result.issue_count == 0

    def test_drop_columns_maintains_data_types(self, tmp_path):
        """Dropping columns preserves data types of remaining columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score,flag,text\n1,95.5,true,hello\n2,88.3,false,world\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns(frame, ["text"])

        df = ar.to_pandas(clean)
        # Verify columns exist and have data
        assert "id" in df.columns
        assert "score" in df.columns
        assert "flag" in df.columns
        assert len(df) == 2
        assert df["id"].tolist() == [1, 2]

    def test_drop_columns_with_special_characters_in_names(self, tmp_path):
        """Handle column names with special characters correctly."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text('"first-name","last_name","debug-col"\nAlice,Smith,x\nBob,Jones,y\n')

        frame = ar.read_csv(str(csv_path))
        clean = ar.drop_columns(frame, ["debug-col"])

        assert "first-name" in clean.columns
        assert "last_name" in clean.columns
        assert "debug-col" not in clean.columns
        assert len(ar.to_pandas(clean).columns) == 2
