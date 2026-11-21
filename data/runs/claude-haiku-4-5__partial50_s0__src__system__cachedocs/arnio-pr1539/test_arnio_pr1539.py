import pytest
import pandas as pd
import arnio as ar
import tempfile
from pathlib import Path


class TestDropColumnsBasic:
    """Test basic drop_columns functionality."""

    def test_drop_columns_with_list(self, tmp_path):
        """Drop columns using a list."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,age,city\n1,Alice,30,NYC\n2,Bob,25,LA\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["age", "city"])

        assert result.columns == ["id", "name"]
        df = ar.to_pandas(result)
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 2

    def test_drop_columns_with_tuple(self, tmp_path):
        """Drop columns using a tuple."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(("b", "d"))

        assert result.columns == ["a", "c"]
        df = ar.to_pandas(result)
        assert list(df.columns) == ["a", "c"]

    def test_drop_columns_preserves_order(self, tmp_path):
        """Original column order is preserved after dropping."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c,d,e\n1,2,3,4,5\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["b", "d"])

        assert result.columns == ["a", "c", "e"]

    def test_drop_columns_with_duplicates(self, tmp_path):
        """Duplicate column names in drop list are silently ignored."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y,z\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["y", "y", "y"])

        assert result.columns == ["x", "z"]

    def test_drop_columns_empty_list(self, tmp_path):
        """Empty drop list returns unchanged copy."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns([])

        assert result.columns == frame.columns
        assert result.shape == frame.shape


class TestDropColumnsErrors:
    """Test error conditions for drop_columns."""

    def test_drop_columns_nonexistent_column(self, tmp_path):
        """ValueError when dropping non-existent column."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["nonexistent"])

    def test_drop_columns_wrong_type_set(self, tmp_path):
        """TypeError when cols is a set instead of list/tuple."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            frame.drop_columns({"a"})

    def test_drop_columns_wrong_type_string(self, tmp_path):
        """TypeError when cols is a string instead of list."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            frame.drop_columns("a")

    def test_drop_columns_non_string_elements(self, tmp_path):
        """TypeError when list contains non-string column names."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="All column names.*must be strings"):
            frame.drop_columns(["a", 1, "c"])

    def test_drop_columns_all_columns(self, tmp_path):
        """ValueError when trying to drop all columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(["a", "b", "c"])


class TestDropColumnsWorkflows:
    """Test drop_columns in real-world workflows."""

    def test_drop_columns_csv_read_write_cycle(self, tmp_path):
        """Read CSV, drop columns, write to new CSV, verify output."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        input_csv.write_text(
            "id,name,email,phone,debug_field\n1,Alice,alice@ex.com,555-1234,xyz\n2,Bob,bob@ex.com,555-5678,abc\n"
        )

        frame = ar.read_csv(str(input_csv))
        clean = frame.drop_columns(["email", "phone", "debug_field"])
        ar.write_csv(clean, str(output_csv))

        assert output_csv.exists()
        output_df = pd.read_csv(output_csv)
        assert list(output_df.columns) == ["id", "name"]
        assert len(output_df) == 2

    def test_drop_columns_in_pipeline(self, tmp_path):
        """Use drop_columns within ar.pipeline()."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,  age  ,temp_col\n1, Alice ,30,x\n2, Bob ,25,y\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_columns", {"cols": ["temp_col"]}),
            ],
        )

        df = ar.to_pandas(clean)
        assert list(df.columns) == ["id", "name", "age"]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_drop_columns_then_validate(self, tmp_path):
        """Drop columns then validate remaining schema."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("user_id,email,unused_field,score\n1,alice@ex.com,xxx,95\n2,bob@ex.com,yyy,87\n")

        frame = ar.read_csv(str(csv_path))
        clean = frame.drop_columns(["unused_field"])

        schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "score": ar.Int64(nullable=False),
            }
        )
        result = ar.validate(clean, schema)

        assert result.passed

    def test_drop_columns_with_profile(self, tmp_path):
        """Profile data before and after dropping columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,notes\n1,Alice,some notes\n2,Bob,other notes\n3,Charlie,\n")

        frame = ar.read_csv(str(csv_path))
        profile_before = ar.profile(frame)

        clean = frame.drop_columns(["notes"])
        profile_after = ar.profile(clean)

        assert len(profile_before.columns) == 3
        assert len(profile_after.columns) == 2
        assert "notes" not in profile_after.columns

    def test_drop_columns_pandas_interop(self, tmp_path):
        """Convert pandas DataFrame, drop columns, convert back."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")

        df = pd.read_csv(csv_path)
        frame = ar.from_pandas(df)
        clean_frame = frame.drop_columns(["b", "d"])
        result_df = ar.to_pandas(clean_frame)

        assert list(result_df.columns) == ["a", "c"]
        assert len(result_df) == 2
        assert result_df["a"].tolist() == [1, 5]

    def test_drop_columns_chained_operations(self, tmp_path):
        """Chain multiple drop_columns calls."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c,d,e,f\n1,2,3,4,5,6\n")

        frame = ar.read_csv(str(csv_path))
        step1 = frame.drop_columns(["f"])
        step2 = step1.drop_columns(["d"])
        step3 = step2.drop_columns(["b"])

        assert step3.columns == ["a", "c", "e"]
        df = ar.to_pandas(step3)
        assert len(df) == 1

    def test_drop_columns_with_auto_clean(self, tmp_path):
        """Drop columns before auto_clean."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,  junk  \n1, Alice ,x\n2, Bob ,y\n")

        frame = ar.read_csv(str(csv_path))
        clean = frame.drop_columns(["junk"])
        final = ar.auto_clean(clean, mode="safe")

        df = ar.to_pandas(final)
        assert list(df.columns) == ["id", "name"]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_drop_columns_case_sensitive(self, tmp_path):
        """Column names are case-sensitive."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("Name,name,NAME\nalice,alice,alice\nbob,bob,bob\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["name"])

        assert result.columns == ["Name", "NAME"]
        assert len(result.columns) == 2

    def test_drop_columns_whitespace_in_names(self, tmp_path):
        """Handle column names with spaces."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("first name,last name,age\nAlice,Smith,30\nBob,Jones,25\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["age"])

        assert result.columns == ["first name", "last name"]

    def test_drop_columns_maintains_data_integrity(self, tmp_path):
        """Data is not corrupted after dropping columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,score,status\n1,Alice,95,active\n2,Bob,87,inactive\n3,Charlie,92,active\n"
        )

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["status"])

        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert df["score"].tolist() == [95, 87, 92]
        assert list(df.columns) == ["id", "name", "score"]

    def test_drop_columns_single_column_drop(self, tmp_path):
        """Drop a single column from multi-column frame."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y,z\n1,2,3\n4,5,6\n")

        frame = ar.read_csv(str(csv_path))
        result = frame.drop_columns(["y"])

        assert result.columns == ["x", "z"]
        assert result.shape == (2, 2)
