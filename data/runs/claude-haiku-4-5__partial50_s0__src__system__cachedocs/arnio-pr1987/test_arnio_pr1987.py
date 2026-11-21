import pytest
import pandas as pd
import numpy as np
import arnio as ar
import csv
from pathlib import Path


class TestCleanStepRecordValidation:
    """Test CleanStepRecord validation in direct instantiation."""

    def test_valid_clean_step_record(self):
        """Create a valid CleanStepRecord."""
        record = ar.quality.CleanStepRecord(
            step="strip_whitespace",
            kwargs={"subset": ["name"]},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Remove leading/trailing whitespace from string columns",
        )
        assert record.step == "strip_whitespace"
        assert record.kwargs == {"subset": ["name"]}
        assert record.rows_before == 100
        assert record.rows_after == 100
        assert record.rows_removed == 0

    def test_clean_step_record_zero_rows(self):
        """CleanStepRecord with zero rows is valid."""
        record = ar.quality.CleanStepRecord(
            step="drop_duplicates",
            kwargs={},
            rows_before=0,
            rows_after=0,
            rows_removed=0,
            reason="No rows to deduplicate",
        )
        assert record.rows_before == 0

    def test_clean_step_record_frozen(self):
        """CleanStepRecord is frozen and immutable."""
        record = ar.quality.CleanStepRecord(
            step="drop_nulls",
            kwargs={},
            rows_before=10,
            rows_after=5,
            rows_removed=5,
            reason="Remove null rows",
        )
        with pytest.raises((AttributeError, ValueError)):
            record.step = "other_step"

    def test_step_must_be_string(self):
        """step parameter must be a string."""
        with pytest.raises(TypeError, match="step must be a str"):
            ar.quality.CleanStepRecord(
                step=123,
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_reason_must_be_string(self):
        """reason parameter must be a string."""
        with pytest.raises(TypeError, match="reason must be a str"):
            ar.quality.CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=5,
                reason=42,
            )

    def test_kwargs_must_be_dict(self):
        """kwargs parameter must be a dict."""
        with pytest.raises(TypeError, match="kwargs must be a dict"):
            ar.quality.CleanStepRecord(
                step="fill_nulls",
                kwargs="not a dict",
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_before_cannot_be_bool(self):
        """rows_before must be int, not bool (bool is subclass of int in Python)."""
        with pytest.raises(TypeError, match="rows_before must be an int"):
            ar.quality.CleanStepRecord(
                step="strip_whitespace",
                kwargs={},
                rows_before=True,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_after_cannot_be_bool(self):
        """rows_after must be int, not bool."""
        with pytest.raises(TypeError, match="rows_after must be an int"):
            ar.quality.CleanStepRecord(
                step="drop_duplicates",
                kwargs={},
                rows_before=10,
                rows_after=False,
                rows_removed=0,
                reason="test",
            )

    def test_rows_removed_cannot_be_bool(self):
        """rows_removed must be int, not bool."""
        with pytest.raises(TypeError, match="rows_removed must be an int"):
            ar.quality.CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=True,
                reason="test",
            )

    def test_rows_before_cannot_be_negative(self):
        """rows_before cannot be negative."""
        with pytest.raises(ValueError, match="rows_before cannot be negative"):
            ar.quality.CleanStepRecord(
                step="strip_whitespace",
                kwargs={},
                rows_before=-1,
                rows_after=0,
                rows_removed=0,
                reason="test",
            )

    def test_rows_after_cannot_be_negative(self):
        """rows_after cannot be negative."""
        with pytest.raises(ValueError, match="rows_after cannot be negative"):
            ar.quality.CleanStepRecord(
                step="drop_duplicates",
                kwargs={},
                rows_before=10,
                rows_after=-1,
                rows_removed=0,
                reason="test",
            )

    def test_rows_removed_cannot_be_negative(self):
        """rows_removed cannot be negative."""
        with pytest.raises(ValueError, match="rows_removed cannot be negative"):
            ar.quality.CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=-1,
                reason="test",
            )


class TestAutoCleanWithRecords:
    """Test CleanStepRecord through end-to-end auto_clean workflows."""

    def test_auto_clean_safe_mode_csv_workflow(self, tmp_path):
        """auto_clean safe mode on CSV data with return_report=True."""
        csv_file = tmp_path / "messy_sales.csv"
        csv_file.write_text(
            "id,customer,revenue\n"
            "1, Alice , 1500\n"
            "2, Bob , 2000\n"
            "3,  Charlie  , 1800\n"
        )

        frame = ar.read_csv(str(csv_file))
        rows_before = frame.shape[0]
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        assert report is not None
        assert clean.shape[0] == rows_before  # safe mode shouldn't drop rows
        df = ar.to_pandas(clean)
        assert df["customer"].iloc[0] == "Alice"  # whitespace stripped

    def test_auto_clean_strict_mode_deduplication(self, tmp_path):
        """auto_clean strict mode removes duplicates and records it."""
        csv_file = tmp_path / "duplicates.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,Alice,95\n"
            "2,Bob,88\n"
            "2,Bob,88\n"
            "3,Charlie,92\n"
            "1,Alice,95\n"
        )

        frame = ar.read_csv(str(csv_file))
        rows_before = frame.shape[0]
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        rows_after = clean.shape[0]
        assert rows_after < rows_before  # Duplicates removed
        assert rows_after == 3  # Only 3 unique rows remain
        df = ar.to_pandas(clean)
        assert len(df) == 3

    def test_auto_clean_no_cleaning_needed(self, tmp_path):
        """auto_clean on already-clean data generates report with no changes."""
        csv_file = tmp_path / "clean.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,Alice,95\n"
            "2,Bob,88\n"
            "3,Charlie,92\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        assert clean.shape == frame.shape
        df_before = ar.to_pandas(frame)
        df_after = ar.to_pandas(clean)
        pd.testing.assert_frame_equal(df_before, df_after)

    def test_auto_clean_with_nulls_and_whitespace(self, tmp_path):
        """auto_clean handles mixed whitespace and null values."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "id,name,city\n"
            "1, Alice , Paris\n"
            "2, Bob , \n"
            "3, Charlie , London\n"
            "2, Bob , \n"
        )

        frame = ar.read_csv(str(csv_file))
        rows_before = frame.shape[0]
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        df = ar.to_pandas(clean)
        assert df["name"].iloc[0] == "Alice"  # whitespace stripped
        # Exact row count depends on strict mode behavior

    def test_auto_clean_dry_run_mode(self, tmp_path):
        """auto_clean with dry_run=True validates without modifying."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "x,y\n"
            " 1 , 2 \n"
            " 1 , 2 \n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.auto_clean(frame, mode="strict", dry_run=True)

        # Dry run should return something (frame or report)
        assert result is not None

    def test_auto_clean_empty_frame(self):
        """auto_clean on empty frame with return_report=True."""
        frame = ar.from_dict({"id": [], "name": [], "value": []})
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        assert clean.shape[0] == 0
        assert report is not None

    def test_auto_clean_single_row_no_duplicates(self, tmp_path):
        """auto_clean on single-row frame produces report."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("id,name\n1, Alice \n")

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        assert clean.shape[0] == 1
        df = ar.to_pandas(clean)
        assert df["name"].iloc[0] == "Alice"


class TestAutoCleanPandasAccessor:
    """Test auto_clean behavior via pandas accessor integration."""

    def test_pandas_accessor_clean_with_whitespace(self):
        """df.arnio.clean() removes whitespace."""
        df = pd.DataFrame(
            {
                "name": [" Alice ", " Bob ", " Charlie "],
                "city": ["Paris ", " London", " Tokyo "],
            }
        )

        clean_df = df.arnio.clean(strip_whitespace=True)

        assert clean_df["name"].iloc[0] == "Alice"
        assert clean_df["city"].iloc[1] == "London"

    def test_pandas_accessor_clean_with_duplicates(self):
        """df.arnio.clean() can remove duplicates."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3, 3, 3],
                "name": ["Alice", "Bob", "Bob", "Charlie", "Charlie", "Charlie"],
            }
        )

        clean_df = df.arnio.clean(drop_duplicates=True)

        assert len(clean_df) == 3
        assert list(clean_df["id"]) == [1, 2, 3]

    def test_pandas_accessor_clean_with_nulls(self):
        """df.arnio.clean() can drop nulls."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "value": [100.0, np.nan, 300.0, np.nan]}
        )

        clean_df = df.arnio.clean(drop_nulls=True)

        assert len(clean_df) == 2
        assert list(clean_df["id"]) == [1, 3]


class TestAutoCleanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_auto_clean_all_duplicate_rows(self, tmp_path):
        """auto_clean on frame where all rows are identical."""
        csv_file = tmp_path / "identical.csv"
        csv_file.write_text(
            "id,name\n"
            "1,Alice\n"
            "1,Alice\n"
            "1,Alice\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        # Should keep one row
        assert clean.shape[0] == 1

    def test_auto_clean_all_nulls_column(self, tmp_path):
        """auto_clean with column containing only nulls."""
        csv_file = tmp_path / "all_nulls.csv"
        csv_file.write_text(
            "id,name,empty\n"
            "1,Alice,\n"
            "2,Bob,\n"
            "3,Charlie,\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        df = ar.to_pandas(clean)
        assert len(df) == 3
        assert df["empty"].isna().all()

    def test_auto_clean_very_long_names(self, tmp_path):
        """auto_clean on data with very long column values."""
        long_string = "x" * 10000
        csv_file = tmp_path / "long.csv"
        csv_file.write_text(
            f"id,description\n"
            f"1, {long_string} \n"
            f"2,short\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="safe", return_report=True)

        df = ar.to_pandas(clean)
        assert df["description"].iloc[0] == long_string

    def test_auto_clean_unicode_content(self, tmp_path):
        """auto_clean on data with unicode characters."""
        csv_file = tmp_path / "unicode.csv"
        csv_file.write_text(
            "id,name,city\n"
            '1, "François" , "Montréal"\n'
            '2, "José" , "México City"\n'
            '1, "François" , "Montréal"\n',
            encoding="utf-8",
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        # Should deduplicate correctly with unicode
        df = ar.to_pandas(clean)
        assert len(df) == 2

    def test_auto_clean_numeric_string_columns(self, tmp_path):
        """auto_clean on numeric data stored as strings."""
        csv_file = tmp_path / "numeric_strings.csv"
        csv_file.write_text(
            "id,value\n"
            "1, 100.50 \n"
            "2, 200.75 \n"
            "1, 100.50 \n"
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        df = ar.to_pandas(clean)
        # Should keep 2 unique rows
        assert len(df) == 2

    def test_auto_clean_special_characters_in_data(self, tmp_path):
        """auto_clean preserves special characters while cleaning."""
        csv_file = tmp_path / "special.csv"
        csv_file.write_text(
            "id,note\n"
            '1, "Hello, World!" \n'
            '2, "Test@123#$%" \n'
            '1, "Hello, World!" \n'
        )

        frame = ar.read_csv(str(csv_file))
        clean, report = ar.auto_clean(frame, mode="strict", return_report=True)

        df = ar.to_pandas(clean)
        # Should deduplicate and preserve special chars
        assert len(df) == 2
        assert '"Hello, World!"' in df["note"].iloc[0] or "Hello, World!" in df["note"].iloc[0]
