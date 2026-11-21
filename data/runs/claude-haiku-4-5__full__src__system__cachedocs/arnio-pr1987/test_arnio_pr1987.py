"""
System (end-to-end) tests for arnio.quality.CleanStepRecord.

Tests drive complete workflows through arnio's public API, using real files
and verifying outcomes that users would observe.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

import arnio as ar
from arnio.quality import CleanStepRecord


class TestCleanStepRecordConstruction:
    """Test direct construction and validation of CleanStepRecord."""

    def test_valid_construction(self):
        """Create a valid CleanStepRecord with all correct types."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={"subset": ["col1"]},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Removed leading/trailing whitespace from text columns",
        )
        assert record.step == "strip_whitespace"
        assert record.rows_before == 100
        assert record.rows_after == 100
        assert record.rows_removed == 0
        assert record.kwargs == {"subset": ["col1"]}

    def test_rows_removed_calculation(self):
        """CleanStepRecord correctly tracks row removal."""
        record = CleanStepRecord(
            step="drop_duplicates",
            kwargs={"keep": "first"},
            rows_before=1000,
            rows_after=950,
            rows_removed=50,
            reason="Removed exact duplicate rows",
        )
        assert record.rows_removed == 50
        assert record.rows_before - record.rows_after == record.rows_removed

    def test_zero_rows(self):
        """CleanStepRecord accepts zero row counts."""
        record = CleanStepRecord(
            step="drop_nulls",
            kwargs={"subset": []},
            rows_before=0,
            rows_after=0,
            rows_removed=0,
            reason="No rows in input",
        )
        assert record.rows_before == 0
        assert record.rows_after == 0

    def test_step_must_be_string(self):
        """CleanStepRecord.step must be a string, not other types."""
        with pytest.raises(TypeError, match="step must be a str"):
            CleanStepRecord(
                step=42,
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_reason_must_be_string(self):
        """CleanStepRecord.reason must be a string."""
        with pytest.raises(TypeError, match="reason must be a str"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason=None,
            )

    def test_kwargs_must_be_dict(self):
        """CleanStepRecord.kwargs must be a dict."""
        with pytest.raises(TypeError, match="kwargs must be a dict"):
            CleanStepRecord(
                step="strip_whitespace",
                kwargs=["subset", "col1"],
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_counts_must_be_int_not_bool(self):
        """CleanStepRecord rejects bool for row counts (bool is int subclass)."""
        with pytest.raises(TypeError, match="rows_before must be an int"):
            CleanStepRecord(
                step="strip_whitespace",
                kwargs={},
                rows_before=True,  # bool passes int check in some contexts
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_counts_must_be_int_not_float(self):
        """CleanStepRecord rejects float for row counts."""
        with pytest.raises(TypeError, match="rows_after must be an int"):
            CleanStepRecord(
                step="strip_whitespace",
                kwargs={},
                rows_before=10,
                rows_after=10.5,
                rows_removed=0,
                reason="test",
            )

    def test_rows_counts_cannot_be_negative(self):
        """CleanStepRecord rejects negative row counts."""
        with pytest.raises(ValueError, match="rows_before cannot be negative"):
            CleanStepRecord(
                step="strip_whitespace",
                kwargs={},
                rows_before=-1,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

        with pytest.raises(ValueError, match="rows_removed cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=-1,
                reason="test",
            )

    def test_frozen_dataclass_immutable(self):
        """CleanStepRecord is immutable (frozen=True)."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason="test",
        )
        with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
            record.step = "drop_nulls"


class TestCleanStepRecordWithAutoClean:
    """Test CleanStepRecord integration with auto_clean and pipeline workflows."""

    def test_auto_clean_safe_mode_produces_records(self, tmp_path):
        """auto_clean in safe mode should track whitespace stripping."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text("name,score\n" " Alice , 95 \n" " Bob , 88 \n")

        frame = ar.read_csv(str(csv_path))
        rows_before = len(frame)

        clean = ar.auto_clean(frame, mode="safe")
        rows_after = len(clean)

        # Safe mode should not drop rows, only strip whitespace
        assert rows_after == rows_before

        df = ar.to_pandas(clean)
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"

    def test_auto_clean_strict_mode_removes_duplicates(self, tmp_path):
        """auto_clean strict mode removes duplicates and tracks row count."""
        csv_path = tmp_path / "dups.csv"
        csv_path.write_text(
            "id,name\n" "1,Alice\n" "2,Bob\n" "1,Alice\n" "3,Charlie\n"
        )

        frame = ar.read_csv(str(csv_path))
        rows_before = len(frame)

        clean = ar.auto_clean(frame, mode="strict")
        rows_after = len(clean)

        # Strict mode should remove duplicates
        assert rows_after < rows_before
        assert rows_after == 3

    def test_pipeline_with_drop_nulls_tracking(self, tmp_path):
        """Pipeline step drop_nulls should be trackable via row counts."""
        csv_path = tmp_path / "nulls.csv"
        csv_path.write_text("id,name,score\n" "1,Alice,95\n" "2,,88\n" "3,Charlie,\n")

        frame = ar.read_csv(str(csv_path))
        rows_before = len(frame)

        clean = ar.pipeline(frame, [("drop_nulls",)])
        rows_after = len(clean)

        assert rows_after < rows_before
        assert rows_after == 1  # Only first row has no nulls

    def test_pipeline_with_drop_duplicates_tracking(self, tmp_path):
        """Pipeline step drop_duplicates should be trackable."""
        csv_path = tmp_path / "dups2.csv"
        csv_path.write_text(
            "id,name\n" "1,Alice\n" "1,Alice\n" "2,Bob\n" "2,Bob\n" "2,Bob\n"
        )

        frame = ar.read_csv(str(csv_path))
        rows_before = len(frame)

        clean = ar.pipeline(frame, [("drop_duplicates", {"keep": "first"})])
        rows_after = len(clean)

        assert rows_before == 5
        assert rows_after == 2


class TestCleanStepRecordWithPandasAccessor:
    """Test CleanStepRecord through pandas .arnio accessor."""

    def test_pandas_accessor_clean_strips_whitespace(self):
        """DataFrame.arnio.clean() with strip_whitespace produces record-like behavior."""
        df = pd.DataFrame(
            {"name": [" Alice ", " Bob ", " Charlie "], "age": [25, 30, 35]}
        )

        clean_df = df.arnio.clean(strip_whitespace=True)

        assert clean_df["name"].iloc[0] == "Alice"
        assert clean_df["name"].iloc[1] == "Bob"
        assert len(clean_df) == len(df)  # No rows dropped

    def test_pandas_accessor_clean_drops_duplicates(self):
        """DataFrame.arnio.clean() with drop_duplicates removes rows."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 1, 3],
                "name": ["Alice", "Bob", "Alice", "Charlie"],
            }
        )

        clean_df = df.arnio.clean(drop_duplicates=True)

        assert len(clean_df) == 3
        assert len(df) == 4

    def test_pandas_accessor_profile_generates_report(self):
        """DataFrame.arnio.profile() returns quality report."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, None], "name": ["Alice", "Bob", "Charlie", "David"]}
        )

        report = df.arnio.profile()

        assert report.row_count == 4
        assert report.column_count == 2


class TestCleanStepRecordEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_no_rows(self, tmp_path):
        """CleanStepRecord behavior with empty DataFrames."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("id,name\n")

        frame = ar.read_csv(str(csv_path))
        assert len(frame) == 0

        clean = ar.auto_clean(frame, mode="safe")
        assert len(clean) == 0

    def test_single_row_dataframe(self, tmp_path):
        """CleanStepRecord works with single-row DataFrames."""
        csv_path = tmp_path / "single.csv"
        csv_path.write_text("id,name\n1, Alice \n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean)

        assert len(df) == 1
        assert df["name"].iloc[0] == "Alice"

    def test_all_duplicate_rows(self, tmp_path):
        """CleanStepRecord handles DataFrames with all identical rows."""
        csv_path = tmp_path / "all_dups.csv"
        csv_path.write_text(
            "id,name\n" "1,Alice\n" "1,Alice\n" "1,Alice\n" "1,Alice\n"
        )

        frame = ar.read_csv(str(csv_path))
        clean = ar.auto_clean(frame, mode="strict")
        df = ar.to_pandas(clean)

        assert len(df) == 1  # Only one unique row remains

    def test_large_row_count_handling(self):
        """CleanStepRecord correctly handles large row counts."""
        large_count = 100_000
        df = pd.DataFrame(
            {
                "id": range(large_count),
                "value": np.random.random(large_count),
            }
        )

        frame = ar.from_pandas(df)
        assert len(frame) == large_count

        # Apply a non-destructive step
        clean = ar.pipeline(frame, [("normalize_case",)])
        assert len(clean) == large_count

    def test_many_duplicates(self, tmp_path):
        """CleanStepRecord tracks removal of many duplicate rows."""
        csv_path = tmp_path / "many_dups.csv"

        # Generate CSV with many duplicates
        with open(csv_path, "w") as f:
            f.write("id,name\n")
            for i in range(100):
                f.write(f"{i % 10},name_{i % 10}\n")

        frame = ar.read_csv(str(csv_path))
        rows_before = len(frame)

        clean = ar.pipeline(frame, [("drop_duplicates", {"keep": "first"})])
        rows_after = len(clean)

        assert rows_before == 100
        assert rows_after == 10
        assert rows_before - rows_after == 90


class TestCleanStepRecordRealWorldWorkflows:
    """Test realistic end-to-end cleaning workflows."""

    def test_complete_cleaning_pipeline_with_export(self, tmp_path):
        """Full workflow: read messy CSV → clean → validate → export."""
        csv_in = tmp_path / "sales_raw.csv"
        csv_in.write_text(
            "order_id,customer,revenue,city\n"
            " 1001 , Alice , 5000 , New York \n"
            " 1002 , Bob , 3000 , Boston \n"
            " 1001 , Alice , 5000 , New York \n"  # Duplicate
            " 1003 , Charlie , , Boston \n"  # Null revenue
        )

        # Load data
        frame = ar.read_csv(str(csv_in))
        rows_start = len(frame)

        # Apply comprehensive pipeline
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_nulls", {"subset": ["revenue"]}),
                ("drop_duplicates",),
            ],
        )

        rows_end = len(clean)

        # Verify row count changed
        assert rows_start == 4
        assert rows_end == 2

        # Export result
        csv_out = tmp_path / "sales_clean.csv"
        ar.write_csv(clean, str(csv_out))

        # Verify output file is valid
        assert csv_out.exists()
        output_df = pd.read_csv(str(csv_out))
        assert len(output_df) == 2
        assert list(output_df["customer"]) == ["Alice", "Bob"]

    def test_profile_suggest_then_clean(self, tmp_path):
        """Workflow: profile → get suggestions → apply cleaning."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,score\n"
            "1, Alice , 95 \n"
            "2, Bob , 88 \n"
            "3, Alice , 95 \n"
        )

        frame = ar.read_csv(str(csv_path))

        # Profile the data
        report = ar.profile(frame)
        assert report.row_count == 3

        # Get cleaning suggestions
        suggestions = ar.suggest_cleaning(frame)
        assert len(suggestions) > 0

        # Apply suggestions
        clean = ar.pipeline(frame, suggestions)
        df = ar.to_pandas(clean)

        # Verify cleaning applied
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"
        assert len(df) == 2  # Duplicates removed

    def test_roundtrip_pandas_to_arnio_to_pandas(self):
        """Verify data integrity in pandas → arnio → pandas roundtrip."""
        df_original = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": [" Alice ", " Bob ", " Charlie ", " David ", " Eve "],
                "score": [95, 88, 92, None, 87],
            }
        )

        # Convert to ArFrame
        frame = ar.from_pandas(df_original)

        # Apply cleaning
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_nulls", {"subset": ["score"]}),
            ],
        )

        # Convert back to pandas
        df_result = ar.to_pandas(clean)

        assert len(df_result) == 4  # One row with null dropped
        assert df_result["name"].iloc[0] == "Alice"
        assert df_result["score"].isna().sum() == 0
