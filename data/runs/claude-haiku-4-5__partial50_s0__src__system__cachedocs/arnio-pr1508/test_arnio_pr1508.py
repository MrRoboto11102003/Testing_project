"""
End-to-end system tests for arnio.pipeline.pipeline.

These tests drive complete user workflows through arnio's public API,
using real files on disk and asserting on observable outputs.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestPipelineBasicWorkflows:
    """Test basic pipeline workflows with real CSV files."""

    def test_pipeline_basic_strip_and_drop_nulls(self, tmp_path):
        """Test a simple pipeline: strip whitespace then drop null rows."""
        # Create a messy CSV with extra spaces and missing values
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "name,age,city\n"
            " Alice ,25, Paris \n"
            "Bob,30,\n"
            "Charlie,35, London "
        )

        # Load, clean, and validate
        frame = ar.read_csv(str(csv_path))
        assert frame.shape[0] == 3

        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls",),
        ])

        df = ar.to_pandas(clean)
        # Bob's row should be dropped (empty city)
        assert len(df) == 2
        assert df["name"].tolist() == ["Alice", "Charlie"]
        assert df["city"].tolist() == ["Paris", "London"]
        # Verify whitespace actually stripped
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "city"] == "Paris"

    def test_pipeline_with_return_metadata(self, tmp_path):
        """Test pipeline with return_metadata=True."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,value\n1,100\n2,200\n2,200")

        frame = ar.read_csv(str(csv_path))
        clean, metadata = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], return_metadata=True)

        # Verify metadata structure
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata

        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        assert len(metadata["step_timings"]) == 2
        assert len(metadata["row_counts"]) == 2

        # Each step should have timing data
        for timing in metadata["step_timings"]:
            assert "seconds" in timing
            assert isinstance(timing["seconds"], (int, float))
            assert timing["seconds"] >= 0

        # Row counts should show before/after
        for rc in metadata["row_counts"]:
            assert "before" in rc
            assert "after" in rc
            assert rc["before"] > 0

    def test_pipeline_dry_run_does_not_modify(self, tmp_path):
        """Test that dry_run=True validates pipeline without changing data."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1, Alice \n2, Bob ")

        frame = ar.read_csv(str(csv_path))
        rows_before = frame.shape[0]

        # Run with dry_run=True
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], dry_run=True)

        # Result frame should not be modified
        assert result.shape[0] == rows_before

        # Original frame is unchanged
        df = ar.to_pandas(result)
        assert " Alice " in df["name"].values or " " in df["name"].iloc[0]

    def test_pipeline_track_lineage_single_drop_step(self, tmp_path):
        """Test track_lineage=True to track which rows were dropped."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,value\n1,100\n2,200\n3,\n4,400")

        frame = ar.read_csv(str(csv_path))
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls",),
        ], track_lineage=True)

        # Verify lineage report
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
        assert lineage.total_dropped == 1  # Row 3 (0-indexed row 2)
        assert "drop_nulls" in lineage.dropped_by_step
        # Row with null was dropped (1-based indexing: row 3)
        assert 3 in lineage.dropped_by_step["drop_nulls"]

        # Result frame should have 3 rows
        assert result.shape[0] == 3

    def test_pipeline_track_lineage_multiple_steps(self, tmp_path):
        """Test lineage tracking with multiple drop operations."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name\n"
            "1,Alice\n"
            "2,Bob\n"
            "3,\n"
            "4,Charlie\n"
            "4,Charlie"
        )

        frame = ar.read_csv(str(csv_path))
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls",),
            ("drop_duplicates",),
        ], track_lineage=True)

        # Verify tracking
        assert lineage.total_dropped > 0
        assert "drop_nulls" in lineage.dropped_by_step
        assert "drop_duplicates" in lineage.dropped_by_step

        df = ar.to_pandas(result)
        # Should have 3 unique rows (Alice, Bob, Charlie)
        assert len(df) == 3

    def test_pipeline_drop_duplicates(self, tmp_path):
        """Test deduplication via pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age\nAlice,25\nBob,30\nAlice,25\nCharlie,25")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"}),
        ])

        df = ar.to_pandas(clean)
        assert len(df) == 3
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_pipeline_cast_types(self, tmp_path):
        """Test type casting in pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score,active\n1,95.5,true\n2,87.3,false\n3,92.1,true")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("cast_types", {"mapping": {"id": "int64", "score": "float64"}}),
        ])

        df = ar.to_pandas(clean)
        assert df["id"].dtype == "int64"
        assert df["score"].dtype == "float64"

    def test_pipeline_rename_columns(self, tmp_path):
        """Test column renaming in pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("first_name,last_name\nAlice,Smith\nBob,Jones")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("rename_columns", {"first_name": "fname", "last_name": "lname"}),
        ])

        df = ar.to_pandas(clean)
        assert "fname" in df.columns
        assert "lname" in df.columns
        assert "first_name" not in df.columns
        assert "last_name" not in df.columns

    def test_pipeline_fill_nulls(self, tmp_path):
        """Test fill_nulls step in pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,95\n2,\n3,87")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["score"]}),
        ])

        df = ar.to_pandas(clean)
        assert df["score"].isna().sum() == 0
        assert 0.0 in df["score"].values or "0" in df["score"].astype(str).values


class TestPipelineValidationAndErrors:
    """Test error handling and validation in pipelines."""

    def test_pipeline_invalid_step_tuple_too_many_elements(self, tmp_path):
        """Test that step tuples with too many elements fail early."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [
                ("strip_whitespace", {}, "extra_arg"),
            ])

    def test_pipeline_invalid_step_kwargs_not_dict(self, tmp_path):
        """Test that non-dict kwargs raises error."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [
                ("strip_whitespace", "not_a_dict"),
            ])

    def test_pipeline_unknown_step_name(self, tmp_path):
        """Test that unknown step names raise UnknownStepError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [
                ("nonexistent_step_name",),
            ])

    def test_pipeline_track_lineage_and_metadata_mutually_exclusive(self, tmp_path):
        """Test that track_lineage and return_metadata cannot be used together."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [
                ("strip_whitespace",),
            ], track_lineage=True, return_metadata=True)

    def test_pipeline_type_check_frame_parameter(self, tmp_path):
        """Test that frame parameter type is validated."""
        # Pass a pandas DataFrame instead of ArFrame
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [("strip_whitespace",)])

    def test_pipeline_type_check_return_metadata_parameter(self, tmp_path):
        """Test that return_metadata parameter type is validated."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob")

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [("strip_whitespace",)], return_metadata="yes")


class TestPipelineComplexWorkflows:
    """Test complex multi-step pipelines."""

    def test_pipeline_multi_step_complex_workflow(self, tmp_path):
        """Test a realistic complex pipeline with multiple transformations."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,score,status\n"
            "1, Alice ,95, Active \n"
            "2, Bob ,87, Inactive \n"
            "3, Alice ,95, Active \n"
            "4,Charlie,,pending"
        )

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_nulls", {"subset": ["score"]}),
            ("drop_duplicates",),
        ])

        df = ar.to_pandas(clean)
        # Should have Alice and Bob (Charlie dropped for null score, Alice duplicate removed)
        assert len(df) == 2
        assert "alice" in df["name"].values or "ALICE" in df["name"].values
        assert "bob" in df["name"].values or "BOB" in df["name"].values

    def test_pipeline_with_custom_python_step(self, tmp_path):
        """Test custom Python step registration and use."""
        def double_numeric_columns(df, columns=None):
            """Multiply numeric columns by 2."""
            df = df.copy()
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col] * 2
            return df

        # Register custom step
        ar.register_step("double_numeric", double_numeric_columns)

        try:
            csv_path = tmp_path / "data.csv"
            csv_path.write_text("id,value\n1,100\n2,200\n3,300")

            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(frame, [
                ("double_numeric", {"columns": ["value"]}),
            ])

            df = ar.to_pandas(clean)
            assert df["value"].tolist() == [200, 400, 600]
        finally:
            ar.reset_steps()

    def test_pipeline_via_pandas_accessor_clean(self, tmp_path):
        """Test pipeline execution via pandas .arnio.clean() accessor."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "name,age\n"
            " Alice ,25\n"
            "Bob,30\n"
            " Alice ,25"
        )

        # Use pandas accessor
        df = pd.read_csv(str(csv_path))
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True,
        )

        assert len(clean_df) == 2
        assert clean_df["name"].tolist() == ["Alice", "Bob"]
        # Verify whitespace removed
        assert " " not in clean_df["name"][0]

    def test_pipeline_round_trip_csv_to_pandas_to_csv(self, tmp_path):
        """Test full round-trip: CSV → ArFrame → pipeline → pandas → CSV."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "id,customer,amount\n"
            "1, Alice ,1000\n"
            "2, Bob ,2000\n"
            "1, Alice ,1000"
        )

        # Load and clean
        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        # Convert to pandas and save
        df = ar.to_pandas(clean)
        output_csv = tmp_path / "output.csv"
        df.to_csv(str(output_csv), index=False)

        # Verify output
        result_df = pd.read_csv(str(output_csv))
        assert len(result_df) == 2
        assert "Alice" in result_df["customer"].values
        assert "Bob" in result_df["customer"].values

    def test_pipeline_filter_rows_with_comparison_operators(self, tmp_path):
        """Test filter_rows with various comparison operators."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,age\n1,15\n2,25\n3,35\n4,45\n5,55")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "age", "op": ">=", "value": 30}),
        ])

        df = ar.to_pandas(clean)
        assert len(df) == 3
        assert (df["age"] >= 30).all()
        assert df["age"].tolist() == [35, 45, 55]

    def test_pipeline_normalize_case_with_different_modes(self, tmp_path):
        """Test normalize_case with different case_type options."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nAlice\nBob\nCharlie")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
        ])

        df = ar.to_pandas(clean)
        assert df["name"].tolist() == ["alice", "bob", "charlie"]
