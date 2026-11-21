"""
System (end-to-end) tests for arnio.pipeline.pipeline

These tests drive complete user workflows through arnio's pipeline interface,
covering basic usage, metadata tracking, lineage tracking, error handling,
custom steps, and integration with pandas and disk I/O.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestPipelineBasic:
    """Basic pipeline execution tests"""

    def test_single_step_pipeline(self, tmp_path):
        """Test pipeline with a single whitespace-stripping step"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age\n alice ,30\n bob ,25\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        result = ar.to_pandas(clean)

        assert result["name"].tolist() == ["alice", "bob"]
        assert len(result) == 2

    def test_multiple_steps_pipeline(self, tmp_path):
        """Test pipeline with multiple sequential steps"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "name,age,city\nalice,30,PARIS\nbob,25,LONDON\nalice,30,PARIS\n"
        )

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
                ("drop_duplicates", {"keep": "first"}),
            ],
        )
        result = ar.to_pandas(clean)

        assert len(result) == 2
        assert result["city"].tolist() == ["paris", "london"]

    def test_pipeline_drop_nulls(self, tmp_path):
        """Test pipeline with drop_nulls to remove incomplete rows"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,email\n1,alice,alice@example.com\n2,bob,\n3,charlie,charlie@example.com\n"
        )

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("drop_nulls", {"subset": ["email"]})])
        result = ar.to_pandas(clean)

        assert len(result) == 2
        assert result["name"].tolist() == ["alice", "charlie"]

    def test_pipeline_fill_nulls(self, tmp_path):
        """Test pipeline with fill_nulls to impute missing values"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,85\n2,\n3,90\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("fill_nulls", {"value": 0.0, "subset": ["score"]})])
        result = ar.to_pandas(clean)

        assert result["score"].tolist() == [85.0, 0.0, 90.0]

    def test_empty_pipeline_returns_frame_unchanged(self, tmp_path):
        """Test that an empty pipeline step list returns the original frame"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\nbob\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [])
        result = ar.to_pandas(clean)

        assert len(result) == 2
        assert result["name"].tolist() == ["alice", "bob"]


class TestPipelineMetadata:
    """Tests for return_metadata=True"""

    def test_return_metadata_structure(self, tmp_path):
        """Test that metadata dict contains expected keys and structure"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age\nalice,30\nbob,25\n")

        frame = ar.read_csv(str(csv_path))
        clean, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True,
        )

        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        assert len(metadata["row_counts"]) == 2
        assert len(metadata["step_timings"]) == 2

    def test_metadata_row_counts_tracking(self, tmp_path):
        """Test that metadata correctly tracks row counts through pipeline"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,val\n1,1\n2,2\n2,2\n3,3\n")

        frame = ar.read_csv(str(csv_path))
        clean, metadata = ar.pipeline(
            frame, [("drop_duplicates", {"keep": "first"})], return_metadata=True
        )

        row_counts = metadata["row_counts"]
        assert row_counts[0]["before"] == 4
        assert row_counts[0]["after"] == 3

    def test_metadata_includes_timing(self, tmp_path):
        """Test that metadata includes per-step timing information"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\nbob\n")

        frame = ar.read_csv(str(csv_path))
        clean, metadata = ar.pipeline(
            frame, [("strip_whitespace",)], return_metadata=True
        )

        step_timings = metadata["step_timings"]
        assert len(step_timings) == 1
        assert "seconds" in step_timings[0]
        assert step_timings[0]["step"] == "strip_whitespace"


class TestPipelineDryRun:
    """Tests for dry_run=True"""

    def test_dry_run_does_not_modify_output(self, tmp_path):
        """Test that dry_run validates pipeline but returns input frame unchanged"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\nbob\nbob\n")

        frame = ar.read_csv(str(csv_path))
        result = ar.pipeline(frame, [("drop_duplicates",)], dry_run=True)

        df = ar.to_pandas(result)
        assert len(df) == 3  # Duplicates not dropped because dry_run=True

    def test_dry_run_validates_step_names(self, tmp_path):
        """Test that dry_run still validates step names before execution"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step_xyz",)], dry_run=True)


class TestPipelineLineage:
    """Tests for track_lineage=True"""

    def test_lineage_tracking_drop_nulls(self, tmp_path):
        """Test that lineage correctly tracks which rows were dropped by drop_nulls"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,val\n1,10\n2,\n3,30\n")

        frame = ar.read_csv(str(csv_path))
        clean, lineage = ar.pipeline(
            frame, [("drop_nulls", {"subset": ["val"]})], track_lineage=True
        )

        assert isinstance(lineage.dropped_by_step, dict)
        assert "drop_nulls" in lineage.dropped_by_step
        assert len(lineage.dropped_by_step["drop_nulls"]) == 1
        assert lineage.total_dropped == 1

    def test_lineage_tracking_drop_duplicates(self, tmp_path):
        """Test that lineage correctly tracks rows dropped by drop_duplicates"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,alice\n2,alice\n3,bob\n")

        frame = ar.read_csv(str(csv_path))
        clean, lineage = ar.pipeline(
            frame,
            [("drop_duplicates", {"subset": ["name"], "keep": "first"})],
            track_lineage=True,
        )

        assert "drop_duplicates" in lineage.dropped_by_step
        assert len(lineage.dropped_by_step["drop_duplicates"]) == 1

    def test_lineage_tracking_multiple_steps(self, tmp_path):
        """Test lineage aggregates drops across multiple row-dropping steps"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,val\n1,10\n2,\n3,30\n4,30\n")

        frame = ar.read_csv(str(csv_path))
        clean, lineage = ar.pipeline(
            frame,
            [
                ("drop_nulls", {"subset": ["val"]}),
                ("drop_duplicates", {"subset": ["val"], "keep": "first"}),
            ],
            track_lineage=True,
        )

        assert len(lineage.dropped_by_step) == 2
        assert "drop_nulls" in lineage.dropped_by_step
        assert "drop_duplicates" in lineage.dropped_by_step
        assert lineage.total_dropped == 2

    def test_lineage_sentinel_column_collision(self, tmp_path):
        """Test that track_lineage raises ValueError if sentinel column already exists"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("__arnio_lineage_id__,name\n0,alice\n1,bob\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)

    def test_lineage_and_metadata_mutually_exclusive(self, tmp_path):
        """Test that track_lineage and return_metadata cannot be used together"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,val\n1,10\n2,20\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True,
            )


class TestPipelineErrors:
    """Tests for error handling and validation"""

    def test_unknown_step_raises_error(self, tmp_path):
        """Test that unknown step names raise UnknownStepError"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("unknown_step_xyz_123",)])

    def test_invalid_step_format_raises_error(self, tmp_path):
        """Test that invalid step tuple format raises ValueError"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step1", {"key": "val"}, "extra")])

    def test_non_dict_kwargs_raises_error(self, tmp_path):
        """Test that non-dict kwargs raises ValueError"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name\nalice\n")

        frame = ar.read_csv(str(csv_path))
        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("strip_whitespace", "not_a_dict")])

    def test_non_arframe_input_raises_error(self):
        """Test that passing a pandas DataFrame instead of ArFrame raises TypeError"""
        df = pd.DataFrame({"name": ["alice"]})

        with pytest.raises(TypeError, match="must be an ArFrame"):
            ar.pipeline(df, [("drop_nulls",)])


class TestPipelineWithMappingSteps:
    """Tests for mapping-based steps like rename_columns and cast_types"""

    def test_rename_columns_in_pipeline(self, tmp_path):
        """Test rename_columns step correctly renames columns in pipeline"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("first_name,last_name\nalice,smith\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(
            frame,
            [
                (
                    "rename_columns",
                    {"first_name": "fname", "last_name": "lname"},
                )
            ],
        )
        result = ar.to_pandas(clean)

        assert "fname" in result.columns
        assert "lname" in result.columns
        assert "first_name" not in result.columns

    def test_cast_types_in_pipeline(self, tmp_path):
        """Test cast_types step correctly changes dtypes in pipeline"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,85\n2,90\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("cast_types", {"id": "int64"})])
        result = ar.to_pandas(clean)

        assert result["id"].dtype == np.int64


class TestPipelineCustomSteps:
    """Tests for custom pipeline steps"""

    def test_custom_step_registration_and_execution(self, tmp_path):
        """Test registering and executing a custom pipeline step"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("value\n10\n20\n100\n")

        def remove_above_threshold(df, threshold):
            """Custom step: filter rows where value <= threshold"""
            return df[df["value"] <= threshold]

        ar.register_step("remove_above_threshold", remove_above_threshold)

        try:
            frame = ar.read_csv(str(csv_path))
            clean = ar.pipeline(
                frame, [("remove_above_threshold", {"threshold": 50})]
            )
            result = ar.to_pandas(clean)

            assert len(result) == 2
            assert result["value"].max() == 20
        finally:
            ar.reset_steps()

    def test_custom_step_with_pipeline_context(self, tmp_path):
        """Test that custom steps receive pipeline execution context"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n2\n")

        contexts_seen = []

        def track_context(df, context=None):
            """Custom step that records pipeline context if provided"""
            if context:
                contexts_seen.append(
                    {
                        "step_name": context.step_name,
                        "step_index": context.step_index,
                        "total_steps": context.total_steps,
                    }
                )
            return df

        ar.register_step("track_context", track_context)

        try:
            frame = ar.read_csv(str(csv_path))
            ar.pipeline(
                frame,
                [
                    ("track_context",),
                    ("track_context",),
                ],
            )

            assert len(contexts_seen) == 2
            assert contexts_seen[0]["step_index"] == 0
            assert contexts_seen[1]["step_index"] == 1
            assert contexts_seen[0]["total_steps"] == 2
        finally:
            ar.reset_steps()


class TestPipelineIntegration:
    """Integration tests with pandas accessor and disk I/O"""

    def test_pandas_accessor_pipeline_method(self, tmp_path):
        """Test using pandas .arnio.pipeline() accessor"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,score\n alice ,85\n bob ,90\n")

        df = pd.read_csv(str(csv_path))

        # Use pandas accessor to run pipeline
        steps = [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ]
        clean_df = df.arnio.pipeline(steps)

        assert clean_df["name"].tolist() == ["alice", "bob"]

    def test_pipeline_followed_by_profiling(self, tmp_path):
        """Test combining pipeline with data profiling"""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score,grade\n1,85,A\n2,85,A\n3,90,A\n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("drop_duplicates",)])

        report = ar.profile(clean)
        assert report.row_count == 2
        assert report.column_count == 3

    def test_pipeline_output_written_to_csv(self, tmp_path):
        """Test writing pipeline output to a CSV file"""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,age\n alice ,30\n bob ,25\n")

        output_csv = tmp_path / "output.csv"

        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        ar.write_csv(clean, str(output_csv))

        assert output_csv.exists()
        result = ar.to_pandas(ar.read_csv(str(output_csv)))
        assert result["name"].tolist() == ["alice", "bob"]

    def test_pipeline_full_workflow_csv_to_csv(self, tmp_path):
        """Test complete workflow: read CSV → pipeline → write CSV → verify"""
        input_csv = tmp_path / "messy.csv"
        input_csv.write_text(
            "customer_id,  status ,score\n"
            "101,ACTIVE,85\n"
            "102,INACTIVE,70\n"
            "103,ACTIVE,85\n"
            "104, ACTIVE , 90\n"
        )
        output_csv = tmp_path / "clean.csv"

        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
                ("drop_duplicates", {"keep": "first"}),
            ],
        )
        ar.write_csv(clean, str(output_csv))

        result = ar.to_pandas(ar.read_csv(str(output_csv)))
        assert len(result) == 3
        assert result["status"].tolist() == ["active", "inactive", "active"]
