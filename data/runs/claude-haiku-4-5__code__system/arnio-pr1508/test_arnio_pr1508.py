import pytest
import pandas as pd
import tempfile
from pathlib import Path

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestPipelineBasic:
    """End-to-end tests for arnio.pipeline.pipeline function."""

    def test_pipeline_simple_drop_nulls(self, tmp_path):
        """Test basic pipeline with single drop_nulls step."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", None, "David"],
            "age": [25, 30, 35, None]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("drop_nulls",)])

        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)
        assert len(result_df) == 2
        assert result_df["name"].tolist() == ["Alice", "Bob"]

    def test_pipeline_multiple_steps(self, tmp_path):
        """Test pipeline with multiple different steps."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "name": ["  alice  ", "  bob  ", "  alice  "],
            "value": [1, 2, 1]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)
        assert len(result_df) == 2
        assert set(result_df["name"].tolist()) == {"alice", "bob"}

    def test_pipeline_with_step_kwargs(self, tmp_path):
        """Test pipeline with step kwargs."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "value": [10, 20, 20, 30]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_duplicates", {"subset": ["id"]})
        ])

        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)
        assert len(result_df) == 3
        assert sorted(result_df["id"].tolist()) == [1, 2, 3]


class TestPipelineMetadata:
    """Tests for return_metadata functionality."""

    def test_pipeline_return_metadata_structure(self, tmp_path):
        """Test return_metadata returns tuple with metadata dict."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, None],
            "y": [4, 5, 6]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            return_metadata=True
        )

        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert metadata["applied_steps"] == ["drop_nulls", "drop_duplicates"]
        assert len(metadata["row_counts"]) == 2
        assert len(metadata["step_timings"]) == 2
        for timing in metadata["step_timings"]:
            assert "step" in timing
            assert "seconds" in timing
            assert "dry_run" in timing

    def test_pipeline_metadata_execution_summary(self, tmp_path):
        """Test metadata includes execution summary."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",)],
            return_metadata=True
        )

        assert "execution_summary" in metadata
        assert "total_runtime_ms" in metadata["execution_summary"]
        assert "steps" in metadata["execution_summary"]


class TestPipelineDryRun:
    """Tests for dry_run functionality."""

    def test_pipeline_dry_run_validates_steps(self, tmp_path):
        """Test dry_run validates pipeline structure without error."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [None, 5, 6]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            dry_run=True
        )

        assert result is not None

    def test_pipeline_dry_run_with_metadata(self, tmp_path):
        """Test dry_run with return_metadata shows dry_run flag."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1, 2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",)],
            dry_run=True,
            return_metadata=True
        )

        assert metadata["row_counts"][0]["dry_run"] is True
        assert metadata["step_timings"][0]["dry_run"] is True


class TestPipelineLineage:
    """Tests for track_lineage functionality."""

    def test_pipeline_track_lineage_basic(self, tmp_path):
        """Test track_lineage returns LineageReport tuple."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, 3, None],
            "y": [10, 20, 30, 40]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )

        assert hasattr(lineage, 'dropped_by_step')
        assert hasattr(lineage, 'total_dropped')
        assert isinstance(lineage.dropped_by_step, dict)
        assert "drop_nulls" in lineage.dropped_by_step

    def test_pipeline_track_lineage_records_drops(self, tmp_path):
        """Test lineage correctly identifies dropped rows."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [None, 2, None, 4]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )

        assert lineage.total_dropped > 0
        dropped_indices = lineage.dropped_by_step["drop_nulls"]
        assert len(dropped_indices) > 0
        assert dropped_indices == sorted(dropped_indices)

    def test_pipeline_track_lineage_multiple_same_steps(self, tmp_path):
        """Test lineage merges drops from repeated step names."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, 2, 3, 3],
            "y": [1, 1, 1, 1, 1]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result, lineage = ar.pipeline(
            frame,
            [
                ("drop_duplicates", {"subset": ["x"]}),
                ("drop_duplicates", {"subset": ["y"]}),
            ],
            track_lineage=True
        )

        # Both invocations merge under single "drop_duplicates" key
        assert "drop_duplicates" in lineage.dropped_by_step
        merged_list = lineage.dropped_by_step["drop_duplicates"]
        assert merged_list == sorted(merged_list)


class TestPipelineErrors:
    """Tests for error handling and validation."""

    def test_pipeline_error_invalid_frame_type(self):
        """Test TypeError when frame is not ArFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [("drop_nulls",)])

    def test_pipeline_error_invalid_return_metadata_type(self, tmp_path):
        """Test TypeError for non-bool return_metadata."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [("drop_nulls",)], return_metadata="yes")

    def test_pipeline_error_invalid_dry_run_type(self, tmp_path):
        """Test TypeError for non-bool dry_run."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [("drop_nulls",)], dry_run=1)

    def test_pipeline_error_invalid_verbose_type(self, tmp_path):
        """Test TypeError for non-bool verbose."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [("drop_nulls",)], verbose=1)

    def test_pipeline_error_invalid_track_lineage_type(self, tmp_path):
        """Test TypeError for non-bool track_lineage."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage="yes")

    def test_pipeline_error_track_lineage_return_metadata_conflict(self, tmp_path):
        """Test ValueError when both track_lineage and return_metadata are True."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True
            )

    def test_pipeline_error_unknown_step(self, tmp_path):
        """Test error for unknown step name."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(Exception):  # UnknownStepError
            ar.pipeline(frame, [("nonexistent_step_xyz_123",)])

    def test_pipeline_error_invalid_step_format_too_many_elements(self, tmp_path):
        """Test ValueError for invalid step format with too many elements."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("drop_nulls", {}, "extra")])

    def test_pipeline_error_invalid_step_kwargs_type(self, tmp_path):
        """Test ValueError when step kwargs is not a dict."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv_file, index=False)
        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("drop_nulls", "not_a_dict")])

    def test_pipeline_error_empty_steps_list_valid(self, tmp_path):
        """Test that empty steps list is valid."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [])

        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)
        assert len(result_df) == 3
