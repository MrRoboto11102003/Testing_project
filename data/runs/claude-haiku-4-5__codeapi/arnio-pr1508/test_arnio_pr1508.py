import pytest
import pandas as pd
import arnio as ar
from arnio.exceptions import UnknownStepError
from arnio.pipeline import LineageReport


@pytest.fixture
def simple_frame():
    """Create a simple test ArFrame."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
        'age': [25, 30, 25, 25, 35],
        'score': [95.5, 87.0, 92.3, 95.5, 88.1],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_with_whitespace():
    """Create a frame with whitespace issues."""
    data = {
        'name': ['  Alice  ', 'Bob', ' Charlie ', '  Alice  ', None],
        'city': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Boston'],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_with_duplicates():
    """Create a frame with duplicate rows."""
    data = {
        'id': [1, 2, 1, 3, 2],
        'value': ['a', 'b', 'a', 'c', 'b'],
    }
    return ar.from_pandas(pd.DataFrame(data))


class TestPipelineBasics:
    """Test basic pipeline execution."""

    def test_pipeline_single_step(self, simple_frame):
        """Test pipeline with a single step."""
        result = ar.pipeline(simple_frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 4

    def test_pipeline_multiple_steps(self, frame_with_whitespace):
        """Test pipeline with multiple sequential steps."""
        steps = [("strip_whitespace",), ("drop_nulls",)]
        result = ar.pipeline(frame_with_whitespace, steps)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 4

    def test_pipeline_empty_steps(self, simple_frame):
        """Test pipeline with no steps returns identical frame."""
        result = ar.pipeline(simple_frame, [])
        assert result.shape == simple_frame.shape

    def test_pipeline_step_with_kwargs(self, simple_frame):
        """Test pipeline step accepts keyword arguments."""
        steps = [("drop_nulls", {"subset": ["name"]})]
        result = ar.pipeline(simple_frame, steps)
        assert result.shape[0] == 4


class TestPipelineReturnFormats:
    """Test different return formats based on flags."""

    def test_pipeline_default_return_arframe(self, simple_frame):
        """Test default return is just ArFrame."""
        result = ar.pipeline(simple_frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)

    def test_pipeline_return_metadata(self, simple_frame):
        """Test return_metadata=True returns tuple with metadata dict."""
        result, metadata = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert len(metadata["applied_steps"]) == 1

    def test_pipeline_metadata_has_timing_data(self, simple_frame):
        """Test metadata includes per-step timing information."""
        _, metadata = ar.pipeline(
            simple_frame,
            [("drop_nulls",), ("strip_whitespace",)],
            return_metadata=True
        )
        assert len(metadata["step_timings"]) == 2
        assert all("seconds" in t for t in metadata["step_timings"])
        assert all("step" in t for t in metadata["step_timings"])

    def test_pipeline_track_lineage(self, simple_frame):
        """Test track_lineage=True returns tuple with LineageReport."""
        result, lineage = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert isinstance(result, ar.ArFrame)
        assert isinstance(lineage, LineageReport)
        assert hasattr(lineage, 'dropped_by_step')
        assert hasattr(lineage, 'total_dropped')

    def test_pipeline_lineage_tracks_dropped_rows(self, simple_frame):
        """Test lineage report tracks which rows were dropped."""
        result, lineage = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert "drop_nulls" in lineage.dropped_by_step
        assert len(lineage.dropped_by_step["drop_nulls"]) == 1
        assert lineage.total_dropped == 1

    def test_pipeline_lineage_removes_sentinel_column(self, simple_frame):
        """Test lineage tracking removes internal sentinel before return."""
        result, lineage = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        # Sentinel column should not be in returned frame
        assert "__arnio_lineage_id__" not in result.columns


class TestPipelineFlags:
    """Test behavior of boolean flags."""

    def test_pipeline_dry_run_preserves_row_count(self, simple_frame):
        """Test dry_run=True validates without modifying."""
        original_rows = simple_frame.shape[0]
        result = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            dry_run=True
        )
        assert result.shape[0] == original_rows

    def test_pipeline_dry_run_with_multiple_steps(self, frame_with_whitespace):
        """Test dry_run validates all steps without applying."""
        original_rows = frame_with_whitespace.shape[0]
        result = ar.pipeline(
            frame_with_whitespace,
            [("strip_whitespace",), ("drop_nulls",)],
            dry_run=True
        )
        assert result.shape[0] == original_rows

    def test_pipeline_track_lineage_and_return_metadata_conflict(self, simple_frame):
        """Test that track_lineage and return_metadata cannot be combined."""
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                simple_frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True
            )


class TestPipelineTypeValidation:
    """Test parameter type validation."""

    def test_pipeline_frame_must_be_arframe(self):
        """Test pipeline rejects non-ArFrame input."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])

    def test_pipeline_return_metadata_type_check(self, simple_frame):
        """Test return_metadata must be boolean."""
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(simple_frame, [], return_metadata="yes")

    def test_pipeline_dry_run_type_check(self, simple_frame):
        """Test dry_run must be boolean."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(simple_frame, [], dry_run=1)

    def test_pipeline_verbose_type_check(self, simple_frame):
        """Test verbose must be boolean."""
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(simple_frame, [], verbose="yes")

    def test_pipeline_track_lineage_type_check(self, simple_frame):
        """Test track_lineage must be boolean."""
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(simple_frame, [], track_lineage=1)


class TestPipelineStepValidation:
    """Test step format and name validation."""

    def test_pipeline_unknown_step_raises_error(self, simple_frame):
        """Test unknown step name raises UnknownStepError."""
        with pytest.raises(UnknownStepError):
            ar.pipeline(simple_frame, [("nonexistent_step",)])

    def test_pipeline_invalid_step_tuple_format(self, simple_frame):
        """Test invalid step format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(simple_frame, [("drop_nulls", {}, "extra")])

    def test_pipeline_invalid_step_kwargs_not_dict(self, simple_frame):
        """Test step kwargs must be dict if present."""
        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(simple_frame, [("drop_nulls", "invalid")])

    def test_pipeline_step_without_kwargs(self, simple_frame):
        """Test step with no kwargs as single-element tuple."""
        result = ar.pipeline(simple_frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)


class TestPipelineLineageTracking:
    """Test lineage tracking specific functionality."""

    def test_pipeline_lineage_requires_no_sentinel_column(self, simple_frame):
        """Test track_lineage fails if input has sentinel column."""
        df = ar.to_pandas(simple_frame)
        df['__arnio_lineage_id__'] = range(len(df))
        frame_with_sentinel = ar.from_pandas(df)

        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame_with_sentinel, [], track_lineage=True)

    def test_pipeline_lineage_multiple_same_step(self):
        """Test lineage merges drops when same step appears multiple times."""
        data = {
            'a': [1, 2, None, 4, 5, 5],
            'b': ['x', 'y', 'z', 'w', 'v', 'v'],
        }
        frame = ar.from_pandas(pd.DataFrame(data))

        result, lineage = ar.pipeline(
            frame,
            [
                ("drop_nulls",),
                ("drop_duplicates",),
            ],
            track_lineage=True
        )
        assert lineage.total_dropped > 0

    def test_pipeline_lineage_no_drops_empty_report(self, simple_frame):
        """Test lineage report when no rows are dropped."""
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        frame = ar.from_pandas(pd.DataFrame(data))

        result, lineage = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            track_lineage=True
        )
        assert lineage.total_dropped == 0


class TestPipelineIntegration:
    """Test real-world pipeline scenarios."""

    def test_pipeline_data_cleaning_workflow(self):
        """Test typical data cleaning pipeline."""
        data = {
            'name': ['  Alice  ', '  Bob  ', '   ', 'Bob', None],
            'age': [25, 30, 35, 30, 40],
            'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                     'bob@example.com', 'dave@example.com'],
        }
        frame = ar.from_pandas(pd.DataFrame(data))

        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls",),
            ("drop_duplicates",),
        ])
        assert result.shape[0] <= frame.shape[0]
        assert result.shape[1] == frame.shape[1]

    def test_pipeline_with_subset_kwargs(self, simple_frame):
        """Test pipeline with subset-based operations."""
        result = ar.pipeline(simple_frame, [
            ("drop_nulls", {"subset": ["name"]}),
        ])
        assert result.shape[0] == 4

    def test_pipeline_verbose_executes_successfully(self, simple_frame):
        """Test verbose flag doesn't interfere with execution."""
        result = ar.pipeline(
            simple_frame,
            [("drop_nulls",)],
            verbose=True
        )
        assert result.shape[0] == 4
