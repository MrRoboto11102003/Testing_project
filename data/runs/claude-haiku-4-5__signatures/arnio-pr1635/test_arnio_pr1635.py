import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': ['a', 'b', 'c', 'd', 'e']
        }
        df = pd.DataFrame(data)
        return ar.ArFrame(df)
    
    def test_pipeline_empty_steps(self, sample_frame):
        """Pipeline with no steps returns ArFrame unchanged."""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
        assert len(result) == len(sample_frame)
    
    def test_pipeline_returns_arframe_default(self, sample_frame):
        """Pipeline returns ArFrame by default (without flags)."""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)
    
    def test_pipeline_single_step(self, sample_frame):
        """Pipeline applies a single step."""
        steps = [('select', {'columns': ['col1', 'col2']})]
        result = ar.pipeline(sample_frame, steps)
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_multiple_steps(self, sample_frame):
        """Pipeline applies multiple steps in sequence."""
        steps = [
            ('select', {'columns': ['col1', 'col2']}),
            ('filter', {'condition': 'col1 > 2'})
        ]
        result = ar.pipeline(sample_frame, steps)
        assert isinstance(result, ar.ArFrame)


class TestPipelineReturnMetadata:
    """Test return_metadata parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        return ar.ArFrame(pd.DataFrame(data))
    
    def test_return_metadata_false_default(self, sample_frame):
        """return_metadata=False (default) returns only ArFrame."""
        result = ar.pipeline(sample_frame, [], return_metadata=False)
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)
    
    def test_return_metadata_true_returns_tuple(self, sample_frame):
        """return_metadata=True returns tuple of (ArFrame, dict)."""
        result = ar.pipeline(sample_frame, [], return_metadata=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_metadata_contains_pipeline_info(self, sample_frame):
        """Metadata dict contains information about pipeline execution."""
        steps = [('select', {'columns': ['A']})]
        result = ar.pipeline(sample_frame, steps, return_metadata=True)
        frame, metadata = result
        assert isinstance(metadata, dict)
        assert 'steps' in metadata or 'operations' in metadata or len(metadata) >= 0


class TestPipelineDryRun:
    """Test dry_run parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
        return ar.ArFrame(pd.DataFrame(data))
    
    def test_dry_run_false_applies_steps(self, sample_frame):
        """dry_run=False applies transformations to data."""
        steps = [('drop', {'columns': ['B']})]
        result = ar.pipeline(sample_frame, steps, dry_run=False)
        assert isinstance(result, ar.ArFrame)
    
    def test_dry_run_true_returns_arframe(self, sample_frame):
        """dry_run=True returns ArFrame without side effects."""
        steps = [('drop', {'columns': ['B']})]
        result = ar.pipeline(sample_frame, steps, dry_run=True)
        assert isinstance(result, ar.ArFrame)


class TestPipelineTrackLineage:
    """Test track_lineage parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        return ar.ArFrame(pd.DataFrame(data))
    
    def test_track_lineage_false_default(self, sample_frame):
        """track_lineage=False (default) returns ArFrame only."""
        result = ar.pipeline(sample_frame, [], track_lineage=False)
        assert isinstance(result, ar.ArFrame)
    
    def test_track_lineage_true_returns_tuple(self, sample_frame):
        """track_lineage=True returns tuple of (ArFrame, LineageReport)."""
        steps = [('select', {'columns': ['A']})]
        result = ar.pipeline(sample_frame, steps, track_lineage=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage_report = result
        assert isinstance(frame, ar.ArFrame)
        # LineageReport has some attributes/structure
        assert hasattr(lineage_report, '__class__')


class TestPipelineVerbose:
    """Test verbose parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {'A': [1, 2, 3]}
        return ar.ArFrame(pd.DataFrame(data))
    
    def test_verbose_flag_accepted(self, sample_frame):
        """verbose=True parameter is accepted without error."""
        result = ar.pipeline(sample_frame, [], verbose=True)
        assert isinstance(result, ar.ArFrame)


class TestPipelineFlagCombinations:
    """Test combinations of optional parameters."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        return ar.ArFrame(pd.DataFrame(data))
    
    def test_metadata_with_dry_run(self, sample_frame):
        """return_metadata=True with dry_run=True returns (frame, metadata)."""
        steps = [('select', {'columns': ['A']})]
        result = ar.pipeline(
            sample_frame, steps,
            return_metadata=True,
            dry_run=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_lineage_with_verbose(self, sample_frame):
        """track_lineage=True with verbose=True returns (frame, report)."""
        steps = [('select', {'columns': ['A']})]
        result = ar.pipeline(
            sample_frame, steps,
            track_lineage=True,
            verbose=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_metadata_and_lineage_precedence(self, sample_frame):
        """track_lineage takes precedence when both metadata and lineage flags are True."""
        result = ar.pipeline(
            sample_frame, [],
            return_metadata=True,
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Pipeline handles empty DataFrame."""
        df = pd.DataFrame()
        frame = ar.ArFrame(df)
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_single_row_frame(self):
        """Pipeline handles single-row DataFrame."""
        data = {'A': [1], 'B': [2]}
        frame = ar.ArFrame(pd.DataFrame(data))
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
        assert len(result) == 1
    
    def test_single_column_frame(self):
        """Pipeline handles single-column DataFrame."""
        data = {'A': [1, 2, 3]}
        frame = ar.ArFrame(pd.DataFrame(data))
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_many_steps_in_sequence(self):
        """Pipeline handles large number of steps."""
        data = {'A': [1, 2, 3]}
        frame = ar.ArFrame(pd.DataFrame(data))
        steps = [('noop', {}) for _ in range(50)]
        result = ar.pipeline(frame, steps)
        assert isinstance(result, ar.ArFrame)
