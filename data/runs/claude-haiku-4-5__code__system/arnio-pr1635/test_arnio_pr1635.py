import sys
import pytest
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
import logging

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


@pytest.fixture
def simple_csv(tmp_path):
    """Create a simple CSV with nulls for testing."""
    csv_file = tmp_path / "simple.csv"
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 35, 40, None],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston']
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def duplicates_csv(tmp_path):
    """Create a CSV with duplicate rows."""
    csv_file = tmp_path / "duplicates.csv"
    df = pd.DataFrame({
        'id': [1, 2, 2, 3, 3, 3],
        'value': ['a', 'b', 'b', 'c', 'c', 'c']
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def all_nulls_csv(tmp_path):
    """Create a CSV that will drop to zero rows."""
    csv_file = tmp_path / "all_nulls.csv"
    df = pd.DataFrame({'value': [None, None, None]})
    df.to_csv(csv_file, index=False)
    return csv_file


class TestBasicPipelineExecution:
    """Test basic pipeline execution with real CSV files."""
    
    def test_pipeline_single_step(self, simple_csv, tmp_path):
        """Test pipeline with one drop_nulls step."""
        frame = ar.read_csv(simple_csv)
        result = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]})
        ])
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 4
        assert result_df['name'].isna().sum() == 0
        
        # Write result to file
        output_file = tmp_path / "output.csv"
        result_df.to_csv(output_file, index=False)
        assert output_file.exists()
    
    def test_pipeline_multiple_sequential_steps(self, simple_csv):
        """Test pipeline with multiple steps applied in order."""
        frame = ar.read_csv(simple_csv)
        result = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]}),
            ("drop_nulls", {"subset": ["age"]})
        ])
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3
        assert result_df['name'].isna().sum() == 0
        assert result_df['age'].isna().sum() == 0
    
    def test_pipeline_drop_duplicates(self, duplicates_csv):
        """Test pipeline with drop_duplicates step."""
        frame = ar.read_csv(duplicates_csv)
        result = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"})
        ])
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3


class TestLineageTracking:
    """Test row-level lineage tracking functionality."""
    
    def test_lineage_single_step(self, simple_csv):
        """Test lineage tracking with one step."""
        frame = ar.read_csv(simple_csv)
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]})
        ], track_lineage=True)
        
        assert hasattr(lineage, 'dropped_by_step')
        assert hasattr(lineage, 'total_dropped')
        assert 'drop_nulls' in lineage.dropped_by_step
        assert lineage.total_dropped == 1
        assert lineage.dropped_by_step['drop_nulls'] == sorted(lineage.dropped_by_step['drop_nulls'])
    
    def test_lineage_multiple_steps_merged(self, simple_csv):
        """Test that repeated step names merge their dropped indices."""
        frame = ar.read_csv(simple_csv)
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]}),
            ("drop_nulls", {"subset": ["age"]})
        ], track_lineage=True)
        
        assert 'drop_nulls' in lineage.dropped_by_step
        # Both nulls should be merged under one key
        assert len(lineage.dropped_by_step['drop_nulls']) == 2
        assert lineage.dropped_by_step['drop_nulls'] == sorted(lineage.dropped_by_step['drop_nulls'])
    
    def test_lineage_conflicts_with_metadata(self, simple_csv):
        """Test that track_lineage and return_metadata are mutually exclusive."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(ValueError, match="track_lineage=True and return_metadata=True"):
            ar.pipeline(frame, [("drop_nulls",)], 
                       track_lineage=True, return_metadata=True)
    
    def test_lineage_rejects_sentinel_column_in_input(self, tmp_path):
        """Test that lineage rejects input with __arnio_lineage_id__ column."""
        csv_file = tmp_path / "conflict.csv"
        df = pd.DataFrame({
            '__arnio_lineage_id__': [0, 1, 2],
            'data': ['x', 'y', 'z']
        })
        df.to_csv(csv_file, index=False)
        frame = ar.read_csv(csv_file)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)
    
    def test_lineage_no_drops(self, tmp_path):
        """Test lineage when no rows are dropped."""
        csv_file = tmp_path / "no_drops.csv"
        df = pd.DataFrame({'val': [1, 2, 3]})
        df.to_csv(csv_file, index=False)
        frame = ar.read_csv(csv_file)
        
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["val"]})
        ], track_lineage=True)
        
        assert lineage.total_dropped == 0


class TestMetadataReturn:
    """Test return_metadata parameter."""
    
    def test_metadata_timing_and_counts(self, simple_csv):
        """Test that metadata returns timing and row count info."""
        frame = ar.read_csv(simple_csv)
        result, metadata = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]})
        ], return_metadata=True)
        
        assert isinstance(metadata, dict)
        assert 'applied_steps' in metadata
        assert 'row_counts' in metadata
        assert 'step_timings' in metadata
        assert metadata['applied_steps'] == ['drop_nulls']
        
        timing = metadata['step_timings'][0]
        assert 'step' in timing
        assert 'seconds' in timing
        assert timing['seconds'] >= 0
        
        counts = metadata['row_counts'][0]
        assert counts['before'] == 5
        assert counts['after'] == 4
    
    def test_metadata_multiple_steps(self, simple_csv):
        """Test metadata accumulation across multiple steps."""
        frame = ar.read_csv(simple_csv)
        result, metadata = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]}),
            ("drop_nulls", {"subset": ["age"]})
        ], return_metadata=True)
        
        assert len(metadata['applied_steps']) == 2
        assert metadata['applied_steps'] == ['drop_nulls', 'drop_nulls']
        assert len(metadata['row_counts']) == 2
        # Row counts show progression: 5 -> 4 -> 3
        assert metadata['row_counts'][0]['before'] == 5
        assert metadata['row_counts'][0]['after'] == 4
        assert metadata['row_counts'][1]['before'] == 4
        assert metadata['row_counts'][1]['after'] == 3


class TestDryRun:
    """Test dry_run parameter."""
    
    def test_dry_run_no_modification(self, simple_csv):
        """Test that dry_run validates without modifying output."""
        frame = ar.read_csv(simple_csv)
        original_len = len(ar.to_pandas(frame))
        
        result = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]})
        ], dry_run=True)
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == original_len
    
    def test_dry_run_with_metadata_shows_zero_impact(self, simple_csv):
        """Test dry_run with return_metadata shows rows_affected as 0."""
        frame = ar.read_csv(simple_csv)
        result, metadata = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["name"]})
        ], dry_run=True, return_metadata=True)
        
        step_diag = metadata['execution_summary']['steps'][0]
        assert step_diag['rows_affected'] == 0
        assert step_diag['dry_run'] is True


class TestParameterValidation:
    """Test parameter type validation."""
    
    def test_invalid_frame_type(self):
        """Test that non-ArFrame raises TypeError."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])
    
    def test_invalid_return_metadata_type(self, simple_csv):
        """Test that invalid return_metadata type raises TypeError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata="yes")
    
    def test_invalid_dry_run_type(self, simple_csv):
        """Test that invalid dry_run type raises TypeError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run=1)
    
    def test_invalid_verbose_type(self, simple_csv):
        """Test that invalid verbose type raises TypeError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [], verbose="true")
    
    def test_invalid_track_lineage_type(self, simple_csv):
        """Test that invalid track_lineage type raises TypeError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=1)


class TestStepFormatValidation:
    """Test step tuple format validation."""
    
    def test_invalid_step_tuple_too_long(self, simple_csv):
        """Test that step tuples with >2 elements raise ValueError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step", {}, "extra")])
    
    def test_invalid_step_kwargs_not_dict(self, simple_csv):
        """Test that non-dict kwargs raise ValueError."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_nulls", "not_a_dict")])


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unknown_step_raises_error(self, simple_csv):
        """Test that unknown step name raises an error."""
        frame = ar.read_csv(simple_csv)
        with pytest.raises(Exception):  # UnknownStepError or KeyError
            ar.pipeline(frame, [("nonexistent_step_xyz",)])
    
    def test_pipeline_all_rows_dropped(self, all_nulls_csv):
        """Test pipeline where all rows are dropped."""
        frame = ar.read_csv(all_nulls_csv)
        result = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["value"]})
        ])
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
    
    def test_empty_pipeline_step_list(self, simple_csv):
        """Test pipeline with empty step list."""
        frame = ar.read_csv(simple_csv)
        original_len = len(ar.to_pandas(frame))
        
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        assert len(result_df) == original_len
    
    def test_verbose_logging_executes(self, simple_csv, caplog):
        """Test that verbose mode enables logging without error."""
        frame = ar.read_csv(simple_csv)
        with caplog.at_level(logging.INFO):
            result = ar.pipeline(frame, [
                ("drop_nulls", {"subset": ["name"]})
            ], verbose=True)
        
        result_df = ar.to_pandas(result)
        assert len(result_df) == 4
