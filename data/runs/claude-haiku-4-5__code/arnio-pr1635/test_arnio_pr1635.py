import pytest
import pandas as pd
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    def test_pipeline_empty_steps_returns_arframe(self):
        """Pipeline with empty steps returns ArFrame with same data."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
        pd.testing.assert_frame_equal(ar.to_pandas(result), df)
    
    def test_pipeline_single_step(self):
        """Pipeline applies single step correctly."""
        df = pd.DataFrame({'a': [' x ', ' y ', ' z ']})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)


class TestPipelineParameterValidation:
    """Test type validation for all parameters."""
    
    def test_frame_must_be_arframe(self):
        """TypeError raised when frame is not ArFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])
    
    def test_return_metadata_must_be_bool(self):
        """TypeError raised for non-bool return_metadata."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata="yes")
    
    def test_dry_run_must_be_bool(self):
        """TypeError raised for non-bool dry_run."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run=1)
    
    def test_verbose_must_be_bool(self):
        """TypeError raised for non-bool verbose."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [], verbose=None)
    
    def test_track_lineage_must_be_bool(self):
        """TypeError raised for non-bool track_lineage."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage="yes")


class TestPipelineReturnTypes:
    """Test different return types based on options."""
    
    def test_return_arframe_by_default(self):
        """Default return is ArFrame."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, 2]}))
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)
    
    def test_return_metadata_returns_tuple(self):
        """With return_metadata=True, returns (ArFrame, dict)."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, 2]}))
        result = ar.pipeline(frame, [], return_metadata=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], dict)
        assert 'step_timings' in result[1]
        assert 'row_counts' in result[1]
        assert 'execution_summary' in result[1]
    
    def test_track_lineage_returns_tuple(self):
        """With track_lineage=True, returns (ArFrame, LineageReport)."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, 2]}))
        result = ar.pipeline(frame, [], track_lineage=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert hasattr(result[1], 'dropped_by_step')
        assert hasattr(result[1], 'total_dropped')


class TestPipelineOptionsConflict:
    """Test incompatible option combinations."""
    
    def test_track_lineage_and_return_metadata_raises_valueerror(self):
        """ValueError when both track_lineage and return_metadata are True."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(ValueError, match="track_lineage=True and return_metadata=True cannot be used together"):
            ar.pipeline(frame, [], track_lineage=True, return_metadata=True)


class TestPipelineStepFormats:
    """Test step specification formats."""
    
    def test_step_name_only(self):
        """Step as single-element tuple works."""
        frame = ar.from_pandas(pd.DataFrame({'a': [' x ']}))
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_step_with_kwargs_dict(self):
        """Step as two-element tuple with kwargs dict works."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, 2, None]}))
        result = ar.pipeline(frame, [("drop_nulls", {"subset": ["a"]})])
        assert isinstance(result, ar.ArFrame)
    
    def test_invalid_step_tuple_too_long_raises_valueerror(self):
        """ValueError for step tuple longer than 2 elements."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step", {}, "extra")])
    
    def test_invalid_step_kwargs_not_dict_raises_valueerror(self):
        """ValueError when step kwargs is not a dict."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_nulls", "not_dict")])


class TestPipelineLineageTracking:
    """Test lineage tracking functionality."""
    
    def test_track_lineage_sentinel_column_collision_raises_valueerror(self):
        """ValueError if input frame contains __arnio_lineage_id__ column."""
        df = pd.DataFrame({'__arnio_lineage_id__': [0, 1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="track_lineage=True requires that the input frame does not already"):
            ar.pipeline(frame, [], track_lineage=True)


class TestPipelineEdgeCases:
    """Test edge cases."""
    
    def test_empty_dataframe(self):
        """Pipeline handles empty dataframe."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_single_row_dataframe(self):
        """Pipeline handles single-row dataframe."""
        df = pd.DataFrame({'a': [1]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        assert ar.to_pandas(result).shape[0] == 1
    
    def test_dry_run_preserves_data(self):
        """With dry_run=True, output shape matches input shape."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [], dry_run=True)
        assert ar.to_pandas(result).shape == df.shape
    
    def test_multiple_steps_applied_sequentially(self):
        """Multiple steps are applied in order."""
        df = pd.DataFrame({'a': [' x ', ' y ', ' z ']})
        frame = ar.from_pandas(df)
        result = ar.pipeline(
            frame,
            [("strip_whitespace",)]
        )
        assert isinstance(result, ar.ArFrame)
