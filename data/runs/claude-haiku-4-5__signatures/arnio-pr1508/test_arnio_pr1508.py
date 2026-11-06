import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def sample_frame():
    """Fixture providing a sample dataframe for tests"""
    return pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        'c': ['x', 'y', 'z', 'w', 'v']
    })


class TestPipelineBasicExecution:
    """Test basic pipeline execution"""
    
    def test_pipeline_empty_steps(self, sample_frame):
        """Pipeline with empty steps list returns frame unchanged"""
        result = ar.pipeline.pipeline(sample_frame, [])
        assert isinstance(result, (pd.DataFrame, ar.ArFrame))
        pd.testing.assert_frame_equal(result, sample_frame, check_dtype=False)
    
    def test_pipeline_returns_frame_by_default(self, sample_frame):
        """Default return without flags is just the frame"""
        result = ar.pipeline.pipeline(sample_frame, [])
        assert not isinstance(result, tuple)
        assert isinstance(result, (pd.DataFrame, ar.ArFrame))


class TestPipelineReturnMetadata:
    """Test return_metadata flag behavior"""
    
    def test_return_metadata_false(self, sample_frame):
        """return_metadata=False returns single frame, not tuple"""
        result = ar.pipeline.pipeline(sample_frame, [], return_metadata=False)
        assert not isinstance(result, tuple)
        assert isinstance(result, (pd.DataFrame, ar.ArFrame))
    
    def test_return_metadata_true(self, sample_frame):
        """return_metadata=True returns (frame, metadata_dict)"""
        result = ar.pipeline.pipeline(sample_frame, [], return_metadata=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, (pd.DataFrame, ar.ArFrame))
        assert isinstance(metadata, dict)


class TestPipelineLineageTracking:
    """Test track_lineage flag behavior"""
    
    def test_track_lineage_false(self, sample_frame):
        """track_lineage=False returns frame or metadata tuple"""
        result = ar.pipeline.pipeline(sample_frame, [], track_lineage=False)
        if isinstance(result, tuple):
            assert len(result) == 2
            assert isinstance(result[1], dict)
    
    def test_track_lineage_true(self, sample_frame):
        """track_lineage=True returns (frame, LineageReport)"""
        result = ar.pipeline.pipeline(sample_frame, [], track_lineage=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage_report = result
        assert isinstance(frame, (pd.DataFrame, ar.ArFrame))
        assert lineage_report is not None


class TestPipelineFlags:
    """Test optional flags"""
    
    def test_dry_run_true(self, sample_frame):
        """dry_run=True should be accepted and not raise"""
        result = ar.pipeline.pipeline(sample_frame, [], dry_run=True)
        assert result is not None
    
    def test_verbose_true(self, sample_frame):
        """verbose=True should be accepted and not raise"""
        result = ar.pipeline.pipeline(sample_frame, [], verbose=True)
        assert result is not None
    
    def test_combined_flags(self, sample_frame):
        """Multiple flags should work together"""
        result = ar.pipeline.pipeline(
            sample_frame, [],
            return_metadata=True,
            dry_run=False,
            verbose=False,
            track_lineage=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPipelineEdgeCases:
    """Test edge cases and special data scenarios"""
    
    def test_empty_dataframe(self):
        """Pipeline should handle empty dataframe"""
        empty_df = pd.DataFrame()
        result = ar.pipeline.pipeline(empty_df, [])
        assert isinstance(result, (pd.DataFrame, ar.ArFrame))
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Pipeline should handle single-row dataframe"""
        single_df = pd.DataFrame({'x': [42], 'y': [3.14]})
        result = ar.pipeline.pipeline(single_df, [])
        assert len(result) == 1
    
    def test_dataframe_with_nan_values(self):
        """Pipeline should preserve NaN values"""
        nan_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.nan, 5.0, 6.0]
        })
        result = ar.pipeline.pipeline(nan_df, [])
        assert isinstance(result, (pd.DataFrame, ar.ArFrame))
        assert result.isna().sum().sum() == 2
    
    def test_mixed_dtypes(self):
        """Pipeline should preserve mixed data types"""
        mixed_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        result = ar.pipeline.pipeline(mixed_df, [])
        assert len(result.columns) == 4
        assert list(result.columns) == ['int_col', 'float_col', 'str_col', 'bool_col']


class TestPipelineInputValidation:
    """Test input validation and error handling"""
    
    def test_none_frame_raises(self):
        """None as frame should raise TypeError or AttributeError"""
        with pytest.raises((TypeError, AttributeError, ValueError)):
            ar.pipeline.pipeline(None, [])
    
    def test_none_steps_raises(self):
        """None as steps should raise TypeError or ValueError"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises((TypeError, ValueError)):
            ar.pipeline.pipeline(df, None)
    
    def test_invalid_steps_type_raises(self):
        """Non-list steps should raise TypeError or ValueError"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises((TypeError, ValueError)):
            ar.pipeline.pipeline(df, "not_a_list")
    
    def test_invalid_steps_type_dict_raises(self):
        """Dict as steps should raise TypeError or ValueError"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises((TypeError, ValueError)):
            ar.pipeline.pipeline(df, {'step': 'value'})
