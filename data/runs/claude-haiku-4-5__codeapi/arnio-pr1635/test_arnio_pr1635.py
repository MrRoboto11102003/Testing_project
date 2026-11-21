import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasic:
    """Test basic pipeline execution"""
    
    def test_empty_pipeline(self):
        """Test pipeline with no steps returns identical frame"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        assert result.shape == frame.shape
        pd.testing.assert_frame_equal(ar.to_pandas(result), df)
    
    def test_single_step_pipeline(self):
        """Test pipeline with single cleaning step"""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)])
        assert result.shape[0] == 2
        assert result.shape[1] == 2
    
    def test_multiple_steps_sequential(self):
        """Test pipeline applies steps sequentially"""
        df = pd.DataFrame({
            "a": [1, 1, 2, 3],
            "b": [" x ", " x ", " y ", " z "]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",)
        ])
        result_df = ar.to_pandas(result)
        assert len(result_df) <= 3


class TestReturnMetadata:
    """Test return_metadata parameter behavior"""
    
    def test_return_metadata_returns_tuple(self):
        """Test return_metadata=True returns (frame, dict)"""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        assert isinstance(result, ar.frame.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_metadata_contains_required_keys(self):
        """Test metadata dict contains expected structure"""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",), ("strip_whitespace",)],
            return_metadata=True
        )
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata
        assert len(metadata["applied_steps"]) == 2
    
    def test_return_metadata_false_returns_frame_only(self):
        """Test return_metadata=False returns only frame"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)], return_metadata=False)
        assert isinstance(result, ar.frame.ArFrame)
        assert not isinstance(result, tuple)


class TestTrackLineage:
    """Test track_lineage parameter and LineageReport"""
    
    def test_track_lineage_returns_tuple(self):
        """Test track_lineage=True returns (frame, LineageReport)"""
        df = pd.DataFrame({"a": [1, None, 3, 4], "b": list("abcd")})
        frame = ar.from_pandas(df)
        result = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.frame.ArFrame)
        assert hasattr(result[1], 'dropped_by_step')
        assert hasattr(result[1], 'total_dropped')
    
    def test_lineage_tracks_dropped_indices(self):
        """Test lineage correctly identifies dropped row indices"""
        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": list("abcde")})
        frame = ar.from_pandas(df)
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert "drop_nulls" in lineage.dropped_by_step
        assert 1 in lineage.dropped_by_step["drop_nulls"]
        assert 3 in lineage.dropped_by_step["drop_nulls"]
        assert lineage.total_dropped == 2
    
    def test_lineage_multiple_steps_merge_correctly(self):
        """Test lineage merges drops from same step name"""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": list("abcde")})
        frame = ar.from_pandas(df)
        result, lineage = ar.pipeline(
            frame,
            [("drop_duplicates",), ("drop_duplicates",)],
            track_lineage=True
        )
        assert "drop_duplicates" in lineage.dropped_by_step
        assert isinstance(lineage.dropped_by_step["drop_duplicates"], list)


class TestParameterValidation:
    """Test type checking for parameters"""
    
    def test_frame_must_be_arframe(self):
        """Test TypeError when frame is not ArFrame"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="ArFrame"):
            ar.pipeline(df, [])
    
    def test_return_metadata_must_be_bool(self):
        """Test TypeError for non-bool return_metadata"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            ar.pipeline(frame, [], return_metadata="true")
    
    def test_dry_run_must_be_bool(self):
        """Test TypeError for non-bool dry_run"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            ar.pipeline(frame, [], dry_run=1)
    
    def test_verbose_must_be_bool(self):
        """Test TypeError for non-bool verbose"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            ar.pipeline(frame, [], verbose="yes")
    
    def test_track_lineage_must_be_bool(self):
        """Test TypeError for non-bool track_lineage"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            ar.pipeline(frame, [], track_lineage=0)


class TestIncompatibleOptions:
    """Test mutually exclusive options"""
    
    def test_track_lineage_and_return_metadata_raises_valueerror(self):
        """Test ValueError when track_lineage and return_metadata both True"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True
            )


class TestStepFormats:
    """Test step tuple format validation"""
    
    def test_step_name_only_format(self):
        """Test valid single-element step tuple"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)])
        assert isinstance(result, ar.frame.ArFrame)
    
    def test_step_with_kwargs_format(self):
        """Test valid two-element step tuple with kwargs"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("select_columns", {"columns": ["a"]})])
        assert result.shape[1] == 1
    
    def test_invalid_step_format_too_many_elements(self):
        """Test ValueError for step tuple with > 2 elements"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("drop_nulls", {}, "extra")])
    
    def test_invalid_step_kwargs_not_dict(self):
        """Test ValueError when kwargs is not a dict"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError):
            ar.pipeline(frame, [("drop_nulls", "invalid")])


class TestUnknownSteps:
    """Test error handling for unknown steps"""
    
    def test_unknown_step_raises_unknownsteperror(self):
        """Test UnknownStepError for unregistered step name"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step_xyz",)])


class TestLineageSentinelColumn:
    """Test sentinel column edge cases"""
    
    def test_input_with_sentinel_column_raises_error(self):
        """Test ValueError when input frame has __arnio_lineage_id__ column"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "__arnio_lineage_id__": [0, 1, 2]
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)


class TestDryRun:
    """Test dry_run mode behavior"""
    
    def test_dry_run_mode_accepted(self):
        """Test dry_run=True is accepted without error"""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)], dry_run=True)
        assert isinstance(result, ar.frame.ArFrame)
    
    def test_dry_run_with_metadata(self):
        """Test dry_run with return_metadata shows dry_run flag"""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",)],
            dry_run=True,
            return_metadata=True
        )
        assert metadata["row_counts"][0]["dry_run"] is True


class TestVerboseMode:
    """Test verbose parameter"""
    
    def test_verbose_mode_accepted(self):
        """Test verbose=True is accepted without error"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)], verbose=True)
        assert isinstance(result, ar.frame.ArFrame)


class TestComplexPipelines:
    """Test realistic multi-step pipelines"""
    
    def test_comprehensive_pipeline(self):
        """Test pipeline with multiple diverse cleaning steps"""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", None, "  Alice  "],
            "age": [25, None, 35, 25],
            "city": [" NYC ", " LA ", " NYC ", " NYC "]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls",),
            ("drop_duplicates",)
        ])
        result_df = ar.to_pandas(result)
        assert len(result_df) <= 3
        assert result_df.shape[1] == 3
    
    def test_pipeline_preserves_column_count(self):
        """Test that select_columns step adjusts column count"""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("select_columns", {"columns": ["col1", "col3"]})
        ])
        assert result.shape == (3, 2)
