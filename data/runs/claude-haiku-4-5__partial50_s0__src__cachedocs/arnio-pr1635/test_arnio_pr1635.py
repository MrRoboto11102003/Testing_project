import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasicExecution:
    """Test basic pipeline execution."""
    
    def test_basic_single_step(self):
        """Test pipeline with a single cleaning step."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  "],
            "age": [30, 25]
        })
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Alice", "Bob"]
        assert result_df["age"].tolist() == [30, 25]
    
    def test_multiple_sequential_steps(self):
        """Test pipeline with multiple steps executed in order."""
        df = pd.DataFrame({
            "name": ["  ALICE  ", "  BOB  ", "  ALICE  "],
            "score": [10, 20, 10]
        })
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates", {"keep": "first"}),
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert result_df["name"].tolist() == ["alice", "bob"]
    
    def test_step_with_kwargs(self):
        """Test pipeline step with keyword arguments."""
        df = pd.DataFrame({"value": [1.5, 2.7, 3.2]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("round_numeric_columns", {"decimals": 0}),
        ])
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].tolist() == [2.0, 3.0, 3.0]


class TestPipelineMetadata:
    """Test metadata return functionality."""
    
    def test_return_metadata_structure(self):
        """Test that return_metadata=True returns correct structure."""
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob  "]})
        frame = ar.from_pandas(df)
        
        result, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        
        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        assert len(metadata["step_timings"]) == 2
    
    def test_return_metadata_row_counts(self):
        """Test that row counts are correctly tracked in metadata."""
        df = pd.DataFrame({"value": [1, 1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result, metadata = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            return_metadata=True
        )
        
        row_counts = metadata["row_counts"]
        assert row_counts[0]["before"] == 4
        assert row_counts[0]["after"] == 3


class TestPipelineDryRun:
    """Test dry_run functionality."""
    
    def test_dry_run_no_modifications(self):
        """Test that dry_run=True doesn't modify the frame."""
        df = pd.DataFrame({"value": [1, 1, 2]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("drop_duplicates",)], dry_run=True)
        result_df = ar.to_pandas(result)
        
        # Frame should remain unchanged (3 rows)
        assert len(result_df) == 3
    
    def test_dry_run_with_metadata(self):
        """Test that dry_run with metadata shows no row changes."""
        df = pd.DataFrame({"value": [1, 1, 2]})
        frame = ar.from_pandas(df)
        
        result, metadata = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            return_metadata=True,
            dry_run=True
        )
        
        row_counts = metadata["row_counts"]
        assert row_counts[0]["before"] == row_counts[0]["after"] == 3
        assert row_counts[0]["dry_run"] is True


class TestPipelineLineageTracking:
    """Test lineage tracking functionality."""
    
    def test_lineage_basic(self):
        """Test basic lineage tracking."""
        df = pd.DataFrame({"value": [1, None, 3]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        
        assert hasattr(lineage, "dropped_by_step")
        assert "drop_nulls" in lineage.dropped_by_step
        assert lineage.total_dropped >= 1
    
    def test_lineage_multiple_steps(self):
        """Test lineage tracking across multiple steps."""
        df = pd.DataFrame({"value": [1, None, 1, 4]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            track_lineage=True
        )
        
        assert "drop_nulls" in lineage.dropped_by_step
        assert "drop_duplicates" in lineage.dropped_by_step
    
    def test_lineage_no_sentinel_in_output(self):
        """Test that sentinel column is removed from output."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            track_lineage=True
        )
        
        result_df = ar.to_pandas(result)
        assert "__arnio_lineage_id__" not in result_df.columns


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""
    
    def test_invalid_frame_type(self):
        """Test that non-ArFrame raises TypeError."""
        df = pd.DataFrame({"a": [1, 2]})
        
        with pytest.raises(TypeError):
            ar.pipeline(df, [("strip_whitespace",)])
    
    def test_invalid_return_metadata_type(self):
        """Test that non-bool return_metadata raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        
        with pytest.raises(TypeError):
            ar.pipeline(frame, [("strip_whitespace",)], return_metadata="yes")
    
    def test_invalid_step_format_too_many_elements(self):
        """Test that step tuple with >2 elements raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step", {}, "extra")])
    
    def test_unknown_step_error(self):
        """Test that unknown step name raises UnknownStepError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step",)])
    
    def test_lineage_and_metadata_mutually_exclusive(self):
        """Test that track_lineage and return_metadata cannot be used together."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("strip_whitespace",)],
                track_lineage=True,
                return_metadata=True
            )
    
    def test_lineage_sentinel_column_collision(self):
        """Test that existing __arnio_lineage_id__ column raises ValueError."""
        df = pd.DataFrame({
            "a": [1, 2],
            "__arnio_lineage_id__": [0, 1],
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)


class TestPipelineCustomSteps:
    """Test custom step registration and execution."""
    
    def test_custom_step_registration_and_use(self):
        """Test registering and using a custom pipeline step."""
        def double_column(df, column):
            df = df.copy()
            df[column] = df[column] * 2
            return df
        
        ar.register_step("double_column", double_column)
        
        try:
            frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))
            result = ar.pipeline(frame, [
                ("double_column", {"column": "value"}),
            ])
            result_df = ar.to_pandas(result)
            assert result_df["value"].tolist() == [2, 4, 6]
        finally:
            ar.reset_steps()


class TestPipelineEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_pipeline(self):
        """Test pipeline with no steps returns the frame unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        
        assert result_df.equals(df)
    
    def test_invalid_step_kwargs_type(self):
        """Test that step with non-dict kwargs raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        
        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("strip_whitespace", "invalid")])
    
    def test_lineage_same_step_multiple_times(self):
        """Test lineage tracking when same step appears multiple times."""
        df = pd.DataFrame({"value": [1, None, 1, None, 3]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_nulls",)],
            track_lineage=True
        )
        
        # Should have a single "drop_nulls" key with all dropped indices merged
        assert "drop_nulls" in lineage.dropped_by_step
        assert isinstance(lineage.dropped_by_step["drop_nulls"], list)
