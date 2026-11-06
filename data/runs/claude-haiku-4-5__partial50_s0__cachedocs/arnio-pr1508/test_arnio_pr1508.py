import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    def test_pipeline_empty_steps(self):
        """Test pipeline with empty steps list returns unchanged frame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        
        pd.testing.assert_frame_equal(result_df, df)
    
    def test_pipeline_single_step_no_args(self):
        """Test pipeline with single step that has no arguments."""
        df = pd.DataFrame({"name": [" Alice ", " Bob "]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Alice", "Bob"]
    
    def test_pipeline_single_step_with_kwargs(self):
        """Test pipeline with single step that has keyword arguments."""
        df = pd.DataFrame({"text": ["hello", "WORLD"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "upper"})
        ])
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].tolist() == ["HELLO", "WORLD"]
    
    def test_pipeline_multiple_steps_chained(self):
        """Test pipeline executes multiple steps in order."""
        df = pd.DataFrame({
            "name": [" alice ", " bob ", " alice "],
            "value": [1, 2, 1]
        })
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "upper"}),
            ("drop_duplicates", {"keep": "first"})
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert result_df["name"].tolist() == ["ALICE", "BOB"]


class TestPipelineMetadata:
    """Test metadata and return value options."""
    
    def test_pipeline_default_return_type(self):
        """Test default return type is ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_return_metadata_true(self):
        """Test return_metadata=True returns tuple with ArFrame and dict."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            return_metadata=True
        )
        
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_metadata_tracks_steps(self):
        """Test metadata tracks applied steps."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        frame = ar.from_pandas(df)
        
        _, metadata = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",)
            ],
            return_metadata=True
        )
        
        assert len(metadata["applied_steps"]) == 2
        assert metadata["applied_steps"][0] == "strip_whitespace"
        assert metadata["applied_steps"][1] == "drop_duplicates"


class TestPipelineValidation:
    """Test step validation and error handling."""
    
    def test_pipeline_unknown_step_raises(self):
        """Test that unknown step names raise UnknownStepError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step",)])
    
    def test_pipeline_malformed_tuple_raises(self):
        """Test that malformed step tuples raise error before execution."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(frame, [("step", "bad_format", "extra")])
    
    def test_pipeline_dry_run_validates_without_executing(self):
        """Test dry_run=True validates pipeline without modifying data."""
        df = pd.DataFrame({"a": [1, 2, None]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(
            frame,
            [("drop_nulls",)],
            dry_run=True
        )
        
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)


class TestPipelineCommonSteps:
    """Test behavior of common pipeline steps."""
    
    def test_pipeline_drop_nulls_step(self):
        """Test drop_nulls step removes null rows."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("drop_nulls",)])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert result_df["a"].tolist() == [1.0, 3.0]
    
    def test_pipeline_drop_duplicates_step(self):
        """Test drop_duplicates step removes duplicate rows."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"})
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
    
    def test_pipeline_fill_nulls_step(self):
        """Test fill_nulls step replaces null values."""
        df = pd.DataFrame({"value": [1.0, None, 3.0]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0})
        ])
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].tolist() == [1.0, 0.0, 3.0]
    
    def test_pipeline_cast_types_step(self):
        """Test cast_types step converts column types."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("cast_types", {"mapping": {"num": "int64"}})
        ])
        result_df = ar.to_pandas(result)
        
        assert result_df["num"].dtype == np.int64


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_pipeline_empty_dataframe(self):
        """Test pipeline on empty dataframe preserves structure."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=str)})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls",)
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert result_df.columns.tolist() == ["a", "b"]
    
    def test_pipeline_single_row(self):
        """Test pipeline on single-row dataframe."""
        df = pd.DataFrame({"x": [42], "y": ["test"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("drop_duplicates",)])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 1
        assert result_df["x"].iloc[0] == 42
    
    def test_pipeline_all_duplicates(self):
        """Test pipeline with all identical rows."""
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"})
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 1
    
    def test_pipeline_preserves_column_order(self):
        """Test that pipeline preserves original column order."""
        df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["z", "a", "m"]
    
    def test_pipeline_complex_workflow(self):
        """Test a realistic multi-step pipeline."""
        df = pd.DataFrame({
            "customer": [" Alice ", " Bob ", " Alice ", "  Charlie  "],
            "revenue": [100.5, None, 100.5, 250.0],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]
        })
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "title"}),
            ("fill_nulls", {"value": 0.0, "subset": ["revenue"]}),
            ("drop_duplicates", {"keep": "first"})
        ])
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 3
        assert result_df["customer"].tolist() == ["Alice", "Bob", "Charlie"]
        assert result_df["revenue"].tolist() == [100.5, 0.0, 250.0]
