import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipeline:
    """Test suite for ar.pipeline() function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Charlie  "],
            "age": [25, 30, 25],
            "city": ["NYC", "LA", "NYC"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "value": [10.0, None, 30.0, None],
            "category": ["A", "B", "A", "B"],
        })
        return ar.from_pandas(df)

    def test_pipeline_single_step_no_kwargs(self, sample_frame):
        """Test pipeline with a single step that has no kwargs."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_pipeline_single_step_with_kwargs(self, sample_frame):
        """Test pipeline with a single step that has kwargs."""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "upper"}),
        ])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df["city"].tolist() == ["NYC", "LA", "NYC"]

    def test_pipeline_multiple_steps_sequence(self, sample_frame):
        """Test pipeline with multiple steps executed in order."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert len(df) == 2

    def test_pipeline_mixed_kwargs_and_no_kwargs(self, sample_frame):
        """Test pipeline mixing steps with and without kwargs."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df["name"].iloc[0] == "alice"

    def test_pipeline_return_metadata(self, sample_frame):
        """Test pipeline with return_metadata=True returns tuple."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            return_metadata=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata

    def test_pipeline_metadata_content(self, sample_frame):
        """Test metadata contains expected step information."""
        frame, metadata = ar.pipeline(
            sample_frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
            return_metadata=True,
        )
        assert len(metadata["applied_steps"]) == 2
        assert metadata["applied_steps"][0] == "strip_whitespace"
        assert metadata["applied_steps"][1] == "drop_duplicates"
        assert len(metadata["step_timings"]) == 2

    def test_pipeline_dry_run_validation(self, sample_frame):
        """Test dry_run=True validates without crashing."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            dry_run=True,
        )
        # Dry run should return without error

    def test_pipeline_unknown_step_raises_error(self, sample_frame):
        """Test unknown step name raises UnknownStepError."""
        with pytest.raises(UnknownStepError):
            ar.pipeline(sample_frame, [("nonexistent_step",)])

    def test_pipeline_malformed_step_tuple_extra_elements(self, sample_frame):
        """Test malformed step tuple raises error."""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(sample_frame, [("step", {}, "extra")])

    def test_pipeline_empty_steps_list(self, sample_frame):
        """Test pipeline with empty steps list returns identical frame."""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
        original_df = ar.to_pandas(sample_frame)
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(original_df, result_df)

    def test_pipeline_fill_nulls_step(self, frame_with_nulls):
        """Test pipeline with fill_nulls step."""
        result = ar.pipeline(frame_with_nulls, [
            ("fill_nulls", {"value": 0.0, "subset": ["value"]}),
        ])
        df = ar.to_pandas(result)
        assert df["value"].isna().sum() == 0
        assert df["value"].tolist() == [10.0, 0.0, 30.0, 0.0]

    def test_pipeline_custom_step_registration(self, sample_frame):
        """Test pipeline with a custom registered step."""
        def custom_upper(df):
            df = df.copy()
            if "name" in df.columns:
                df["name"] = df["name"].str.upper()
            return df

        ar.register_step("custom_upper", custom_upper)
        try:
            result = ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("custom_upper",),
            ])
            df = ar.to_pandas(result)
            assert df["name"].iloc[0] == "ALICE"
        finally:
            ar.reset_steps()

    def test_pipeline_preserves_column_order(self, sample_frame):
        """Test pipeline preserves column order with non-schema-changing steps."""
        original_columns = sample_frame.columns
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert result.columns == list(original_columns)

    def test_pipeline_verbose_mode_no_crash(self, sample_frame):
        """Test verbose mode executes without crashing."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            verbose=True,
        )
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_complex_multi_step_workflow(self):
        """Test complex pipeline with multiple step types."""
        df = pd.DataFrame({
            "name": ["  ALICE  ", "  bob  ", "  alice  "],
            "score": [95.0, 88.0, 95.0],
            "status": ["active", "inactive", "active"],
        })
        frame = ar.from_pandas(df)

        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates", {"keep": "first"}),
        ])

        df_result = ar.to_pandas(result)
        assert len(df_result) == 2
        assert df_result["name"].iloc[0] == "alice"
        assert df_result["score"].iloc[0] == 95.0
