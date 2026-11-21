"""
System (end-to-end) tests for arnio.pipeline.pipeline
Tests the complete pipeline workflow with real files and observable results.
"""
import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path
import logging


class TestPipelineEndToEnd:
    """End-to-end tests for arnio.pipeline.pipeline"""
    
    def test_pipeline_basic_cleaning_workflow(self, tmp_path):
        """Test basic cleaning pipeline: read CSV, clean, verify results."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,age,city\n  Alice  ,30,  Paris  \n  Bob  ,25,London\n")
        
        frame = ar.read_csv(str(csv_file))
        cleaned = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        
        df = ar.to_pandas(cleaned)
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["city"] == "Paris"
    
    def test_pipeline_multiple_steps_sequence(self, tmp_path):
        """Test pipeline with multiple sequential steps."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score\n  Alice  ,95\n  Bob  ,88\nAlice,95\n  Charlie  ,\n")
        
        frame = ar.read_csv(str(csv_file))
        cleaned = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
            ("drop_nulls",),
        ])
        
        df = ar.to_pandas(cleaned)
        assert len(df) == 2  # Alice (kept first), Bob
    
    def test_pipeline_with_metadata_tracking(self, tmp_path):
        """Test pipeline returns detailed metadata about step execution."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,city\nAlice,Paris\nAlice,Paris\nBob,London\n")
        
        frame = ar.read_csv(str(csv_file))
        result, metadata = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], return_metadata=True)
        
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        assert len(metadata["step_timings"]) == 2
        assert metadata["row_counts"][0]["before"] == 3
        assert metadata["row_counts"][1]["after"] == 2
        assert "execution_summary" in metadata
        assert metadata["execution_summary"]["total_runtime_ms"] >= 0
    
    def test_pipeline_dry_run_preserves_data(self, tmp_path):
        """Test dry_run validates without modifying data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], dry_run=True)
        
        # dry_run should not actually remove duplicates
        assert result.shape[0] == 3
    
    def test_pipeline_track_lineage_identifies_dropped_rows(self, tmp_path):
        """Test track_lineage identifies which rows are dropped by each step."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score\nAlice,95\nBob,\nCharlie,88\nCharlie,88\n")
        
        frame = ar.read_csv(str(csv_file))
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls",),
            ("drop_duplicates",),
        ], track_lineage=True)
        
        assert "drop_nulls" in lineage.dropped_by_step
        assert "drop_duplicates" in lineage.dropped_by_step
        assert lineage.total_dropped == 2
        assert result.shape[0] == 2
    
    def test_pipeline_incompatible_flags_raises_error(self, tmp_path):
        """Test that track_lineage and return_metadata together raise ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [
                ("strip_whitespace",),
            ], track_lineage=True, return_metadata=True)
    
    def test_pipeline_invalid_step_raises_error(self, tmp_path):
        """Test pipeline with unknown step name raises UnknownStepError early."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step",)])
    
    def test_pipeline_invalid_input_raises_type_error(self):
        """Test pipeline rejects non-ArFrame input with clear TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="must be an ArFrame"):
            ar.pipeline(df, [("strip_whitespace",)])
    
    def test_pipeline_with_custom_registered_step(self, tmp_path):
        """Test pipeline using a custom registered Python step."""
        def uppercase_names(df, column="name"):
            df = df.copy()
            df[column] = df[column].str.upper()
            return df
        
        ar.register_step("uppercase_names", uppercase_names)
        
        try:
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("name,city\nAlice,Paris\nBob,London\n")
            
            frame = ar.read_csv(str(csv_file))
            result = ar.pipeline(frame, [
                ("strip_whitespace",),
                ("uppercase_names", {"column": "name"}),
            ])
            
            df = ar.to_pandas(result)
            assert df.iloc[0]["name"] == "ALICE"
        finally:
            ar.reset_steps()
    
    def test_pipeline_column_renaming_and_dropping(self, tmp_path):
        """Test pipeline with column schema changes."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("old_id,old_name,debug\n1,Alice,xyz\n2,Bob,abc\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("rename_columns", {"old_id": "id", "old_name": "name"}),
            ("drop_columns", {"columns": ["debug"]}),
        ])
        
        assert result.columns == ["id", "name"]
    
    def test_pipeline_type_casting_in_pipeline(self, tmp_path):
        """Test pipeline with type casting."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,active\n1,95.5,true\n2,88.3,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("cast_types", {"mapping": {"id": "int64", "score": "float64"}}),
        ])
        
        df = ar.to_pandas(result)
        assert df["id"].dtype == np.int64
        assert df["score"].dtype == np.float64
    
    def test_pipeline_read_clean_write_roundtrip(self, tmp_path):
        """Test complete I/O workflow: read, clean, write to CSV."""
        input_file = tmp_path / "messy.csv"
        input_file.write_text("name,city,score\n  Alice  ,Paris,95\n  Bob  ,London,88\n  Alice  ,Paris,95\n")
        output_file = tmp_path / "clean.csv"
        
        # Read
        frame = ar.read_csv(str(input_file))
        # Clean
        cleaned = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        # Write
        ar.write_csv(cleaned, str(output_file))
        
        # Verify output file exists and has correct content
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 2
        assert output_df.iloc[0]["name"] == "Alice"
        assert output_df.iloc[1]["name"] == "Bob"
    
    def test_pipeline_with_pandas_accessor_integration(self, tmp_path):
        """Test pipeline integration through pandas .arnio accessor."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Charlie  "],
            "age": [30, 25, 30],
        })
        
        # Clean using accessor
        cleaned_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates={"keep": "first"}
        )
        
        assert len(cleaned_df) == 2
        assert cleaned_df.iloc[0]["name"] == "Alice"
        assert cleaned_df.iloc[1]["name"] == "Bob"
    
    def test_pipeline_verbose_logging_produces_output(self, tmp_path, caplog):
        """Test that verbose=True produces logged diagnostics."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,city\n  Alice  ,Paris\n  Bob  ,London\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with caplog.at_level(logging.INFO):
            ar.pipeline(frame, [
                ("strip_whitespace",),
            ], verbose=True)
        
        # Check for logging output
        assert len(caplog.records) > 0 or "strip_whitespace" in caplog.text
    
    def test_pipeline_all_rows_dropped_produces_empty_frame(self, tmp_path):
        """Test pipeline that drops all rows produces valid empty frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,score\nAlice,\nBob,\nCharlie,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_nulls",),
        ])
        
        assert result.shape[0] == 0
        assert result.shape[1] == 2
        assert result.columns == ["name", "score"]
    
    def test_pipeline_step_format_validation_raises_error(self, tmp_path):
        """Test pipeline validates step tuple format before execution."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\n")
        
        frame = ar.read_csv(str(csv_file))
        
        # Too many elements in step tuple
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step_name", {}, "extra_arg")])
        
        # kwargs is not a dict
        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("normalize_case", "invalid")])
