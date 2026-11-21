import pytest
import pandas as pd
import arnio as ar
from pathlib import Path
import numpy as np


class TestAutoCleanBasic:
    """Test basic auto_clean functionality with real CSV files."""
    
    def test_safe_mode_default_strips_whitespace(self, tmp_path):
        """Test safe mode strips whitespace by default."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,age\n  Alice  ,25\n  Bob  ,30\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame)
        
        df = ar.to_pandas(clean)
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[1, "name"] == "Bob"
    
    def test_safe_mode_explicit_parameter(self, tmp_path):
        """Test safe mode with explicit mode='safe' parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n  x  ,1\n  y  ,2\n  z  ,3\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        
        df = ar.to_pandas(clean)
        assert df.shape == (3, 2)
        assert df.loc[0, "col1"] == "x"
    
    def test_safe_mode_preserves_nulls(self, tmp_path):
        """Test safe mode does not drop nulls."""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text("a,b\n1,\n2,20\n,30\n")
        
        frame = ar.read_csv(str(csv_file))
        rows_before = frame.shape[0]
        clean = ar.auto_clean(frame, mode="safe")
        rows_after = clean.shape[0]
        
        assert rows_after == rows_before


class TestAutoCleanStrictMode:
    """Test strict mode cleaning behavior."""
    
    def test_strict_mode_requires_confirmed_casts(self, tmp_path):
        """Test strict mode errors without confirmed_casts when casts are needed."""
        csv_file = tmp_path / "typed.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        
        # Strict mode may propose casts; without allow_lossy_casts, it should error
        with pytest.raises(ValueError, match="allow_lossy_casts|cast"):
            ar.auto_clean(frame, mode="strict")
    
    def test_strict_mode_with_safe_data_no_error(self, tmp_path):
        """Test strict mode works on well-formed data."""
        csv_file = tmp_path / "clean.csv"
        csv_file.write_text("name,score\nAlice,95\nBob,87\nAlice,95\n")
        
        frame = ar.read_csv(str(csv_file))
        
        # Might still error if strict suggests casts, so we check graceful behavior
        try:
            clean = ar.auto_clean(frame, mode="strict")
            assert isinstance(clean, ar.ArFrame)
        except ValueError as e:
            # If it errors, it should be about casts
            assert "cast" in str(e).lower() or "allow_lossy_casts" in str(e)


class TestAutoCleanDryRun:
    """Test dry_run behavior."""
    
    def test_dry_run_returns_report(self, tmp_path):
        """Test dry_run returns report without modifying frame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city\n  Alice  , Paris \n  Bob  , London \n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.auto_clean(frame, dry_run=True)
        
        assert isinstance(report, ar.DataQualityReport)
        # Original frame should be unchanged
        df_orig = ar.to_pandas(frame)
        assert df_orig.loc[0, "name"] == "  Alice  "
    
    def test_dry_run_with_strict_mode(self, tmp_path):
        """Test dry_run in strict mode shows what would happen."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n  10  ,  20  \n  30  ,  40  \n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        
        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(report, "suggestions")


class TestAutoCleanReturns:
    """Test different return modes."""
    
    def test_return_report_flag(self, tmp_path):
        """Test return_report returns tuple of (frame, report)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n  1  ,  2  \n  3  ,  4  \n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.auto_clean(frame, return_report=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, report = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)
    
    def test_explain_flag(self, tmp_path):
        """Test explain flag returns tuple with explanation."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n  hello  , world \n  foo  , bar \n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.auto_clean(frame, explain=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        clean_frame, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        # Explanation should have useful attributes
        assert hasattr(explanation, "mode") or hasattr(explanation, "steps")
    
    def test_explain_with_return_report(self, tmp_path):
        """Test combining explain and return_report."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n  1  ,  2  \n  1  ,  2  \n  3  ,  4  \n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.auto_clean(
            frame,
            mode="safe",
            return_report=True,
            explain=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        clean_frame, report, explanation = result
        assert isinstance(clean_frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(explanation, "mode") or hasattr(explanation, "steps")


class TestAutoCleanValidation:
    """Test input validation and error handling."""
    
    def test_invalid_mode_raises(self, tmp_path):
        """Test invalid mode parameter raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="mode must be"):
            ar.auto_clean(frame, mode="invalid_mode")
    
    def test_invalid_return_report_type_raises(self, tmp_path):
        """Test non-bool return_report raises TypeError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError):
            ar.auto_clean(frame, return_report="yes")
    
    def test_dry_run_with_explain_raises(self, tmp_path):
        """Test dry_run=True with explain=True raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run"):
            ar.auto_clean(frame, dry_run=True, explain=True)
    
    def test_dry_run_with_return_report_raises(self, tmp_path):
        """Test dry_run=True with return_report=True raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run"):
            ar.auto_clean(frame, dry_run=True, return_report=True)


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_frame(self, tmp_path):
        """Test auto_clean handles empty frame gracefully."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("a,b\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame)
        
        assert clean.shape[0] == 0
        assert clean.shape[1] == 2
    
    def test_single_row_frame(self, tmp_path):
        """Test auto_clean on single-row frame."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("name,value\n  Alice  ,  100  \n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        
        df = ar.to_pandas(clean)
        assert df.shape[0] == 1
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "value"] == "100"
    
    def test_all_whitespace_cells(self, tmp_path):
        """Test frame with mostly whitespace cells."""
        csv_file = tmp_path / "spaces.csv"
        csv_file.write_text("col1,col2\n   ,   \n  a  ,  b  \n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        
        df = ar.to_pandas(clean)
        # Whitespace should be trimmed
        assert df.loc[1, "col1"] == "a"


class TestAutoCleanIntegration:
    """Test integration with other arnio and pandas features."""
    
    def test_cleaned_frame_converts_to_pandas(self, tmp_path):
        """Test cleaned frame can be converted to pandas."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,  Alice  \n2,  Bob  \n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name"]
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[1, "name"] == "Bob"
    
    def test_write_csv_after_cleaning(self, tmp_path):
        """Test writing cleaned frame back to CSV."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        input_file.write_text("col1,col2\n  x  ,  1  \n  y  ,  2  \n")
        
        frame = ar.read_csv(str(input_file))
        clean = ar.auto_clean(frame, mode="safe")
        ar.write_csv(clean, str(output_file))
        
        assert output_file.exists()
        output_content = output_file.read_text()
        assert "x,1" in output_content
        assert "y,2" in output_content
    
    def test_pipeline_after_auto_clean(self, tmp_path):
        """Test applying further pipeline steps after auto_clean."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,score\n  alice  ,95\n  bob  ,87\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        
        # Apply additional pipeline steps
        result = ar.pipeline(clean, [
            ("normalize_case", {"case_type": "upper"}),
        ])
        
        df = ar.to_pandas(result)
        assert df.loc[0, "name"] == "ALICE"
        assert df.loc[1, "name"] == "BOB"
    
    def test_pandas_accessor_after_to_pandas(self, tmp_path):
        """Test using pandas .arnio accessor on cleaned data."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("value,category\n  10  ,  A  \n  20  ,  B  \n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean)
        
        # Can chain arnio operations on pandas
        profile = df.arnio.profile()
        assert isinstance(profile, ar.DataQualityReport)
