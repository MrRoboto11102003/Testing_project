import pytest
import re
from pathlib import Path

import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumnsMatching:
    """End-to-end tests for arnio.cleaning.drop_columns_matching."""

    def test_basic_arframe_drop_columns_matching(self, tmp_path):
        """Test basic column dropping from ArFrame via CSV workflow."""
        # Setup: create a CSV file
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,temp_col1,temp_col2,revenue\n"
            "1,Alice,X,Y,100\n"
            "2,Bob,X,Y,200\n"
        )
        
        # Load CSV into ArFrame
        frame = ar.read_csv(str(csv_path))
        original_cols = set(frame.columns)
        assert "temp_col1" in original_cols
        assert "temp_col2" in original_cols
        
        # Clean: drop columns matching pattern
        cleaned = ar.drop_columns_matching(frame, r"^temp_")
        
        # Verify: check columns
        assert set(cleaned.columns) == {"id", "name", "revenue"}
        
        # Convert to pandas and verify data integrity
        df = ar.to_pandas(cleaned)
        assert len(df) == 2
        assert df["id"].tolist() == [1, 2]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_pandas_dataframe_drop_columns_matching(self, tmp_path):
        """Test column dropping from pandas DataFrame."""
        # Create DataFrame
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "debug_flag": ["a", "b", "c"],
            "debug_info": ["x", "y", "z"],
            "name": ["Alice", "Bob", "Charlie"],
        })
        
        # Clean using arnio function
        result = ar.drop_columns_matching(df, r"^debug_")
        
        # Verify: result is DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"user_id", "name"}
        assert len(result) == 3
        assert result["user_id"].tolist() == [1, 2, 3]

    def test_invalid_regex_raises_error(self, tmp_path):
        """Test that invalid regex pattern raises re.error."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_path))
        
        # Invalid regex should raise re.error
        with pytest.raises(re.error):
            ar.drop_columns_matching(frame, r"[invalid(")

    def test_non_string_pattern_raises_typeerror(self, tmp_path):
        """Test that non-string pattern parameter raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_path))
        
        # Non-string pattern should raise TypeError
        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.drop_columns_matching(frame, 123)

    def test_pattern_matches_all_columns_raises_error(self, tmp_path):
        """Test that pattern matching all columns raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2,col3\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_path))
        
        # Pattern matching all columns should raise ValueError
        with pytest.raises(ValueError, match="at least one column must remain"):
            ar.drop_columns_matching(frame, r"col")

    def test_no_matching_columns(self, tmp_path):
        """Test that pattern with no matches returns frame unchanged."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,revenue\n1,Alice,100\n2,Bob,200\n")
        
        frame = ar.read_csv(str(csv_path))
        original_cols = frame.columns
        
        # Pattern with no matches
        result = ar.drop_columns_matching(frame, r"^nonexistent_")
        
        # Should return frame with same columns
        assert result.columns == original_cols
        df = ar.to_pandas(result)
        assert len(df) == 2

    def test_csv_to_csv_workflow(self, tmp_path):
        """Test complete workflow: read CSV, clean, write CSV."""
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        
        # Setup: write input CSV
        input_csv.write_text(
            "id,name,_cache_key,_cache_val,amount\n"
            "1,Alice,k1,v1,100\n"
            "2,Bob,k2,v2,200\n"
            "3,Charlie,k3,v3,300\n"
        )
        
        # Load, clean, convert to pandas, write
        frame = ar.read_csv(str(input_csv))
        cleaned = ar.drop_columns_matching(frame, r"^_cache_")
        df = ar.to_pandas(cleaned)
        
        # Write output
        df.to_csv(str(output_csv), index=False)
        
        # Verify: output file exists and has correct content
        output_text = output_csv.read_text()
        assert "id,name,amount" in output_text
        assert "_cache_key" not in output_text
        assert "_cache_val" not in output_text
        assert len(output_text.strip().split("\n")) == 4  # header + 3 data rows

    def test_pipeline_integration(self, tmp_path):
        """Test using drop_columns_matching as a pipeline step."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,temp_1,temp_2,score\n"
            "1,Alice,X,Y,95\n"
            "2,Bob,A,B,87\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Use in pipeline
        clean = ar.pipeline(frame, [
            ("drop_columns_matching", {"pattern": r"^temp_"}),
        ])
        
        assert clean.columns == ["id", "name", "score"]
        df = ar.to_pandas(clean)
        assert len(df) == 2

    def test_case_sensitive_matching(self, tmp_path):
        """Test that pattern matching is case-sensitive."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "ID,id,Name,name\n"
            "1,2,Alice,Bob\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Pattern ^id$ should only match lowercase 'id'
        result = ar.drop_columns_matching(frame, r"^id$")
        
        assert "ID" in result.columns
        assert "id" not in result.columns
        assert "Name" in result.columns
        assert "name" in result.columns

    def test_complex_regex_pattern(self, tmp_path):
        """Test with complex regex patterns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "col_1,col_2,col_x,metric_123,metric_456,name\n"
            "1,2,3,4,5,test\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Pattern: drop columns ending with digits
        result = ar.drop_columns_matching(frame, r"\d+$")
        
        assert "col_1" not in result.columns
        assert "col_2" not in result.columns
        assert "metric_123" not in result.columns
        assert "metric_456" not in result.columns
        assert set(result.columns) == {"col_x", "name"}

    def test_partial_name_matching(self, tmp_path):
        """Test matching partial column names."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "user_id,user_name,user_email,product_id,product_name\n"
            "1,Alice,alice@example.com,10,Widget\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Drop all 'user_*' columns
        result = ar.drop_columns_matching(frame, r"^user_")
        
        assert set(result.columns) == {"product_id", "product_name"}

    def test_column_order_preservation(self, tmp_path):
        """Test that column order is preserved after dropping."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "z_keep,a_drop,m_keep,b_drop\n"
            "1,2,3,4\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns_matching(frame, r"^[ab]_")
        
        # Remaining columns should maintain original order
        assert result.columns == ["z_keep", "m_keep"]

    def test_special_regex_characters(self, tmp_path):
        """Test patterns with special regex characters."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "col.a,col-b,col[c],normal\n"
            "1,2,3,4\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Match columns with literal dots
        result = ar.drop_columns_matching(frame, r"col\.")
        
        assert "col.a" not in result.columns
        assert "col-b" in result.columns
        assert "col[c]" in result.columns
        assert "normal" in result.columns

    def test_data_integrity_after_dropping(self, tmp_path):
        """Test that data values are not corrupted when dropping columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,value,debug_col\n"
            "1,10.5,X\n"
            "2,20.3,Y\n"
            "3,15.7,Z\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_columns_matching(frame, r"^debug_")
        df = ar.to_pandas(result)
        
        # Verify data integrity
        assert df["id"].tolist() == [1, 2, 3]
        assert np.allclose(df["value"].tolist(), [10.5, 20.3, 15.7])
        assert len(df) == 3

    def test_multiple_pattern_steps_in_pipeline(self, tmp_path):
        """Test chaining multiple drop_columns_matching steps."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,temp_col,debug_flag,score\n"
            "1,Alice,X,Y,95\n"
            "2,Bob,A,B,87\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Drop columns in sequence
        result = ar.pipeline(frame, [
            ("drop_columns_matching", {"pattern": r"^temp_"}),
            ("drop_columns_matching", {"pattern": r"^debug_"}),
        ])
        
        assert set(result.columns) == {"id", "name", "score"}
        df = ar.to_pandas(result)
        assert len(df) == 2
