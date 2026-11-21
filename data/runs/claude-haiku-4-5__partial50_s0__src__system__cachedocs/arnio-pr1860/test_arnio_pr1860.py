import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

import arnio as ar


class TestWriteParquetBasic:
    """Basic write_parquet functionality."""
    
    def test_write_parquet_basic(self, tmp_path):
        """Write an ArFrame to Parquet and verify file creation."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")
        
        frame = ar.read_csv(str(csv_path))
        parquet_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, str(parquet_path))
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
    
    def test_write_parquet_pq_extension(self, tmp_path):
        """Test .pq extension (alternative to .parquet)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(str(csv_path))
        parquet_path = tmp_path / "output.pq"
        ar.write_parquet(frame, str(parquet_path))
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 2


class TestWriteParquetCompression:
    """Test compression codecs and row_group_size parameter."""
    
    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_compression_codecs(self, tmp_path, codec):
        """Test all supported compression codecs."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,value\n1,100\n2,200\n3,300\n")
        
        frame = ar.read_csv(str(csv_path))
        parquet_path = tmp_path / f"output_{codec}.parquet"
        ar.write_parquet(frame, str(parquet_path), compression=codec)
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 3
        assert list(df.columns) == ["id", "value"]
    
    def test_row_group_size(self, tmp_path):
        """Write with custom row_group_size parameter."""
        csv_path = tmp_path / "data.csv"
        content = "id,value\n"
        for i in range(100):
            content += f"{i},{i*10}\n"
        csv_path.write_text(content)
        
        frame = ar.read_csv(str(csv_path))
        parquet_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, str(parquet_path), row_group_size=25)
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 100


class TestWriteParquetInputValidation:
    """Test input validation and type checking."""
    
    def test_invalid_file_extension(self, tmp_path):
        """Raise ValueError for unsupported file extensions."""
        frame = ar.from_dict({"id": [1, 2], "name": ["A", "B"]})
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, str(tmp_path / "output.csv"))
    
    def test_invalid_compression_codec(self, tmp_path):
        """Raise ValueError for invalid compression codec."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), compression="lz4")
    
    def test_invalid_row_group_size_negative(self, tmp_path):
        """Raise ValueError for negative row_group_size."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), row_group_size=-1)
    
    def test_invalid_row_group_size_zero(self, tmp_path):
        """Raise ValueError for zero row_group_size."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), row_group_size=0)
    
    def test_invalid_row_group_size_bool(self, tmp_path):
        """Raise TypeError if row_group_size is a bool."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), row_group_size=True)
    
    def test_invalid_preserve_attrs_type(self, tmp_path):
        """Raise TypeError if preserve_attrs is not bool."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), preserve_attrs="yes")
    
    def test_non_arframe_input(self, tmp_path):
        """Raise TypeError if frame is not an ArFrame."""
        df = pd.DataFrame({"id": [1, 2]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, str(tmp_path / "output.parquet"))
    
    def test_invalid_path_type(self, tmp_path):
        """Raise TypeError for invalid path type."""
        frame = ar.from_dict({"id": [1, 2]})
        
        with pytest.raises(TypeError, match="path must be a string"):
            ar.write_parquet(frame, 12345)
    
    def test_non_json_serializable_attrs(self, tmp_path):
        """Raise TypeError for non-JSON-serializable attrs with preserve_attrs=True."""
        df = pd.DataFrame({"id": [1, 2]})
        df.attrs = {"callback": lambda x: x}  # Non-serializable object
        
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, str(tmp_path / "output.parquet"), preserve_attrs=True)


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_frame(self, tmp_path):
        """Write empty ArFrame successfully with column structure intact."""
        frame = ar.from_dict({"id": [], "name": [], "score": []})
        parquet_path = tmp_path / "empty.parquet"
        ar.write_parquet(frame, str(parquet_path))
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 0
        assert list(df.columns) == ["id", "name", "score"]
    
    def test_pathlib_path_object(self, tmp_path):
        """Accept pathlib.Path objects as path argument."""
        frame = ar.from_dict({"id": [1, 2, 3], "value": [10, 20, 30]})
        parquet_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, parquet_path)  # Pass Path object directly
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 3
    
    def test_preserve_attrs_true_with_valid_metadata(self, tmp_path):
        """Preserve JSON-serializable DataFrame.attrs when preserve_attrs=True."""
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df.attrs = {"author": "test_user", "version": 1, "tags": ["valid", "data"]}
        
        frame = ar.from_pandas(df)
        parquet_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, str(parquet_path), preserve_attrs=True)
        
        assert parquet_path.exists()
        df_read = pd.read_parquet(str(parquet_path))
        assert len(df_read) == 2
    
    def test_preserve_attrs_false_drops_metadata(self, tmp_path):
        """Drop DataFrame.attrs when preserve_attrs=False."""
        df = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        df.attrs = {"metadata": "will_be_dropped"}
        
        frame = ar.from_pandas(df)
        parquet_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, str(parquet_path), preserve_attrs=False)
        
        assert parquet_path.exists()
        df_read = pd.read_parquet(str(parquet_path))
        assert len(df_read) == 2


class TestWriteParquetWorkflows:
    """Test complete data preparation workflows with write_parquet."""
    
    def test_csv_clean_pipeline_write(self, tmp_path):
        """Complete workflow: load messy CSV -> clean with pipeline -> write Parquet."""
        # Create messy CSV with whitespace and duplicates
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "id,name,score\n"
            "1, Alice ,95\n"
            "2, Bob ,88\n"
            "3, Alice ,95\n"  # Duplicate
            "4,  Charlie ,92\n"
        )
        
        # Load and apply cleaning pipeline
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        
        # Write to Parquet
        parquet_path = tmp_path / "clean.parquet"
        ar.write_parquet(clean, str(parquet_path), compression="gzip")
        
        # Verify results
        df = pd.read_parquet(str(parquet_path))
        assert len(df) == 3
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
    
    def test_validate_then_write_parquet(self, tmp_path):
        """Workflow: validate schema -> write Parquet on success."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,Alice,95\n2,Bob,88\n")
        
        frame = ar.read_csv(str(csv_path))
        
        # Define and apply schema validation
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "score": ar.Int64(nullable=False, min=0, max=100),
        })
        result = ar.validate(frame, schema)
        
        # Write only if validation passed
        assert result.passed
        parquet_path = tmp_path / "validated.parquet"
        ar.write_parquet(frame, str(parquet_path))
        assert parquet_path.exists()
    
    def test_profile_clean_write_workflow(self, tmp_path):
        """Workflow: profile -> suggest cleaning -> apply -> write Parquet."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,value,category\n"
            "1, 100 , A \n"
            "2, 200 , B \n"
            "3, 100 , A \n"  # Duplicate after cleaning
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Get quality profile and cleaning suggestions
        report = ar.profile(frame)
        assert report.row_count == 3
        
        suggestions = ar.suggest_cleaning(frame)
        
        # Apply suggested steps and write
        clean = ar.pipeline(frame, suggestions)
        parquet_path = tmp_path / "cleaned.parquet"
        ar.write_parquet(clean, str(parquet_path), row_group_size=10)
        
        assert parquet_path.exists()
        df = pd.read_parquet(str(parquet_path))
        assert len(df) <= 3
    
    def test_pandas_accessor_workflow(self, tmp_path):
        """Use pandas .arnio accessor for cleaning then write Parquet."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": [" Alice ", " Bob ", " Alice ", " Charlie "],
            "score": [95, 88, 95, 92]
        })
        
        # Use pandas accessor for cleaning
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates={"keep": "first"},
        )
        
        # Convert to ArFrame and write
        frame = ar.from_pandas(clean_df)
        parquet_path = tmp_path / "accessor_output.parquet"
        ar.write_parquet(frame, str(parquet_path))
        
        assert parquet_path.exists()
        result_df = pd.read_parquet(str(parquet_path))
        assert result_df["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert len(result_df) == 3
