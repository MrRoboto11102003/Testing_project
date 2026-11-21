"""
End-to-end pytest system tests for arnio.io.write_parquet.

These tests drive complete user workflows through arnio's public API,
including CSV loading, data cleaning, validation, and Parquet export.
"""

import pytest
import pandas as pd
import arnio as ar


class TestWriteParquetBasicFunctionality:
    """Basic write_parquet functionality."""

    def test_write_parquet_basic(self, tmp_path):
        """Write an ArFrame to parquet and verify file exists and is readable."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,value\n1,alice,10\n2,bob,20\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
        assert df["name"].tolist() == ["alice", "bob"]

    def test_write_parquet_pq_extension(self, tmp_path):
        """Write using .pq file extension (alternative to .parquet)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n5,6\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.pq"
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 3


class TestWriteParquetCompressionAndOptions:
    """Test compression codecs and row_group_size parameter."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "none"])
    def test_write_parquet_compression_codecs(self, tmp_path, codec):
        """Write with different compression codecs and verify data integrity."""
        csv_path = tmp_path / "data.csv"
        # Create dataset with 50 rows
        rows = "\n".join(f"{i},{i*2}" for i in range(50))
        csv_path.write_text(f"a,b\n{rows}\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / f"output_{codec}.parquet"
        ar.write_parquet(frame, output_path, compression=codec)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 50
        # Verify data matches original
        assert df["a"].tolist() == list(range(50))

    def test_write_parquet_with_row_group_size(self, tmp_path):
        """Write with explicit row_group_size for large files."""
        csv_path = tmp_path / "data.csv"
        rows = "\n".join(f"{i},{i*2}" for i in range(100))
        csv_path.write_text(f"id,value\n{rows}\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, output_path, row_group_size=25)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 100


class TestWriteParquetErrorHandling:
    """Test error conditions and input validation."""

    def test_write_parquet_invalid_file_extension(self, tmp_path):
        """Invalid file extension raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, output_path)

    def test_write_parquet_invalid_compression_codec(self, tmp_path):
        """Unknown compression codec raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, output_path, compression="invalid_codec")

    def test_write_parquet_non_arframe_input(self, tmp_path):
        """Non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output_path = tmp_path / "output.parquet"
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, output_path)

    def test_write_parquet_negative_row_group_size(self, tmp_path):
        """Negative row_group_size raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, output_path, row_group_size=-1)

    def test_write_parquet_zero_row_group_size(self, tmp_path):
        """Zero row_group_size raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, output_path, row_group_size=0)

    def test_write_parquet_invalid_preserve_attrs_type(self, tmp_path):
        """Invalid preserve_attrs type raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, output_path, preserve_attrs="true")


class TestWriteParquetWithPipelines:
    """Test write_parquet in complete data cleaning workflows."""

    def test_write_parquet_after_pipeline_cleanup(self, tmp_path):
        """Full workflow: read messy CSV, clean with pipeline, write Parquet."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "id,name,score\n"
            " 1 , alice , 95 \n"
            " 2 , bob , 87 \n"
            " 3 , alice , 95 \n"  # duplicate row
        )
        
        # Load and apply cleaning pipeline
        frame = ar.read_csv(csv_path)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        
        output_path = tmp_path / "clean.parquet"
        ar.write_parquet(clean, output_path)
        
        # Verify output
        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert df["name"].tolist() == ["alice", "bob"]
        assert df["score"].tolist() == [95, 87]

    def test_write_parquet_after_validation(self, tmp_path):
        """Write Parquet after validating data against schema."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "user_id,email\n"
            "1,alice@example.com\n"
            "2,bob@example.com\n"
        )
        
        frame = ar.read_csv(csv_path)
        
        # Validate against schema
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)
        assert result.passed
        
        # Write validated data
        output_path = tmp_path / "valid.parquet"
        ar.write_parquet(frame, output_path)
        assert output_path.exists()
        
        df = pd.read_parquet(output_path)
        assert len(df) == 2


class TestWriteParquetRoundTrip:
    """Test round-trip: write to Parquet and read back."""

    def test_write_read_roundtrip_preserves_data(self, tmp_path):
        """Write and read back preserves data integrity."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,value\n1,alice,10.5\n2,bob,20.3\n")
        
        # Write via arnio
        original = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(original, output_path)
        
        # Read back and compare
        df_original = ar.to_pandas(original)
        df_roundtrip = pd.read_parquet(output_path)
        
        pd.testing.assert_frame_equal(
            df_original.reset_index(drop=True),
            df_roundtrip.reset_index(drop=True),
        )

    def test_write_read_roundtrip_mixed_types(self, tmp_path):
        """Round-trip with mixed data types."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "int_col,float_col,str_col\n"
            "1,1.5,hello\n"
            "2,2.5,world\n"
            "3,3.5,test\n"
        )
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, output_path, compression="zstd")
        
        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert len(df.columns) == 3
        assert df["str_col"].tolist() == ["hello", "world", "test"]


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_write_parquet_empty_frame(self, tmp_path):
        """Write frame with headers but no data rows."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, output_path)
        
        df = pd.read_parquet(output_path)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]

    def test_write_parquet_pathlib_path(self, tmp_path):
        """Write using Path object (not just string)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_path))
        output_path = tmp_path / "output.parquet"
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 2

    def test_write_parquet_case_insensitive_extension(self, tmp_path):
        """File extension check is case-insensitive."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(csv_path)
        output_path = tmp_path / "output.PARQUET"
        ar.write_parquet(frame, output_path)
        
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 1
