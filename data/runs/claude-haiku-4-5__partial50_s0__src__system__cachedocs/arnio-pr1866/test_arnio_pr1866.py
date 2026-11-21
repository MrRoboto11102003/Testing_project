import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path

import arnio as ar

# Skip all tests if pyarrow is not available
pyarrow = pytest.importorskip("pyarrow")


class TestWriteParquetBasicRoundtrip:
    """E2E: Load, process, and write CSV data to Parquet."""

    def test_csv_to_parquet_simple(self, tmp_path):
        """Load CSV, write to Parquet, read back and verify."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")

        # Load and write
        frame = ar.read_csv(str(csv_file))
        parquet_file = tmp_path / "output.parquet"
        ar.write_parquet(frame, parquet_file)

        # Verify output file exists and can be read
        assert parquet_file.exists()
        result_df = pd.read_parquet(str(parquet_file))
        assert result_df.shape == (3, 3)
        assert list(result_df.columns) == ["id", "name", "value"]
        assert result_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_pq_extension_accepted(self, tmp_path):
        """Write to .pq file extension (alternative to .parquet)."""
        frame = ar.from_dict({"id": ["1", "2"], "value": ["100", "200"]})
        pq_file = tmp_path / "output.pq"

        ar.write_parquet(frame, pq_file)

        assert pq_file.exists()
        result = pd.read_parquet(str(pq_file))
        assert result.shape == (2, 2)

    def test_pandas_dataframe_via_arnio_to_parquet(self, tmp_path):
        """Convert pandas DataFrame to Parquet via Arnio."""
        pdf = pd.DataFrame({
            "user_id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "score": [95.5, 88.2, 92.1],
        })

        frame = ar.from_pandas(pdf)
        parquet_file = tmp_path / "output.parquet"
        ar.write_parquet(frame, parquet_file)

        result_df = pd.read_parquet(str(parquet_file))
        assert len(result_df) == 3
        assert "email" in result_df.columns


class TestWriteParquetCompressionAndRowGroups:
    """Test compression codecs and row group size parameters."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "brotli", "zstd", "none"])
    def test_all_compression_codecs(self, tmp_path, codec):
        """Write with each supported compression codec."""
        frame = ar.from_dict({
            "id": [str(i) for i in range(30)],
            "value": [str(i * 10) for i in range(30)],
        })

        parquet_file = tmp_path / f"output_{codec}.parquet"
        ar.write_parquet(frame, parquet_file, compression=codec)

        assert parquet_file.exists()
        result = pd.read_parquet(str(parquet_file))
        assert len(result) == 30

    def test_invalid_compression_codec_raises_valueerror(self, tmp_path):
        """Reject invalid compression codec with clear error."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(ValueError) as exc_info:
            ar.write_parquet(frame, parquet_file, compression="unsupported_codec")

        assert "Unknown compression codec" in str(exc_info.value)
        assert "unsupported_codec" in str(exc_info.value)

    def test_explicit_row_group_size(self, tmp_path):
        """Explicit row_group_size parameter is applied."""
        frame = ar.from_dict({
            "id": [str(i) for i in range(100)],
        })
        parquet_file = tmp_path / "output.parquet"

        ar.write_parquet(frame, parquet_file, row_group_size=25)

        assert parquet_file.exists()
        result = pd.read_parquet(str(parquet_file))
        assert len(result) == 100

    @pytest.mark.parametrize("bad_size", [0, -1, -100])
    def test_invalid_row_group_size_raises(self, tmp_path, bad_size):
        """row_group_size must be positive; <= 0 raises ValueError."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(frame, parquet_file, row_group_size=bad_size)


class TestWriteParquetFileExtensionValidation:
    """Test file extension validation and path handling."""

    @pytest.mark.parametrize("bad_ext", ["output.csv", "output.txt", "output.json"])
    def test_invalid_extensions_rejected(self, tmp_path, bad_ext):
        """Non-.parquet/.pq files are rejected."""
        frame = ar.from_dict({"x": ["1"]})
        bad_file = tmp_path / bad_ext

        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, bad_file)

    def test_extension_check_is_case_insensitive(self, tmp_path):
        """File extension check accepts .PARQUET, .Parquet, .pQ, etc."""
        frame = ar.from_dict({"x": ["1"]})

        for ext in ["output.PARQUET", "output.Parquet", "output.PQ"]:
            parquet_file = tmp_path / ext
            ar.write_parquet(frame, parquet_file)
            assert parquet_file.exists()

    def test_pathlib_path_object(self, tmp_path):
        """Accept pathlib.Path objects."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        ar.write_parquet(frame, parquet_file)  # Pass Path directly, not str()
        assert parquet_file.exists()


class TestWriteParquetInputValidation:
    """Test input type and parameter validation."""

    def test_non_arframe_input_raises_typeerror(self, tmp_path):
        """Passing a pandas DataFrame (not ArFrame) raises TypeError."""
        pdf = pd.DataFrame({"x": [1, 2, 3]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(pdf, parquet_file)

    def test_non_string_compression_raises_typeerror(self, tmp_path):
        """compression parameter must be a string."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(frame, parquet_file, compression=123)

    def test_non_bool_preserve_attrs_raises_typeerror(self, tmp_path):
        """preserve_attrs parameter must be a boolean."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, parquet_file, preserve_attrs="yes")

    def test_non_int_row_group_size_raises_typeerror(self, tmp_path):
        """row_group_size must be an int (not str, bool, float)."""
        frame = ar.from_dict({"x": ["1"]})
        parquet_file = tmp_path / "output.parquet"

        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, parquet_file, row_group_size="50")


class TestWriteParquetWithCleaningPipeline:
    """E2E: Load messy data, clean with pipeline, write to Parquet."""

    def test_csv_pipeline_to_parquet_complete_workflow(self, tmp_path):
        """Full workflow: messy CSV → pipeline → Parquet."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "id,name,score\n1, Alice ,95.5\n2, Bob ,87.3\n1, Alice ,95.5\n3, Charlie ,92.1\n"
        )

        # Load, clean, and write
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        parquet_file = tmp_path / "output.parquet"
        ar.write_parquet(clean, parquet_file)

        # Verify results
        result_df = pd.read_parquet(str(parquet_file))
        assert len(result_df) == 3  # Duplicate removed
        assert result_df.iloc[0]["name"] == "Alice"  # Whitespace stripped
        assert result_df.iloc[0]["score"] in ["95.5", 95.5]  # Type preserved


class TestWriteParquetWithSchemaValidation:
    """E2E: Validate frame against schema, then write to Parquet."""

    def test_validate_then_write_to_parquet(self, tmp_path):
        """Validate frame with schema, confirm pass, write to Parquet."""
        frame = ar.from_dict({
            "id": ["1", "2", "3"],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
        })

        schema = ar.Schema({
            "id": ar.String(nullable=False),
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(frame, schema)
        assert result.passed

        # Write validated frame to Parquet
        parquet_file = tmp_path / "validated.parquet"
        ar.write_parquet(frame, parquet_file)

        assert parquet_file.exists()
        output_df = pd.read_parquet(str(parquet_file))
        assert len(output_df) == 3
        assert list(output_df.columns) == ["id", "email"]


class TestWriteParquetWithPandasAccessor:
    """E2E: Use pandas .arnio accessor to clean, then write Parquet."""

    def test_pandas_accessor_clean_to_parquet(self, tmp_path):
        """Use df.arnio.clean() accessor, convert to ArFrame, write Parquet."""
        pdf = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": [" Alice ", " Bob ", " Bob ", " Charlie "],
        })

        # Clean using accessor
        clean_df = pdf.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True,
        )

        # Convert and write
        frame = ar.from_pandas(clean_df)
        parquet_file = tmp_path / "output.parquet"
        ar.write_parquet(frame, parquet_file, compression="gzip")

        result_df = pd.read_parquet(str(parquet_file))
        assert len(result_df) == 3  # Duplicates removed
        assert result_df.iloc[0]["name"] == "Alice"  # Whitespace stripped


class TestWriteParquetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame_zero_rows_with_columns(self, tmp_path):
        """Write an empty frame (0 rows but columns defined)."""
        frame = ar.from_dict({"name": [], "age": [], "score": []})
        parquet_file = tmp_path / "empty.parquet"

        ar.write_parquet(frame, parquet_file)

        assert parquet_file.exists()
        result = pd.read_parquet(str(parquet_file))
        assert len(result) == 0
        assert list(result.columns) == ["name", "age", "score"]

    def test_single_row_frame(self, tmp_path):
        """Write a frame with exactly one row."""
        frame = ar.from_dict({
            "id": ["1"],
            "name": ["Alice"],
            "value": ["100"],
        })
        parquet_file = tmp_path / "single.parquet"

        ar.write_parquet(frame, parquet_file)

        result = pd.read_parquet(str(parquet_file))
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_many_columns(self, tmp_path):
        """Write a frame with many columns."""
        data = {f"col_{i:03d}": [str(i)] for i in range(50)}
        frame = ar.from_dict(data)
        parquet_file = tmp_path / "many_cols.parquet"

        ar.write_parquet(frame, parquet_file)

        result = pd.read_parquet(str(parquet_file))
        assert len(result.columns) == 50
        assert len(result) == 1
