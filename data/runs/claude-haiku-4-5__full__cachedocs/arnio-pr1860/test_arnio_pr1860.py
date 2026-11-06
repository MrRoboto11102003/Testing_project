import pytest
import pandas as pd
import os
from pathlib import Path

import arnio as ar


class TestWriteParquetBasic:
    """Basic write_parquet functionality."""

    def test_creates_parquet_file(self, tmp_path):
        """Write creates a .parquet file."""
        frame = ar.from_pandas(
            pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        )
        output_path = tmp_path / "output.parquet"

        ar.write_parquet(frame, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pq_extension_supported(self, tmp_path):
        """Write accepts .pq extension."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / "output.pq"

        ar.write_parquet(frame, output_path)

        assert output_path.exists()

    def test_roundtrip_preserves_data(self, tmp_path):
        """Data survives write and read back."""
        df_original = pd.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.5, 2.5, 3.5],
                "str": ["a", "b", "c"],
            }
        )
        frame = ar.from_pandas(df_original)
        output_path = tmp_path / "roundtrip.parquet"

        ar.write_parquet(frame, output_path)
        df_read = pd.read_parquet(output_path)

        pd.testing.assert_frame_equal(df_original, df_read)

    def test_accepts_string_path(self, tmp_path):
        """String paths work."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = str(tmp_path / "output.parquet")

        ar.write_parquet(frame, output_path)

        assert os.path.exists(output_path)


class TestWriteParquetCompression:
    """Compression codec parameter."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_all_compression_codecs_accepted(self, tmp_path, codec):
        """All documented compression codecs work."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / f"output_{codec}.parquet"

        ar.write_parquet(frame, output_path, compression=codec)

        assert output_path.exists()

    def test_invalid_compression_raises_valueerror(self, tmp_path):
        """Unknown compression codec raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError):
            ar.write_parquet(frame, output_path, compression="unknown_codec")


class TestWriteParquetRowGroupSize:
    """row_group_size parameter."""

    def test_row_group_size_positive_integer(self, tmp_path):
        """Positive row_group_size is accepted."""
        frame = ar.from_pandas(pd.DataFrame({"a": range(100)}))
        output_path = tmp_path / "output.parquet"

        ar.write_parquet(frame, output_path, row_group_size=25)

        assert output_path.exists()

    def test_row_group_size_none_uses_default(self, tmp_path):
        """row_group_size=None uses pyarrow default."""
        frame = ar.from_pandas(pd.DataFrame({"a": range(100)}))
        output_path = tmp_path / "output.parquet"

        ar.write_parquet(frame, output_path, row_group_size=None)

        assert output_path.exists()

    def test_row_group_size_zero_raises_valueerror(self, tmp_path):
        """Zero row_group_size raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError):
            ar.write_parquet(frame, output_path, row_group_size=0)

    def test_row_group_size_negative_raises_valueerror(self, tmp_path):
        """Negative row_group_size raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError):
            ar.write_parquet(frame, output_path, row_group_size=-1)


class TestWriteParquetPreserveAttrs:
    """preserve_attrs parameter."""

    def test_preserve_attrs_json_serializable(self, tmp_path):
        """JSON-serializable attrs preserved when preserve_attrs=True."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"key": "value", "num": 42, "list": [1, 2]}
        frame = ar.from_pandas(df)
        output_path = tmp_path / "output.parquet"

        ar.write_parquet(frame, output_path, preserve_attrs=True)

        assert output_path.exists()

    def test_preserve_attrs_false_drops_silently(self, tmp_path):
        """preserve_attrs=False silently ignores attrs."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"key": "value"}
        frame = ar.from_pandas(df)
        output_path = tmp_path / "output.parquet"

        ar.write_parquet(frame, output_path, preserve_attrs=False)

        assert output_path.exists()

    def test_non_serializable_attrs_raise_typeerror(self, tmp_path):
        """Non-JSON-serializable attrs raise TypeError with preserve_attrs=True."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"key": object()}
        frame = ar.from_pandas(df)
        output_path = tmp_path / "output.parquet"

        with pytest.raises(TypeError):
            ar.write_parquet(frame, output_path, preserve_attrs=True)

    def test_preserve_attrs_non_boolean_raises_typeerror(self, tmp_path):
        """Non-boolean preserve_attrs raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / "output.parquet"

        with pytest.raises(TypeError):
            ar.write_parquet(frame, output_path, preserve_attrs="yes")


class TestWriteParquetFileExtension:
    """File extension validation."""

    @pytest.mark.parametrize("ext", ["csv", "txt", "json", "feather"])
    def test_invalid_extensions_raise_valueerror(self, tmp_path, ext):
        """Non-parquet extensions raise ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        output_path = tmp_path / f"output.{ext}"

        with pytest.raises(ValueError):
            ar.write_parquet(frame, output_path)
