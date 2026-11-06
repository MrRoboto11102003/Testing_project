import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test arnio.convert.to_pandas function."""

    def test_basic_csv_conversion(self, tmp_path):
        """Test basic conversion from CSV -> ArFrame -> pandas."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name,score\n1,alice,95.5\n2,bob,87.0\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name", "score"]
        assert len(df) == 2

    def test_int64_column_conversion(self):
        """Test int64 column is converted to pd.Int64Dtype."""
        frame = ar.from_dict({"nums": [1, 2, 3, 4, 5]})
        df = ar.to_pandas(frame)
        
        assert df["nums"].dtype == pd.Int64Dtype()
        assert list(df["nums"]) == [1, 2, 3, 4, 5]

    def test_float64_column_conversion(self):
        """Test float64 column is converted to numpy float64."""
        frame = ar.from_dict({"vals": [1.5, 2.7, 3.14]})
        df = ar.to_pandas(frame)
        
        assert df["vals"].dtype == np.float64
        np.testing.assert_array_almost_equal(df["vals"], [1.5, 2.7, 3.14])

    def test_bool_column_conversion(self):
        """Test bool column is converted to pd.BooleanDtype."""
        frame = ar.from_dict({"flags": [True, False, True]})
        df = ar.to_pandas(frame)
        
        assert df["flags"].dtype == pd.BooleanDtype()
        assert list(df["flags"]) == [True, False, True]

    def test_string_column_conversion(self):
        """Test string column is converted to pd.StringDtype."""
        frame = ar.from_dict({"names": ["alice", "bob", "charlie"]})
        df = ar.to_pandas(frame)
        
        assert df["names"].dtype == pd.StringDtype()
        assert list(df["names"]) == ["alice", "bob", "charlie"]

    def test_null_handling_mixed_dtypes(self):
        """Test null handling across multiple dtype columns."""
        df_input = pd.DataFrame({
            "ints": pd.array([1, None, 3], dtype="Int64"),
            "floats": [1.0, np.nan, 3.0],
            "bools": pd.array([True, None, False], dtype="boolean"),
            "strs": pd.array(["a", None, "c"], dtype="string"),
        })
        frame = ar.from_pandas(df_input)
        df = ar.to_pandas(frame)
        
        # Check nulls are preserved
        assert df["ints"].isna().sum() == 1
        assert df["floats"].isna().sum() == 1
        assert df["bools"].isna().sum() == 1
        assert df["strs"].isna().sum() == 1
        
        # Check non-null values
        assert df["ints"][0] == 1
        assert df["floats"][2] == 3.0
        assert df["bools"][0] == True
        assert df["strs"][2] == "c"

    def test_mixed_dtypes_single_frame(self):
        """Test frame with all supported dtypes."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 87.0, 92.3],
            "active": [True, False, True]
        })
        df = ar.to_pandas(frame)
        
        assert df["id"].dtype == pd.Int64Dtype()
        assert df["name"].dtype == pd.StringDtype()
        assert df["score"].dtype == np.float64
        assert df["active"].dtype == pd.BooleanDtype()
        assert len(df) == 3

    def test_copy_parameter_false(self):
        """Test copy=False preserves zero-copy path where possible."""
        frame = ar.from_dict({"nums": [1, 2, 3, 4, 5]})
        df = ar.to_pandas(frame, copy=False)
        
        assert len(df) == 5
        assert df["nums"].dtype == pd.Int64Dtype()

    def test_copy_parameter_true(self):
        """Test copy=True creates defensive copies."""
        frame = ar.from_dict({"vals": [1.5, 2.5, 3.5]})
        df = ar.to_pandas(frame, copy=True)
        
        assert len(df) == 3
        assert df["vals"].dtype == np.float64

    def test_empty_frame_conversion(self):
        """Test conversion of empty ArFrame."""
        frame = ar.from_dict({"col1": [], "col2": []})
        df = ar.to_pandas(frame)
        
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_attrs_preservation_from_pandas(self):
        """Test DataFrame.attrs are restored after round-trip."""
        df_input = pd.DataFrame({"nums": [1, 2, 3]})
        df_input.attrs = {"source": "test", "version": 1}
        
        frame = ar.from_pandas(df_input)
        df = ar.to_pandas(frame)
        
        assert df.attrs == {"source": "test", "version": 1}

    def test_column_order_preserved(self):
        """Test column order is preserved in conversion."""
        frame = ar.from_dict({
            "zebra": [1, 2],
            "apple": [3, 4],
            "middle": [5, 6]
        })
        df = ar.to_pandas(frame)
        
        assert list(df.columns) == ["zebra", "apple", "middle"]

    def test_type_error_non_arframe_input(self):
        """Test TypeError raised for non-ArFrame input."""
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas({"not": "an arframe"})
        
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas([1, 2, 3])

    def test_type_error_copy_not_bool(self):
        """Test TypeError raised when copy is not a bool."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="True")
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)

    def test_large_int64_boundary_values(self):
        """Test conversion with max/min int64 values."""
        large_nums = [9223372036854775806, -9223372036854775807, 0]
        frame = ar.from_dict({"big": large_nums})
        df = ar.to_pandas(frame)
        
        assert list(df["big"]) == large_nums
