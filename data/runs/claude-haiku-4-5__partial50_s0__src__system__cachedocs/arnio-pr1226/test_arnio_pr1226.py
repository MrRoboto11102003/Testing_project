import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandasBasicConversion:
    """Test basic to_pandas conversion workflows."""
    
    def test_read_csv_to_pandas_basic(self, tmp_path):
        """Read CSV into ArFrame, convert to pandas DataFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95.5\n2,Bob,88.0\n3,Carol,92.5\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["id", "name", "score"]
        assert df["name"].tolist() == ["Alice", "Bob", "Carol"]

    def test_from_pandas_round_trip_preserves_attrs(self):
        """from_pandas -> to_pandas round trip preserves DataFrame.attrs."""
        original_df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        original_df.attrs = {"version": "1.0", "source": "test"}
        
        frame = ar.from_pandas(original_df)
        result_df = ar.to_pandas(frame)
        
        assert result_df.attrs == {"version": "1.0", "source": "test"}
        assert result_df.shape == original_df.shape

    def test_from_dict_to_pandas(self):
        """Create ArFrame from dict, convert to pandas."""
        data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert df.shape == (3, 3)
        assert df["a"].tolist() == [1, 2, 3]


class TestToPandasDtypeHandling:
    """Test dtype handling in to_pandas conversion."""
    
    def test_int64_with_nulls_to_Int64Dtype(self, tmp_path):
        """INT64 columns with nulls convert to Int64Dtype."""
        csv_file = tmp_path / "ints.csv"
        csv_file.write_text("id\n1\n\n3\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["id"].dtype == pd.Int64Dtype()
        assert pd.isna(df["id"].iloc[1])
        assert df["id"].iloc[0] == 1

    def test_float64_with_nulls_to_float64_with_nan(self, tmp_path):
        """FLOAT64 columns with nulls become float64 with NaN."""
        csv_file = tmp_path / "floats.csv"
        csv_file.write_text("value\n1.5\n\n3.5\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["value"].dtype == np.float64
        assert np.isnan(df["value"].iloc[1])
        assert df["value"].iloc[0] == 1.5

    def test_bool_with_nulls_to_BooleanDtype(self, tmp_path):
        """BOOL columns with nulls convert to BooleanDtype."""
        csv_file = tmp_path / "bools.csv"
        csv_file.write_text("flag\ntrue\n\nfalse\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["flag"].dtype == pd.BooleanDtype()
        assert pd.isna(df["flag"].iloc[1])
        assert df["flag"].iloc[0] is True

    def test_string_column_to_StringDtype(self, tmp_path):
        """String columns convert to StringDtype."""
        csv_file = tmp_path / "strings.csv"
        csv_file.write_text("text\nhello\nworld\ntest\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["text"].dtype == pd.StringDtype()
        assert df["text"].tolist() == ["hello", "world", "test"]


class TestToPandasCopyParameter:
    """Test copy parameter behavior."""
    
    def test_to_pandas_with_copy_false(self, tmp_path):
        """copy=False preserves zero-copy where supported."""
        csv_file = tmp_path / "numeric.csv"
        csv_file.write_text("a,b\n1,2.5\n3,4.5\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame, copy=False)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_to_pandas_with_copy_true(self, tmp_path):
        """copy=True forces defensive pandas-owned copies."""
        csv_file = tmp_path / "defensive.csv"
        csv_file.write_text("val\n100\n200\n300\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame, copy=True)
        
        assert df.shape == (3, 1)
        # Modifying should not affect original frame
        df_modified = df.copy()
        df_modified["val"] = df_modified["val"] * 2
        assert df_modified["val"].iloc[0] == 200


class TestToPandasPipelineWorkflows:
    """Test to_pandas in complete data cleaning workflows."""
    
    def test_pipeline_cleanup_to_pandas(self, tmp_path):
        """Read -> pipeline cleanup -> to_pandas workflow."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,score\n  Alice  ,95\n  Bob  ,88\n  Alice  ,95\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        df = ar.to_pandas(clean)
        
        assert df.shape[0] == 2
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_fill_nulls_pipeline_to_pandas(self, tmp_path):
        """fill_nulls pipeline step -> to_pandas."""
        csv_file = tmp_path / "incomplete.csv"
        csv_file.write_text("id,value\n1,100\n2,\n3,300\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0, "subset": ["value"]}),
        ])
        df = ar.to_pandas(clean)
        
        assert len(df) == 3
        assert df["value"].tolist() == [100.0, 0.0, 300.0]

    def test_auto_clean_to_pandas(self, tmp_path):
        """auto_clean workflow -> to_pandas."""
        csv_file = tmp_path / "auto_clean.csv"
        csv_file.write_text("name,city\n  Alice  ,  NYC  \n  Bob  ,  LA  \n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(clean)
        
        assert len(df) == 2
        assert df.shape[1] == 2


class TestToPandasValidationWorkflow:
    """Test to_pandas with schema validation workflows."""
    
    def test_validate_then_to_pandas(self, tmp_path):
        """Validate ArFrame, then convert to pandas."""
        csv_file = tmp_path / "validated.csv"
        csv_file.write_text("email,age\nalice@example.com,30\nbob@example.com,25\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        assert result.passed
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)


class TestToPandasEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_frame_to_pandas(self, tmp_path):
        """Convert empty ArFrame to empty pandas DataFrame."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("id,name,value\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert len(df) == 0
        assert list(df.columns) == ["id", "name", "value"]

    def test_single_row_to_pandas(self, tmp_path):
        """Convert single-row ArFrame to pandas."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("x,y,z\n1,2,3\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df.shape == (1, 3)

    def test_all_nulls_column_to_pandas(self, tmp_path):
        """Column with all nulls converts properly."""
        csv_file = tmp_path / "all_nulls.csv"
        csv_file.write_text("a,b,c\n1,,3\n2,,4\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert pd.isna(df["b"]).all()
        assert df["a"].tolist() == [1, 2]

    def test_copy_parameter_type_error(self):
        """TypeError when copy is not a bool."""
        frame = ar.from_dict({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="yes")

    def test_non_arframe_input_raises_typeerror(self):
        """TypeError when input is not an ArFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas(df)

    def test_mixed_dtypes_comprehensive(self, tmp_path):
        """Frame with all supported dtype combinations converts correctly."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text("id,name,score,active\n1,Alice,95.5,true\n2,Bob,88.0,false\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert len(df) == 2
        assert df.shape[1] == 4
        # Verify conversion succeeded with proper structure
        assert isinstance(df, pd.DataFrame)
        assert all(isinstance(col, str) for col in df.columns)
