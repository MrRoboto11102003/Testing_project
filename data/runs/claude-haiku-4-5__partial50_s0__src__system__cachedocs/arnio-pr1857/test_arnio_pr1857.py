import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import sys
from pathlib import Path


class TestFromDictBasic:
    """Basic functionality tests for from_dict"""
    
    def test_simple_dict_conversion(self):
        """Test basic dict to ArFrame conversion"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}
        
        df = ar.to_pandas(frame)
        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["age"].tolist() == [25, 30]
    
    def test_dict_with_integers(self):
        """Test dict with integer columns"""
        data = {"id": [1, 2, 3], "count": [10, 20, 30]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert df["id"].tolist() == [1, 2, 3]
        assert df["count"].tolist() == [10, 20, 30]
    
    def test_dict_with_floats(self):
        """Test dict with float data"""
        data = {"value": [1.5, 2.5, 3.5]}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert np.allclose(df["value"].tolist(), [1.5, 2.5, 3.5])
    
    def test_dict_with_mixed_types_per_column(self):
        """Test dict where pandas auto-coerces mixed numeric types"""
        data = {"values": [1, 2.5, 3]}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert len(df) == 3
        assert frame.shape[0] == 3


class TestFromDictNullHandling:
    """Tests for None/null value handling"""
    
    def test_dict_with_none_values(self):
        """Test dict containing None values"""
        data = {"name": ["Alice", None, "Bob"], "age": [25, 30, None]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert pd.isna(df["name"].iloc[1])
        assert pd.isna(df["age"].iloc[2])
    
    def test_dict_all_nulls_in_column(self):
        """Test column with all None values"""
        data = {"col1": [1, 2, 3], "col2": [None, None, None]}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert df["col2"].isna().all()


class TestFromDictEdgeCases:
    """Edge cases and boundary conditions"""
    
    def test_empty_dict(self):
        """Test empty dictionary returns empty ArFrame"""
        frame = ar.from_dict({})
        assert frame.shape == (0, 0)
    
    def test_single_row(self):
        """Test dict with single row"""
        data = {"name": ["Alice"], "age": [25]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (1, 2)
    
    def test_single_column(self):
        """Test dict with single column"""
        data = {"values": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 1)
        assert frame.columns == ["values"]
    
    def test_numpy_array_values(self):
        """Test dict with numpy arrays as column values"""
        data = {
            "integers": np.array([1, 2, 3]),
            "floats": np.array([4.0, 5.0, 6.0])
        }
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert df["integers"].tolist() == [1, 2, 3]
    
    def test_tuple_values(self):
        """Test dict with tuples as column values"""
        data = {"col": (10, 20, 30)}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert df["col"].tolist() == [10, 20, 30]


class TestFromDictWithPipeline:
    """Test from_dict integration with cleaning pipelines"""
    
    def test_from_dict_pipeline_strip_whitespace(self):
        """Test applying strip_whitespace to from_dict result"""
        data = {"name": [" Alice ", " Bob ", " Charlie "]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(clean)
        
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
    
    def test_from_dict_pipeline_drop_duplicates(self):
        """Test deduplication on from_dict result"""
        data = {
            "id": [1, 2, 2, 3],
            "name": ["Alice", "Bob", "Bob", "Charlie"]
        }
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"}),
        ])
        
        assert clean.shape[0] == 3
    
    def test_from_dict_pipeline_validation(self):
        """Test schema validation on from_dict result"""
        data = {
            "id": [1, 2, 3],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"]
        }
        frame = ar.from_dict(data)
        
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0


class TestFromDictPandas:
    """Test from_dict with pandas integration"""
    
    def test_from_dict_to_pandas_roundtrip(self):
        """Test converting from_dict to pandas and back"""
        data = {"values": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]}
        frame1 = ar.from_dict(data)
        df = ar.to_pandas(frame1)
        frame2 = ar.from_pandas(df)
        df2 = ar.to_pandas(frame2)
        
        pd.testing.assert_frame_equal(df, df2)
    
    def test_from_dict_profile(self):
        """Test profiling a from_dict frame"""
        data = {
            "id": [1, 2, 3, 4, 5],
            "value": [10.5, 20.3, None, 40.1, 50.2],
            "category": ["A", "B", "A", None, "B"]
        }
        frame = ar.from_dict(data)
        
        report = ar.profile(frame)
        summary = report.summary()
        
        assert report.row_count == 5
        assert report.column_count == 3
        assert "id" in summary["columns"]
        assert "value" in summary["columns"]


class TestFromDictFileIO:
    """Test from_dict with file I/O operations"""
    
    def test_from_dict_write_read_csv_roundtrip(self, tmp_path):
        """Test writing from_dict to CSV and reading it back"""
        data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 28]}
        frame1 = ar.from_dict(data)
        
        output_file = tmp_path / "output.csv"
        ar.write_csv(frame1, str(output_file))
        assert output_file.exists()
        
        # Read back and verify
        frame2 = ar.read_csv(str(output_file))
        df1 = ar.to_pandas(frame1)
        df2 = ar.to_pandas(frame2)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_from_dict_write_json(self, tmp_path):
        """Test writing from_dict to JSON"""
        data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        frame = ar.from_dict(data)
        
        output_file = tmp_path / "output.json"
        ar.write_json(frame, str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "id" in content
        assert "name" in content


class TestFromDictErrorHandling:
    """Error cases and input validation"""
    
    def test_non_dict_input_raises_typeerror(self):
        """Test that non-dict input raises TypeError"""
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict([1, 2, 3])
        
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict("not a dict")
        
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict(42)
    
    def test_non_string_keys_raises_typeerror(self):
        """Test that non-string dictionary keys raise TypeError"""
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({1: [1, 2, 3], 2: [4, 5, 6]})
        
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({("tuple",): [1, 2, 3]})
    
    def test_mismatched_column_lengths_raises_valueerror(self):
        """Test that mismatched column lengths raise ValueError"""
        data = {"col1": [1, 2, 3], "col2": [4, 5], "col3": [6]}
        
        with pytest.raises(ValueError, match="column lengths differ"):
            ar.from_dict(data)
    
    def test_nested_dict_raises_valueerror(self):
        """Test that nested dictionaries raise ValueError"""
        data = {"col": [{"nested": "value"}]}
        
        with pytest.raises(ValueError, match="Nested objects are not supported"):
            ar.from_dict(data)
    
    def test_string_as_column_value_raises_typeerror(self):
        """Test that string (not list) raises TypeError"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"col": "just a string"})
    
    def test_bytes_as_column_value_raises_typeerror(self):
        """Test that bytes (not list) raises TypeError"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"col": b"bytes data"})
    
    def test_scalar_value_raises_typeerror(self):
        """Test that scalar values raise TypeError"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"col": 42})
    
    def test_dict_with_boolean_values(self):
        """Test dict with boolean columns (valid case)"""
        data = {"active": [True, False, True], "id": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert df["active"].tolist() == [True, False, True]


class TestFromDictAdvancedWorkflows:
    """Advanced real-world workflows"""
    
    def test_from_dict_large_dataset(self):
        """Test from_dict with larger dataset"""
        data = {
            "id": list(range(1000)),
            "value": [x * 1.5 for x in range(1000)],
            "category": ["A" if i % 2 == 0 else "B" for i in range(1000)]
        }
        frame = ar.from_dict(data)
        
        assert frame.shape == (1000, 3)
        df = ar.to_pandas(frame)
        assert len(df) == 1000
    
    def test_from_dict_to_pandas_copy_parameter(self):
        """Test to_pandas copy parameter with from_dict"""
        data = {"values": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        df1 = ar.to_pandas(frame, copy=False)
        df2 = ar.to_pandas(frame, copy=True)
        
        assert len(df1) == 3
        assert len(df2) == 3
        assert df1.equals(df2)
    
    def test_from_dict_consistency_with_from_pandas(self):
        """Test that from_dict and from_pandas produce equivalent results"""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 28],
            "score": [95.5, 87.3, 91.2]
        }
        
        frame1 = ar.from_dict(data)
        df1 = ar.to_pandas(frame1)
        
        pdf = pd.DataFrame(data)
        frame2 = ar.from_pandas(pdf)
        df2 = ar.to_pandas(frame2)
        
        pd.testing.assert_frame_equal(df1, df2)
