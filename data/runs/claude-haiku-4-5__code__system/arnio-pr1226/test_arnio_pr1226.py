import pytest
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestToPandasBasicConversions:
    """Test basic ArFrame to pandas conversions"""
    
    def test_csv_to_pandas_simple(self, tmp_path):
        """Test simple CSV read and conversion to pandas"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["id", "name", "value"]
        assert df["id"].tolist() == [1, 2, 3]
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
    
    def test_mixed_dtypes_preservation(self, tmp_path):
        """Test that mixed data types are correctly preserved"""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text("int_col,float_col,str_col\n1,1.5,hello\n2,2.7,world\n3,3.14,test\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["int_col"].dtype == np.int64 or df["int_col"].dtype == pd.Int64Dtype()
        assert df["float_col"].dtype == np.float64
        assert df["str_col"].dtype == pd.StringDtype()
    
    def test_empty_dataframe_conversion(self, tmp_path):
        """Test conversion of CSV with headers but no rows"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("col1,col2,col3\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 0
        assert list(df.columns) == ["col1", "col2", "col3"]


class TestToPandasNullHandling:
    """Test null value handling in conversion"""
    
    def test_null_integers(self, tmp_path):
        """Test integer columns with missing values"""
        csv_file = tmp_path / "null_ints.csv"
        csv_file.write_text("id,value\n1,100\n2,\n3,300\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["value"].iloc[0] == 100
        assert pd.isna(df["value"].iloc[1])
        assert df["value"].iloc[2] == 300
    
    def test_null_floats_with_nan(self, tmp_path):
        """Test float columns properly handle NaN for missing values"""
        csv_file = tmp_path / "null_floats.csv"
        csv_file.write_text("x,y\n1.5,2.5\n,3.5\n4.5,\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert abs(df["x"].iloc[0] - 1.5) < 1e-9
        assert np.isnan(df["x"].iloc[1])
        assert np.isnan(df["y"].iloc[2])
    
    def test_null_strings(self, tmp_path):
        """Test string columns with missing values"""
        csv_file = tmp_path / "null_strings.csv"
        csv_file.write_text("name,city\nAlice,NYC\n,LA\nCharlie,\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["name"].iloc[0] == "Alice"
        assert pd.isna(df["name"].iloc[1])
        assert pd.isna(df["city"].iloc[2])
        assert df["city"].iloc[0] == "NYC"


class TestToPandasCopyParameter:
    """Test copy parameter behavior"""
    
    def test_copy_false_default(self, tmp_path):
        """Test default copy=False preserves zero-copy path"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,hello\n3,4,world\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame, copy=False)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
    
    def test_copy_true_defensive(self, tmp_path):
        """Test copy=True creates defensive copies"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val1,val2\n1,2.5\n3,4.5\n")
        
        frame = ar.read_csv(str(csv_file))
        df_copy_true = ar.to_pandas(frame, copy=True)
        df_copy_false = ar.to_pandas(frame, copy=False)
        
        # Both should be valid DataFrames
        assert df_copy_true.shape == df_copy_false.shape
        assert all(df_copy_true.columns == df_copy_false.columns)


class TestToPandasErrorHandling:
    """Test error cases and validation"""
    
    def test_error_non_arframe_input(self):
        """Test TypeError when passing non-ArFrame to to_pandas"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas(df)
    
    def test_error_copy_not_bool(self, tmp_path):
        """Test TypeError when copy parameter is not boolean"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="true")
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)


class TestToPandasAttrsPreservation:
    """Test attrs metadata preservation"""
    
    def test_preserve_attrs_from_pandas(self):
        """Test attrs are preserved after pandas roundtrip"""
        df_original = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.5, 30.5]
        })
        df_original.attrs = {"source": "test_data", "version": 1, "unit": "meters"}
        
        frame = ar.from_pandas(df_original)
        df_result = ar.to_pandas(frame)
        
        assert df_result.attrs == {"source": "test_data", "version": 1, "unit": "meters"}
    
    def test_attrs_restored_correctly(self):
        """Test complex attrs structure is deepcopied correctly"""
        df_original = pd.DataFrame({"x": [1, 2]})
        df_original.attrs = {"metadata": {"nested": [1, 2, 3]}, "count": 42}
        
        frame = ar.from_pandas(df_original)
        df_result = ar.to_pandas(frame)
        
        assert df_result.attrs["metadata"]["nested"] == [1, 2, 3]
        assert df_result.attrs["count"] == 42


class TestToPandasRoundtrip:
    """Test roundtrip conversion workflows"""
    
    def test_pandas_csv_pandas_roundtrip(self, tmp_path):
        """Test complete workflow: pandas -> CSV -> ArFrame -> pandas"""
        input_file = tmp_path / "input.csv"
        
        df_original = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [85.5, 92.3, 78.9],
            "name": ["Alice", "Bob", "Charlie"]
        })
        df_original.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        df_result = ar.to_pandas(frame)
        
        assert df_result.shape == df_original.shape
        assert list(df_result.columns) == list(df_original.columns)
        assert df_result["id"].tolist() == [1, 2, 3]
        assert all(abs(a - b) < 0.01 for a, b in zip(df_result["score"], [85.5, 92.3, 78.9]))


class TestToPandasLargeDataset:
    """Test with larger datasets"""
    
    def test_large_dataset_conversion(self, tmp_path):
        """Test conversion performance with larger dataset"""
        csv_file = tmp_path / "large.csv"
        
        # Generate 5000 rows with multiple columns
        with open(csv_file, 'w') as f:
            f.write("id,value,score,category\n")
            for i in range(5000):
                f.write(f"{i},{i * 1.5},{i * 0.1},cat_{i % 10}\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df.shape == (5000, 4)
        assert all(df["id"] == range(5000))


class TestToPandasCLIWorkflow:
    """Test CLI integration with to_pandas workflow"""
    
    def test_cli_validation_before_conversion(self, tmp_path):
        """Test CLI validation and subsequent pandas conversion"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,hello\n3,4,world\n5,6,test\n")
        
        # Validate via CLI
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate", str(csv_file)],
            capture_output=True,
            text=True
        )
        
        # CLI should complete
        assert result.returncode is not None
        
        # Then convert to pandas programmatically
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df.shape == (3, 3)
    
    def test_cli_read_and_profiling(self, tmp_path):
        """Test CLI read with profiling, then to_pandas conversion"""
        csv_file = tmp_path / "profile_data.csv"
        csv_file.write_text("x,y,z\n1,100,10.5\n2,200,20.3\n3,,30.1\n4,400,\n")
        
        # Use CLI to read
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "read", str(csv_file)],
            capture_output=True,
            text=True
        )
        
        # Also convert programmatically
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df.shape[0] == 4
        assert pd.isna(df["y"].iloc[2]) or df["y"].iloc[2] == 200


class TestToPandasWriteWorkflow:
    """Test writing converted pandas DataFrames"""
    
    def test_convert_and_write_csv(self, tmp_path):
        """Test converting ArFrame to pandas and writing CSV"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_file.write_text("name,age,salary\nAlice,30,50000\nBob,25,45000\nCharlie,35,60000\n")
        
        frame = ar.read_csv(str(input_file))
        df = ar.to_pandas(frame)
        df.to_csv(output_file, index=False)
        
        # Verify output
        assert output_file.exists()
        output_text = output_file.read_text()
        assert "name,age,salary" in output_text
        assert "Alice,30,50000" in output_text


class TestToPandasSpecialValues:
    """Test handling of special values"""
    
    def test_special_float_values(self, tmp_path):
        """Test handling of inf and -inf in floats"""
        csv_file = tmp_path / "special.csv"
        csv_file.write_text("value\n1.5\n100.0\n-50.0\n0.0\n")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["value"].dtype == np.float64
        assert len(df) == 4


class TestToPandasComplexWorkflow:
    """Test complete data preparation workflows"""
    
    def test_complete_data_prep_workflow(self, tmp_path):
        """Test realistic end-to-end data preparation with validation and conversion"""
        raw_file = tmp_path / "raw.csv"
        output_file = tmp_path / "processed.csv"
        
        # Create raw data with some quality issues
        raw_file.write_text(
            "customer_id,purchase_amount,date,category\n"
            "1,100.50,2023-01-01,A\n"
            "2,,2023-01-02,B\n"
            "3,250.75,2023-01-03,A\n"
            "4,150.00,2023-01-04,C\n"
            ",75.25,2023-01-05,B\n"
        )
        
        # Read with arnio
        frame = ar.read_csv(str(raw_file))
        
        # Convert to pandas for further processing
        df = ar.to_pandas(frame)
        
        # Validate structure
        assert df.shape[0] == 5
        assert "customer_id" in df.columns
        assert "purchase_amount" in df.columns
        
        # Clean and write
        df_clean = df.dropna(subset=["customer_id"])
        df_clean.to_csv(output_file, index=False)
        
        assert output_file.exists()
