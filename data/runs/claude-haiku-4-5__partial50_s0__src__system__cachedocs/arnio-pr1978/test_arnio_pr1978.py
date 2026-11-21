import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path


class TestSelectDtypesBasic:
    """Test basic select_dtypes functionality."""
    
    def test_select_dtypes_include_single_dtype(self, tmp_path):
        """Test including a single dtype."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,active\n1,alice,true\n2,bob,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include="int64")
        
        assert result.columns == ["id"]
        assert result.shape[0] == 2
        df = ar.to_pandas(result)
        assert list(df.columns) == ["id"]
    
    def test_select_dtypes_include_multiple_dtypes_as_list(self, tmp_path):
        """Test including multiple dtypes as list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,name,active\n1,95.5,alice,true\n2,88.3,bob,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=["int64", "float64"])
        
        assert set(result.columns) == {"id", "score"}
        assert result.columns[0] == "id"
        assert result.columns[1] == "score"
    
    def test_select_dtypes_include_multiple_dtypes_as_tuple(self, tmp_path):
        """Test including multiple dtypes as tuple."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,name\n1,95.5,alice\n2,88.3,bob\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=("int64", "float64"))
        
        assert set(result.columns) == {"id", "score"}
    
    def test_select_dtypes_exclude_single_dtype(self, tmp_path):
        """Test excluding a single dtype."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,active\n1,alice,true\n2,bob,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(exclude="string")
        
        assert set(result.columns) == {"id", "active"}
    
    def test_select_dtypes_exclude_multiple_dtypes(self, tmp_path):
        """Test excluding multiple dtypes."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,name,active\n1,95.5,alice,true\n2,88.3,bob,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(exclude=["string", "bool"])
        
        assert set(result.columns) == {"id", "score"}
    
    def test_select_dtypes_preserve_column_order(self, tmp_path):
        """Test that column order is preserved."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("z,a,m,b\n1,2.5,test,true\n2,3.5,foo,false\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=["int64", "float64"])
        
        assert result.columns == ["z", "a"]


class TestSelectDtypesValidation:
    """Test error handling and validation."""
    
    def test_select_dtypes_no_include_or_exclude(self, tmp_path):
        """Test ValueError when neither include nor exclude provided."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="at least one of"):
            frame.select_dtypes()
    
    def test_select_dtypes_overlapping_include_exclude(self, tmp_path):
        """Test ValueError when include and exclude overlap."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="overlap"):
            frame.select_dtypes(include="int64", exclude="int64")
    
    def test_select_dtypes_unknown_dtype(self, tmp_path):
        """Test ValueError for unknown dtype."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            frame.select_dtypes(include="unknown_type")
    
    def test_select_dtypes_invalid_include_type(self, tmp_path):
        """Test TypeError for invalid include type."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            frame.select_dtypes(include=123)
    
    def test_select_dtypes_invalid_exclude_type(self, tmp_path):
        """Test TypeError for invalid exclude type (set)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            frame.select_dtypes(exclude={"int64"})
    
    def test_select_dtypes_non_string_in_list(self, tmp_path):
        """Test TypeError for non-string elements in include list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,alice\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(TypeError, match="must contain only strings"):
            frame.select_dtypes(include=["int64", 123])
    
    def test_select_dtypes_no_matching_columns(self, tmp_path):
        """Test ValueError when no columns match the filter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,city\nalice,paris\nbob,london\n")
        
        frame = ar.read_csv(str(csv_file))
        
        with pytest.raises(ValueError):
            frame.select_dtypes(include="int64")


class TestSelectDtypesIntegration:
    """Test integration with arnio workflows and pandas."""
    
    def test_select_dtypes_with_pipeline_cleaning(self, tmp_path):
        """Test select_dtypes used after pipeline cleaning."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,revenue,customer,active\n1, 1000.5, Alice , true\n2, 2000.0, Bob , false\n")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        numeric = clean.select_dtypes(include=["int64", "float64"])
        
        df = ar.to_pandas(numeric)
        assert set(df.columns) == {"id", "revenue"}
        assert df.shape == (2, 2)
    
    def test_select_dtypes_to_pandas_conversion(self, tmp_path):
        """Test select_dtypes with to_pandas conversion."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score,active\n1,alice,95.5,true\n2,bob,88.3,false\n")
        
        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])
        df = ar.to_pandas(numeric)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "score"]
        assert len(df) == 2
        assert df["id"].dtype in [np.int64, np.dtype("int64")]
    
    def test_select_dtypes_write_and_read_back(self, tmp_path):
        """Test select_dtypes with write_csv and read_csv round trip."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score,active\n1,alice,95.5,true\n2,bob,88.3,false\n")
        
        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])
        output_file = tmp_path / "numeric_only.csv"
        ar.write_csv(numeric, str(output_file))
        
        result = ar.read_csv(str(output_file))
        assert result.columns == ["id", "score"]
        assert result.shape == (2, 2)
        
        df = ar.to_pandas(result)
        assert len(df) == 2
    
    def test_select_dtypes_chained_with_drop_duplicates(self, tmp_path):
        """Test select_dtypes chained with pipeline operations."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score,duplicate\n1,alice,95.5,1\n1,alice,95.5,1\n")
        
        frame = ar.read_csv(str(csv_file))
        numeric = frame.select_dtypes(include=["int64", "float64"])
        clean = ar.pipeline(numeric, [("drop_duplicates",)])
        
        df = ar.to_pandas(clean)
        assert df.shape[0] == 1
        assert set(df.columns) == {"id", "score", "duplicate"}
    
    def test_select_dtypes_exclude_to_keep_strings(self, tmp_path):
        """Test using exclude to keep only string columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score,city\n1,alice,95.5,paris\n2,bob,88.3,london\n")
        
        frame = ar.read_csv(str(csv_file))
        strings = frame.select_dtypes(exclude=["int64", "float64", "bool"])
        
        df = ar.to_pandas(strings)
        assert set(df.columns) == {"name", "city"}
    
    def test_select_dtypes_all_columns_match(self, tmp_path):
        """Test selecting all columns with numeric filter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,95.5\n2,88.3\n")
        
        frame = ar.read_csv(str(csv_file))
        result = frame.select_dtypes(include=["int64", "float64"])
        
        assert set(result.columns) == {"id", "score"}
        assert result.shape == (2, 2)
        df = ar.to_pandas(result)
        assert len(df) == 2
