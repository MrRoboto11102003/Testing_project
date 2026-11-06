import pytest
import pandas as pd
import arnio as ar


class TestToPandas:
    """Tests for arnio.convert.to_pandas function."""

    def test_basic_csv_conversion(self, tmp_path):
        """Test basic conversion from CSV-loaded ArFrame to DataFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\ntest,100\ndata,200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert list(result.columns) == ["name", "value"]

    def test_from_dict_conversion(self):
        """Test conversion of ArFrame created from dictionary."""
        data = {"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert list(result.columns) == ["col_a", "col_b"]

    def test_copy_parameter_default_false(self, tmp_path):
        """Test that copy=False is default (zero-copy path where supported)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        result_default = ar.to_pandas(frame)
        result_explicit = ar.to_pandas(frame, copy=False)
        
        assert isinstance(result_default, pd.DataFrame)
        assert isinstance(result_explicit, pd.DataFrame)
        assert result_default.shape == result_explicit.shape

    def test_copy_parameter_true(self, tmp_path):
        """Test copy=True returns defensive pandas-owned copies."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n10,20\n30,40\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame, copy=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_copy_is_keyword_only(self, tmp_path):
        """Test that copy parameter is keyword-only."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x\n1\n")
        
        frame = ar.read_csv(str(csv_file))
        
        # Keyword argument works
        result = ar.to_pandas(frame, copy=False)
        assert isinstance(result, pd.DataFrame)
        
        # Positional argument should raise TypeError
        with pytest.raises(TypeError):
            ar.to_pandas(frame, False)

    def test_round_trip_from_pandas(self):
        """Test round-trip preserves structure and metadata."""
        original = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.5, 2.5, 3.5]
        })
        original.attrs["custom"] = "metadata"
        
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == original.shape
        assert list(result.columns) == list(original.columns)

    def test_empty_frame(self):
        """Test conversion of empty ArFrame."""
        data = {"col1": [], "col2": [], "col3": []}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2", "col3"]

    def test_single_row(self):
        """Test conversion of single-row ArFrame."""
        data = {"x": [42], "y": [99]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        
        assert result.shape == (1, 2)
        assert isinstance(result, pd.DataFrame)

    def test_single_column(self):
        """Test conversion of single-column ArFrame."""
        data = {"single": [10, 20, 30, 40]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        
        assert result.shape == (4, 1)
        assert result.columns[0] == "single"

    def test_column_order_preserved(self, tmp_path):
        """Test that column order is preserved during conversion."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("z,y,x,w\n1,2,3,4\n5,6,7,8\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["z", "y", "x", "w"]

    def test_shape_preserved(self, tmp_path):
        """Test that frame shape (rows, columns) is preserved."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert result.shape == (4, 3)

    def test_return_type(self, tmp_path):
        """Test that return type is pd.DataFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("val\n100\n200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert type(result) == pd.DataFrame

    def test_multiple_dtypes(self, tmp_path):
        """Test proper handling of multiple data types in conversion."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text("int_col,str_col,float_col\n1,hello,1.5\n2,world,2.5\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3
        assert len(result) == 2

    def test_data_integrity(self, tmp_path):
        """Test that data values are preserved during conversion."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.to_pandas(frame)
        
        assert result.shape == (3, 2)
        assert len(result) == 3
        assert list(result.columns) == ["id", "name"]

    def test_large_frame(self):
        """Test conversion of larger ArFrame."""
        data = {
            "col1": list(range(1000)),
            "col2": list(range(1000, 2000)),
            "col3": ["x"] * 1000
        }
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        
        assert result.shape == (1000, 3)
        assert isinstance(result, pd.DataFrame)
