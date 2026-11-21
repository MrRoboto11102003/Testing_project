import pytest
import pandas as pd
import numpy as np
import duckdb
import arnio as ar


class TestRegisterDuckDB:
    """Test suite for arnio.integrations.duckdb.register_duckdb"""

    def test_register_duckdb_basic_success(self):
        """Test basic successful registration of an ArFrame."""
        frame = ar.from_dict({'id': [1, 2, 3], 'name': ['alice', 'bob', 'charlie']})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'test_table')
        
        result = conn.execute("SELECT COUNT(*) as cnt FROM test_table").fetchall()
        assert result[0][0] == 3

    def test_register_duckdb_query_registered_table(self):
        """Test that registered table can be queried with SQL."""
        frame = ar.from_dict({'value': [10, 20, 30]})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'numbers')
        result_df = conn.execute("SELECT * FROM numbers WHERE value > 15").fetchdf()
        
        assert len(result_df) == 2
        assert list(result_df['value']) == [20, 30]

    def test_register_duckdb_preserves_data_types(self):
        """Test that data types are preserved through registration."""
        frame = ar.from_dict({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.5, 2.5, 3.5]
        })
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'typed_data')
        result = conn.execute("SELECT * FROM typed_data LIMIT 1").fetchdf()
        
        assert result['int_col'].dtype in [np.int64, np.int32]
        assert result['str_col'].dtype == object
        assert result['float_col'].dtype == np.float64

    def test_register_duckdb_with_nulls(self):
        """Test registration with null values."""
        frame = ar.from_dict({'a': [1, None, 3], 'b': ['x', None, 'z']})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'null_table')
        result = conn.execute("SELECT COUNT(*) FROM null_table WHERE a IS NULL").fetchall()
        
        assert result[0][0] == 1

    def test_register_duckdb_empty_frame(self):
        """Test registration of an empty frame."""
        frame = ar.from_dict({'col1': [], 'col2': []})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'empty_table')
        result = conn.execute("SELECT COUNT(*) FROM empty_table").fetchall()[0][0]
        
        assert result == 0

    def test_register_duckdb_frame_not_arframe_raises_typeerror(self):
        """Test TypeError when frame is not an ArFrame."""
        conn = duckdb.connect(':memory:')
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(pd.DataFrame({'a': [1, 2]}), conn, 'table')
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb({'a': [1, 2]}, conn, 'table')
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb([1, 2, 3], conn, 'table')

    def test_register_duckdb_name_not_string_raises_typeerror(self):
        """Test TypeError when name is not a string."""
        frame = ar.from_dict({'a': [1, 2]})
        conn = duckdb.connect(':memory:')
        
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, 123)
        
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, None)
        
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, ['table_name'])

    def test_register_duckdb_empty_name_raises_valueerror(self):
        """Test ValueError when name is empty string."""
        frame = ar.from_dict({'a': [1, 2]})
        conn = duckdb.connect(':memory:')
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, conn, '')

    def test_register_duckdb_whitespace_only_name_raises_valueerror(self):
        """Test ValueError when name is only whitespace."""
        frame = ar.from_dict({'a': [1, 2]})
        conn = duckdb.connect(':memory:')
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, conn, '   ')
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, conn, '\t\n')

    def test_register_duckdb_invalid_connection_raises_typeerror(self):
        """Test TypeError when connection lacks register method."""
        frame = ar.from_dict({'a': [1, 2]})
        
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(frame, object(), 'table')

    def test_register_duckdb_multiple_registrations(self):
        """Test registering multiple frames in same connection."""
        frame1 = ar.from_dict({'x': [1, 2]})
        frame2 = ar.from_dict({'y': [3, 4, 5]})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame1, conn, 'table1')
        ar.register_duckdb(frame2, conn, 'table2')
        
        count1 = conn.execute("SELECT COUNT(*) FROM table1").fetchall()[0][0]
        count2 = conn.execute("SELECT COUNT(*) FROM table2").fetchall()[0][0]
        
        assert count1 == 2
        assert count2 == 3

    def test_register_duckdb_aggregation_operations(self):
        """Test aggregation operations on registered table."""
        frame = ar.from_dict({'id': [1, 2, 3], 'amount': [10.5, 20.3, 30.2]})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame, conn, 'sales')
        result = conn.execute("SELECT SUM(amount) as total FROM sales").fetchall()[0][0]
        
        assert abs(result - 61.0) < 0.01

    def test_register_duckdb_join_multiple_tables(self):
        """Test joining multiple registered tables."""
        frame1 = ar.from_dict({'id': [1, 2], 'value1': [10, 20]})
        frame2 = ar.from_dict({'id': [1, 2], 'value2': [100, 200]})
        conn = duckdb.connect(':memory:')
        
        ar.register_duckdb(frame1, conn, 'left_table')
        ar.register_duckdb(frame2, conn, 'right_table')
        
        result = conn.execute(
            "SELECT l.id, l.value1, r.value2 FROM left_table l "
            "JOIN right_table r ON l.id = r.id"
        ).fetchdf()
        
        assert len(result) == 2
        assert list(result.columns) == ['id', 'value1', 'value2']
