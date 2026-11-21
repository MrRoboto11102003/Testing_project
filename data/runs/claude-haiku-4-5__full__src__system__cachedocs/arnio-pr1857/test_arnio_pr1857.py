import pytest
import pandas as pd
import arnio as ar


class TestFromDict:
    """End-to-end tests for arnio.convert.from_dict"""

    def test_basic_conversion_to_arframe(self):
        """Test basic dictionary to ArFrame conversion"""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
        frame = ar.from_dict(data)

        assert frame.shape == (3, 3)
        assert set(frame.columns) == {"id", "name", "age"}

        df = ar.to_pandas(frame)
        assert df["id"].tolist() == [1, 2, 3]
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert df["age"].tolist() == [25, 30, 35]

    def test_various_data_types(self):
        """Test from_dict with floats, booleans, and mixed types"""
        data = {
            "id": [1, 2, 3],
            "price": [10.5, 20.75, 30.99],
            "is_active": [True, False, True],
        }
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)

        assert df["price"].tolist() == [10.5, 20.75, 30.99]
        assert df["is_active"].tolist() == [True, False, True]

    def test_pipeline_cleaning_after_from_dict(self):
        """Test applying cleaning pipeline to from_dict result"""
        data = {
            "name": [" Alice ", " Bob ", " Bob "],
            "score": [85.5, 90.0, 90.0],
        }
        frame = ar.from_dict(data)

        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "title"}),
                ("drop_duplicates",),
            ],
        )

        df = ar.to_pandas(clean)
        assert len(df) == 2
        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["score"].tolist() == [85.5, 90.0]

    def test_schema_validation_passes(self):
        """Test successful schema validation on from_dict data"""
        data = {
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "age": [25, 30, 35],
        }
        frame = ar.from_dict(data)

        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "age": ar.Int64(nullable=False),
            }
        )

        result = ar.validate(frame, schema)
        assert result.passed

    def test_schema_validation_fails(self):
        """Test failed schema validation on from_dict data"""
        data = {
            "id": [1, 2],
            "email": ["alice@example.com", "invalid-email"],
        }
        frame = ar.from_dict(data)

        schema = ar.Schema({"email": ar.Email(nullable=False)})

        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) > 0

    def test_data_profiling(self):
        """Test profiling a from_dict frame"""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85.5, 90.0, 78.5],
        }
        frame = ar.from_dict(data)

        report = ar.profile(frame)

        assert report.row_count == 3
        assert report.column_count == 3
        assert "id" in report.columns
        assert "score" in report.columns

    def test_csv_roundtrip(self, tmp_path):
        """Test from_dict -> write CSV -> read CSV roundtrip"""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85.5, 90.0, 78.5],
        }
        frame = ar.from_dict(data)

        csv_path = tmp_path / "test.csv"
        ar.write_csv(frame, str(csv_path))

        assert csv_path.exists()

        frame2 = ar.read_csv(str(csv_path))
        df = ar.to_pandas(frame2)

        assert len(df) == 3
        assert df.columns.tolist() == ["id", "name", "score"]
        assert df["id"].tolist() == [1, 2, 3]

    def test_error_non_dict_input(self):
        """Test TypeError when input is not a dict"""
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict([1, 2, 3])

    def test_error_non_string_keys(self):
        """Test TypeError when keys are not strings"""
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({1: [1, 2], "name": ["a", "b"]})

    def test_error_unequal_column_lengths(self):
        """Test ValueError when column lengths differ"""
        with pytest.raises(ValueError, match="column lengths differ"):
            ar.from_dict({"id": [1, 2, 3], "name": ["Alice", "Bob"]})

    def test_error_nested_objects(self):
        """Test ValueError for nested objects in columns"""
        with pytest.raises(ValueError, match="Nested objects are not supported"):
            ar.from_dict({"id": [1, 2], "data": [{"x": 1}, {"x": 2}]})

    def test_error_string_instead_of_list(self):
        """Test TypeError when column value is string instead of list"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"id": [1, 2, 3], "name": "Alice"})

    def test_error_bytes_column(self):
        """Test TypeError when column value is bytes"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"id": [1, 2, 3], "data": b"binary"})

    def test_pandas_accessor_workflow(self):
        """Test from_dict output works with pandas .arnio accessor"""
        data = {
            "id": [1, 2, 3],
            "name": [" Alice ", " Bob ", " Charlie "],
        }
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)

        clean_df = df.arnio.clean(strip_whitespace=True)

        assert clean_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_empty_dict_edge_case(self):
        """Test from_dict with empty dictionary"""
        frame = ar.from_dict({})

        assert frame.shape == (0, 0)
        df = ar.to_pandas(frame)
        assert len(df) == 0
        assert df.columns.tolist() == []
