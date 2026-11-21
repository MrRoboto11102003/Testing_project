import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file with various data types."""
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "active": [True, False, True, False, True],
        "notes": ["", "good", None, "excellent", "ok"]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


class TestDropColumns:
    """Test ArFrame.drop_columns with complete user workflows."""
    
    def test_drop_single_and_multiple_columns(self, sample_csv, tmp_path):
        """Workflow: load CSV, drop columns, save and verify output."""
        frame = ar.read_csv(sample_csv)
        result = frame.drop_columns(["notes", "active"])
        
        assert list(result.columns) == ["id", "name", "age", "salary"]
        assert len(result) == 5
        
        # Save result to file and validate
        output_file = tmp_path / "dropped.csv"
        result.to_pandas().to_csv(output_file, index=False)
        saved = pd.read_csv(output_file)
        assert list(saved.columns) == ["id", "name", "age", "salary"]
    
    def test_drop_with_tuple_and_silent_deduplication(self, sample_csv):
        """Test drop with tuple input; duplicates silently ignored."""
        frame = ar.read_csv(sample_csv)
        result = frame.drop_columns(("name", "age", "name"))
        
        assert "name" not in result.columns
        assert "age" not in result.columns
        assert len(result.columns) == 4
        # Verify order is preserved
        assert result.columns[0] == "id"
    
    def test_drop_empty_list_returns_unchanged_copy(self, sample_csv):
        """Test that empty drop list returns a copy unchanged."""
        frame = ar.read_csv(sample_csv)
        result = frame.drop_columns([])
        
        assert list(result.columns) == list(frame.columns)
        assert len(result) == len(frame)
        assert result is not frame  # Verify it's a new object
    
    def test_drop_nonexistent_column_raises_valueerror(self, sample_csv):
        """Test ValueError when dropping non-existent column."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError) as exc:
            frame.drop_columns(["missing_column"])
        assert "Unknown column" in str(exc.value)
    
    def test_drop_all_columns_and_invalid_type_errors(self, sample_csv):
        """Test ValueError when dropping all columns; TypeError for invalid type."""
        frame = ar.read_csv(sample_csv)
        
        # Cannot drop all columns
        with pytest.raises(ValueError) as exc:
            frame.drop_columns(list(frame.columns))
        assert "cannot remove all columns" in str(exc.value).lower()
        
        # Invalid type error
        with pytest.raises(TypeError) as exc:
            frame.drop_columns("name")
        assert "list or tuple" in str(exc.value)


class TestSelectColumns:
    """Test ArFrame.select_columns with complete user workflows."""
    
    def test_select_subset_and_save(self, sample_csv, tmp_path):
        """Workflow: load CSV, select columns, save and verify."""
        frame = ar.read_csv(sample_csv)
        result = frame.select_columns(["id", "name", "age"])
        
        assert list(result.columns) == ["id", "name", "age"]
        assert len(result) == 5
        
        # Save and validate
        output_file = tmp_path / "selected.csv"
        result.to_pandas().to_csv(output_file, index=False)
        saved = pd.read_csv(output_file)
        assert list(saved.columns) == ["id", "name", "age"]
    
    def test_select_respects_requested_order(self, sample_csv):
        """Test that selection respects requested column order."""
        frame = ar.read_csv(sample_csv)
        result = frame.select_columns(["salary", "id", "name"])
        
        assert list(result.columns) == ["salary", "id", "name"]
        pdf = result.to_pandas()
        assert list(pdf.columns) == ["salary", "id", "name"]
    
    def test_select_empty_raises_valueerror(self, sample_csv):
        """Test ValueError when selecting empty column list."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError) as exc:
            frame.select_columns([])
        assert "empty" in str(exc.value).lower()
    
    def test_select_duplicates_and_missing_column_errors(self, sample_csv):
        """Test errors for duplicate columns and missing columns."""
        frame = ar.read_csv(sample_csv)
        
        # Duplicates not allowed
        with pytest.raises(ValueError) as exc:
            frame.select_columns(["name", "age", "name"])
        assert "duplicate" in str(exc.value).lower()
        
        # Missing column error
        with pytest.raises(ValueError) as exc:
            frame.select_columns(["name", "missing_column"])
        assert "Unknown column" in str(exc.value)
    
    def test_select_invalid_input_type_errors(self, sample_csv):
        """Test TypeError for string input and invalid types."""
        frame = ar.read_csv(sample_csv)
        
        # String instead of list
        with pytest.raises(TypeError) as exc:
            frame.select_columns("name")
        assert "sequence" in str(exc.value).lower()


class TestSelectDtypes:
    """Test ArFrame.select_dtypes with complete user workflows."""
    
    def test_select_numeric_dtypes_and_save(self, sample_csv, tmp_path):
        """Workflow: load CSV, filter numeric dtypes, save result."""
        frame = ar.read_csv(sample_csv)
        numeric = frame.select_dtypes(include=["int64", "float64"])
        
        assert "id" in numeric.columns
        assert "age" in numeric.columns
        assert "salary" in numeric.columns
        assert "name" not in numeric.columns
        assert len(numeric.columns) == 3
        
        # Save and validate
        output_file = tmp_path / "numeric.csv"
        numeric.to_pandas().to_csv(output_file, index=False)
        saved = pd.read_csv(output_file)
        assert len(saved.columns) == 3
    
    def test_exclude_string_and_tuple_include_input(self, sample_csv):
        """Test excluding string type; tuple input for include."""
        frame = ar.read_csv(sample_csv)
        
        # Exclude strings
        result = frame.select_dtypes(exclude="string")
        assert "name" not in result.columns
        assert "notes" not in result.columns
        
        # Tuple input works
        with_tuple = frame.select_dtypes(include=("int64", "float64"))
        assert all(c in with_tuple.columns for c in ["id", "age", "salary"])
    
    def test_select_dtypes_preserves_column_order(self, sample_csv):
        """Test that dtype filtering preserves original column order."""
        frame = ar.read_csv(sample_csv)
        result = frame.select_dtypes(include=["int64", "float64", "string"])
        
        # Order should match original frame order
        expected = [c for c in frame.columns if c in result.columns]
        assert list(result.columns) == expected
    
    def test_select_dtypes_no_args_raises_valueerror(self, sample_csv):
        """Test ValueError when neither include nor exclude provided."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError) as exc:
            frame.select_dtypes()
        assert "at least one" in str(exc.value).lower()
    
    def test_select_dtypes_overlap_and_invalid_dtype_errors(self, sample_csv):
        """Test errors for overlapping include/exclude and invalid dtype."""
        frame = ar.read_csv(sample_csv)
        
        # Overlap error
        with pytest.raises(ValueError) as exc:
            frame.select_dtypes(include="string", exclude="string")
        assert "overlap" in str(exc.value).lower()
        
        # Invalid dtype error
        with pytest.raises(ValueError) as exc:
            frame.select_dtypes(include="unknown_dtype")
        assert "Unrecognised dtype" in str(exc.value)
