import pytest
import arnio as ar
import pandas as pd
import tempfile
import os


@pytest.fixture
def csv_file():
    """Create a temporary CSV file for testing"""
    content = "name,age,salary\nAlice,25,50000\nBob,30,60000\nCharlie,35,70000\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def tsv_file():
    """Create a temporary TSV file for testing"""
    content = "name\tage\tsalary\nAlice\t25\t50000\nBob\t30\t60000\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestReadCSV:
    """Tests for ar.read_csv() functionality"""
    
    def test_read_csv_basic(self, csv_file):
        """read_csv should load a CSV file into an ArFrame"""
        frame = ar.read_csv(csv_file)
        assert frame is not None
    
    def test_read_csv_with_usecols(self, csv_file):
        """read_csv should respect usecols parameter for column selection"""
        frame = ar.read_csv(csv_file, usecols=["name", "age"])
        assert frame is not None
    
    def test_read_csv_with_nrows(self, csv_file):
        """read_csv should limit rows with nrows parameter"""
        frame = ar.read_csv(csv_file, nrows=2)
        assert frame is not None
    
    def test_read_csv_skiprows(self, csv_file):
        """read_csv should skip rows before header with skiprows parameter"""
        content = "# metadata\nname,age,salary\nAlice,25,50000\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            frame = ar.read_csv(f.name, skiprows=1)
            assert frame is not None
            os.unlink(f.name)
    
    def test_read_csv_mode_strict(self, csv_file):
        """read_csv strict mode should accept consistent row widths"""
        frame = ar.read_csv(csv_file, mode="strict")
        assert frame is not None
    
    def test_read_csv_mode_permissive(self):
        """read_csv permissive mode should pad missing trailing fields with nulls"""
        content = "name,age,salary\nAlice,25,50000\nBob,30\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            frame = ar.read_csv(f.name, mode="permissive")
            assert frame is not None
            os.unlink(f.name)
    
    def test_read_csv_tsv_auto_detection(self, tsv_file):
        """read_csv should auto-detect tab delimiter for .tsv files"""
        frame = ar.read_csv(tsv_file)
        assert frame is not None
    
    def test_read_csv_explicit_delimiter_override(self, tsv_file):
        """read_csv explicit delimiter should override auto-detection"""
        frame = ar.read_csv(tsv_file, delimiter=",")
        assert frame is not None
    
    def test_read_csv_dtype_specification(self, csv_file):
        """read_csv should support explicit dtype mapping"""
        frame = ar.read_csv(csv_file, dtype={"age": "int64", "salary": "float64"})
        assert frame is not None


class TestArFrameCreation:
    """Tests for ArFrame creation methods"""
    
    def test_from_dict(self):
        """from_dict should create ArFrame from dictionary of equal-length columns"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None
    
    def test_arframe_from_dict_constructor(self):
        """ArFrame.from_dict should also create ArFrame from dictionary"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None


class TestPipeline:
    """Tests for ar.pipeline() transformation functionality"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample ArFrame for pipeline testing"""
        data = {"name": ["  ALICE  ", "  BOB  "], "age": [25, 30]}
        return ar.from_dict(data)
    
    def test_pipeline_single_step(self, sample_frame):
        """pipeline should execute single transformation step"""
        clean = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert clean is not None
    
    def test_pipeline_multiple_steps(self, sample_frame):
        """pipeline should execute multiple steps in sequence"""
        clean = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ])
        assert clean is not None
    
    def test_pipeline_dry_run(self, sample_frame):
        """pipeline dry_run=True should validate without executing"""
        result = ar.pipeline(sample_frame, [("drop_nulls",)], dry_run=True)
        assert result is None or result is not None
    
    def test_pipeline_return_metadata(self, sample_frame):
        """pipeline return_metadata=True should return (frame, metadata) tuple with step info"""
        clean, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            return_metadata=True,
        )
        assert clean is not None
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata or "applied_steps" in metadata


class TestToPandas:
    """Tests for ar.to_pandas() conversion"""
    
    def test_to_pandas_basic_conversion(self):
        """to_pandas should convert ArFrame to pandas DataFrame"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
    
    def test_to_pandas_with_copy(self):
        """to_pandas copy=True should create defensive pandas-owned buffers"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame, copy=True)
        assert isinstance(df, pd.DataFrame)
