import pytest
import pandas as pd
import arnio as ar
import csv
from pathlib import Path


def create_test_csv(path, data):
    """Helper to create a CSV file from a dict of lists."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        writer.writeheader()
        rows = len(list(data.values())[0])
        for i in range(rows):
            writer.writerow({k: v[i] for k, v in data.items()})


class TestRegisterStepBasic:
    """Test basic registration and usage of custom steps."""

    def test_register_and_use_simple_custom_step(self, tmp_path):
        """Register a simple custom step and use it in a pipeline."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {
            "name": ["alice", "bob", "charlie"],
            "age": ["25", "30", "35"]
        })

        def uppercase_names(df, **kwargs):
            df = df.copy()
            df["name"] = df["name"].str.upper()
            return df

        ar.register_step("uppercase_names", uppercase_names)
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [("uppercase_names",)])
            result = ar.to_pandas(clean)
            
            assert result["name"].tolist() == ["ALICE", "BOB", "CHARLIE"]
        finally:
            ar.reset_steps()

    def test_custom_step_with_parameters(self, tmp_path):
        """Register a custom step that accepts parameters."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {
            "value": ["10", "20", "30"],
            "name": ["a", "b", "c"]
        })

        def multiply_column(df, column, factor, **kwargs):
            df = df.copy()
            if column in df.columns:
                df[column] = pd.to_numeric(df[column]) * factor
            return df

        ar.register_step("multiply", multiply_column)
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [
                ("multiply", {"column": "value", "factor": 2}),
            ])
            result = ar.to_pandas(clean)
            
            assert result["value"].tolist() == [20.0, 40.0, 60.0]
        finally:
            ar.reset_steps()

    def test_multiple_custom_steps_in_pipeline(self, tmp_path):
        """Chain multiple custom steps in a single pipeline."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {
            "value": ["1", "2", "3"],
            "name": ["alice", "bob", "charlie"]
        })

        def add_one(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) + 1
            return df

        def multiply_two(df, **kwargs):
            df = df.copy()
            df["value"] = df["value"] * 2
            return df

        ar.register_step("add_one", add_one)
        ar.register_step("multiply_two", multiply_two)
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [
                ("add_one",),
                ("multiply_two",),
            ])
            result = ar.to_pandas(clean)
            
            # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
            assert result["value"].tolist() == [4.0, 6.0, 8.0]
        finally:
            ar.reset_steps()


class TestOverwriteBehavior:
    """Test overwrite behavior of register_step."""

    def test_cannot_overwrite_without_flag(self):
        """Registering same name twice without overwrite flag should raise ValueError."""
        def step1(df, **kwargs):
            return df.copy()

        def step2(df, **kwargs):
            return df.copy()

        ar.register_step("my_test_step", step1)
        try:
            with pytest.raises(ValueError, match="already registered"):
                ar.register_step("my_test_step", step2)
        finally:
            ar.reset_steps()

    def test_overwrite_with_flag(self, tmp_path):
        """Overwrite=True should allow replacing a custom step."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {"value": ["1", "2", "3"]})

        def step1(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) * 10
            return df

        def step2(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) * 100
            return df

        ar.register_step("test_overwrite", step1)
        ar.register_step("test_overwrite", step2, overwrite=True)
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [("test_overwrite",)])
            result = ar.to_pandas(clean)
            
            # Should use step2 (multiply by 100)
            assert result["value"].tolist() == [100.0, 200.0, 300.0]
        finally:
            ar.reset_steps()

    def test_cannot_overwrite_builtin_cpp_step(self):
        """Cannot overwrite built-in C++ steps."""
        def fake_drop_nulls(df, **kwargs):
            return df.copy()

        with pytest.raises(ValueError, match="conflicts with built-in"):
            ar.register_step("drop_nulls", fake_drop_nulls)

    def test_cannot_overwrite_builtin_python_step(self):
        """Cannot overwrite built-in Python steps like filter_rows."""
        def fake_filter_rows(df, **kwargs):
            return df.copy()

        with pytest.raises(ValueError, match="conflicts with built-in"):
            ar.register_step("filter_rows", fake_filter_rows)


class TestResetSteps:
    """Test reset_steps functionality."""

    def test_reset_clears_custom_steps(self, tmp_path):
        """reset_steps should clear all custom steps, making them unavailable."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {"value": ["1", "2", "3"]})

        def custom_step(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) * 2
            return df

        ar.register_step("custom_cleanup", custom_step)
        
        # Verify step is registered
        frame = ar.read_csv(str(csv_file))
        clean1 = ar.pipeline(frame, [("custom_cleanup",)])
        result1 = ar.to_pandas(clean1)
        assert result1["value"].tolist() == [2.0, 4.0, 6.0]
        
        # Reset and verify step is gone
        ar.reset_steps()
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises((ar.UnknownStepError, ValueError, KeyError)):
            ar.pipeline(frame, [("custom_cleanup",)])

    def test_reset_and_reregister(self, tmp_path):
        """After reset, can register steps with the same name again."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {"value": ["1", "2", "3"]})

        def step_v1(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) + 1
            return df

        def step_v2(df, **kwargs):
            df = df.copy()
            df["value"] = pd.to_numeric(df["value"]) + 10
            return df

        ar.register_step("step_reregister", step_v1)
        ar.reset_steps()
        ar.register_step("step_reregister", step_v2)
        
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [("step_reregister",)])
            result = ar.to_pandas(clean)
            
            # Should use step_v2 (add 10)
            assert result["value"].tolist() == [11.0, 12.0, 13.0]
        finally:
            ar.reset_steps()


class TestErrorHandling:
    """Test error handling for register_step."""

    def test_invalid_step_name_empty_string(self):
        """Empty string as name should raise ValueError."""
        def dummy(df, **kwargs):
            return df

        with pytest.raises(ValueError, match="non-empty"):
            ar.register_step("", dummy)

    def test_invalid_step_name_whitespace(self):
        """Whitespace-only name should raise ValueError."""
        def dummy(df, **kwargs):
            return df

        with pytest.raises(ValueError, match="non-empty"):
            ar.register_step("   ", dummy)

    def test_non_callable_function(self):
        """Non-callable fn should raise TypeError."""
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("my_step", "not a function")

    def test_invalid_overwrite_parameter(self):
        """Non-bool overwrite should raise TypeError."""
        def dummy(df, **kwargs):
            return df

        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("my_step", dummy, overwrite="yes")


class TestCustomStepErrorHandling:
    """Test behavior when custom steps raise errors."""

    def test_custom_step_raises_error(self, tmp_path):
        """Errors in custom steps should propagate to the caller."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {"value": ["1", "2", "3"]})

        def broken_step(df, **kwargs):
            raise ValueError("Intentional error in custom step")

        ar.register_step("broken_step", broken_step)
        try:
            frame = ar.read_csv(str(csv_file))
            with pytest.raises(ValueError, match="Intentional error"):
                ar.pipeline(frame, [("broken_step",)])
        finally:
            ar.reset_steps()

    def test_custom_step_missing_required_column(self, tmp_path):
        """Step operating on missing column should handle gracefully."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {"value": ["1", "2", "3"]})

        def step_needs_name(df, **kwargs):
            df = df.copy()
            if "name" not in df.columns:
                raise ValueError("Column 'name' not found")
            df["name"] = df["name"].str.upper()
            return df

        ar.register_step("needs_name", step_needs_name)
        try:
            frame = ar.read_csv(str(csv_file))
            with pytest.raises(ValueError, match="Column 'name' not found"):
                ar.pipeline(frame, [("needs_name",)])
        finally:
            ar.reset_steps()


class TestComplexWorkflow:
    """Test more complex, realistic workflows."""

    def test_data_cleaning_workflow_with_custom_steps(self, tmp_path):
        """Complete workflow: read messy CSV, apply custom steps, validate, write output."""
        input_file = tmp_path / "messy.csv"
        create_test_csv(input_file, {
            "user_id": ["1", "2", "3", "4"],
            "email": ["alice@example.com", "bob@example.com", 
                     "charlie@example.com", "dave@example.com"],
            "status": ["active", "inactive", "active", "inactive"]
        })

        def normalize_status(df, **kwargs):
            """Convert status to title case."""
            df = df.copy()
            df["status"] = df["status"].str.title()
            return df

        def add_domain_column(df, **kwargs):
            """Extract domain from email."""
            df = df.copy()
            df["domain"] = df["email"].str.split("@").str[1]
            return df

        ar.register_step("normalize_status", normalize_status)
        ar.register_step("add_domain", add_domain_column)
        
        try:
            # Read and process
            frame = ar.read_csv(str(input_file))
            clean = ar.pipeline(frame, [
                ("normalize_status",),
                ("add_domain",),
            ])
            
            # Convert to pandas and save
            result = ar.to_pandas(clean)
            output_file = tmp_path / "clean.csv"
            result.to_csv(output_file, index=False)
            
            # Verify
            saved = pd.read_csv(output_file)
            assert saved["status"].tolist() == ["Active", "Inactive", "Active", "Inactive"]
            assert saved["domain"].tolist() == ["example.com"] * 4
            assert "user_id" in saved.columns
            assert "email" in saved.columns
        finally:
            ar.reset_steps()

    def test_custom_step_with_validation_schema(self, tmp_path):
        """Use custom step, then validate result with schema."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {
            "user_id": ["1", "2", "3"],
            "email": ["alice@example.com", "bob@example.com", 
                     "charlie@example.com"]
        })

        def ensure_valid_id(df, **kwargs):
            df = df.copy()
            df["user_id"] = pd.to_numeric(df["user_id"]).astype("int64")
            return df

        ar.register_step("ensure_valid_id", ensure_valid_id)
        
        try:
            frame = ar.read_csv(str(csv_file))
            clean = ar.pipeline(frame, [("ensure_valid_id",)])
            
            # Validate result
            schema = ar.Schema({
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
            })
            
            result = ar.validate(clean, schema)
            assert result.passed
        finally:
            ar.reset_steps()

    def test_custom_step_with_pandas_accessor(self, tmp_path):
        """Custom steps work with pandas .arnio accessor."""
        df = pd.DataFrame({
            "name": ["alice", "bob"],
            "age": [25, 30]
        })

        def uppercase_names(df, **kwargs):
            df = df.copy()
            df["name"] = df["name"].str.upper()
            return df

        ar.register_step("uppercase_accessor", uppercase_names)
        try:
            # Using pandas accessor with custom step
            result = df.arnio.clean(steps=[("uppercase_accessor",)])
            
            assert result["name"].tolist() == ["ALICE", "BOB"]
        finally:
            ar.reset_steps()

    def test_profiling_data_before_and_after_custom_steps(self, tmp_path):
        """Profile dataset before and after applying custom steps."""
        csv_file = tmp_path / "data.csv"
        create_test_csv(csv_file, {
            "value": ["  1  ", "  2  ", "  3  "],
            "name": ["alice", "bob", "charlie"]
        })

        def strip_values(df, **kwargs):
            df = df.copy()
            df["value"] = df["value"].str.strip()
            return df

        ar.register_step("strip_values", strip_values)
        
        try:
            frame = ar.read_csv(str(csv_file))
            
            # Profile before
            report_before = ar.profile(frame)
            assert report_before.row_count == 3
            
            # Apply custom step
            clean = ar.pipeline(frame, [("strip_values",)])
            
            # Profile after
            report_after = ar.profile(clean)
            assert report_after.row_count == 3
            
            # Verify data was cleaned
            result = ar.to_pandas(clean)
            assert result["value"].tolist() == ["1", "2", "3"]
        finally:
            ar.reset_steps()
