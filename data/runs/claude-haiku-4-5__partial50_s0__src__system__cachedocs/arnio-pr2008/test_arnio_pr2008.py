import pytest
import pandas as pd
import arnio as ar
import tempfile
import os


class TestUnregisterStepE2E:
    """End-to-end tests for arnio.pipeline.unregister_step"""

    def test_register_and_unregister_basic(self, tmp_path):
        """Basic workflow: register custom step, use in pipeline, then unregister."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value\nalice,10\nbob,20\n")

        def double_value(df, column="value"):
            df = df.copy()
            df[column] = df[column] * 2
            return df

        ar.register_step("double_value", double_value)

        # Verify step works in pipeline
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("double_value",)])
        df = ar.to_pandas(result)
        assert df["value"].tolist() == [20, 40]

        # Unregister the step
        ar.unregister_step("double_value")

        # Verify step is no longer available
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("double_value",)])

        ar.reset_steps()

    def test_unregister_nonexistent_step_raises_error(self):
        """Attempting to unregister non-existent step raises UnknownStepError."""
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("does_not_exist_xyz")

    def test_unregister_builtin_step_raises_error(self):
        """Attempting to unregister built-in steps raises ValueError."""
        builtin_steps = ["drop_nulls", "strip_whitespace", "drop_duplicates"]

        for step_name in builtin_steps:
            with pytest.raises(ValueError, match="Cannot unregister built-in step"):
                ar.unregister_step(step_name)

    def test_unregister_invalid_name_type(self):
        """Non-string names raise TypeError."""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(123)

        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(None)

        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(["step"])

    def test_unregister_empty_string_raises_error(self):
        """Empty string names raise ValueError."""
        with pytest.raises(ValueError, match="parameter 'name' must be a non-empty string"):
            ar.unregister_step("")

    def test_unregister_selective_from_multiple_steps(self, tmp_path):
        """Unregister one custom step while others remain operational."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n5\n10\n")

        def add_ten(df, column="val"):
            df = df.copy()
            df[column] = df[column] + 10
            return df

        def multiply_two(df, column="val"):
            df = df.copy()
            df[column] = df[column] * 2
            return df

        ar.register_step("add_ten", add_ten)
        ar.register_step("multiply_two", multiply_two)

        # Both steps work together
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("add_ten",), ("multiply_two",)])
        df = ar.to_pandas(result)
        assert df["val"].tolist() == [30, 40]  # (5+10)*2=30, (10+10)*2=40

        # Unregister only add_ten
        ar.unregister_step("add_ten")

        # multiply_two still works
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("multiply_two",)])
        df = ar.to_pandas(result)
        assert df["val"].tolist() == [10, 20]

        # add_ten is gone
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("add_ten",)])

        ar.reset_steps()

    def test_unregister_then_reregister_new_version(self, tmp_path):
        """Can register a new version of a step after unregistering."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n5\n")

        def v1(df, column="val"):
            df = df.copy()
            df[column] = 100
            return df

        ar.register_step("process", v1)
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("process",)])
        assert ar.to_pandas(result)["val"].tolist() == [100]

        ar.unregister_step("process")

        def v2(df, column="val"):
            df = df.copy()
            df[column] = 200
            return df

        ar.register_step("process", v2)
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("process",)])
        assert ar.to_pandas(result)["val"].tolist() == [200]

        ar.reset_steps()

    def test_unregister_affects_pandas_accessor(self, tmp_path):
        """Unregister makes step unavailable through pandas .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("score\n95\n88\n")

        def boost(df, points=5):
            df = df.copy()
            df["score"] = df["score"] + points
            return df

        ar.register_step("boost", boost)

        # Works via pandas accessor
        df = pd.read_csv(csv_file)
        result = df.arnio.clean(steps=[("boost", {"points": 10})])
        assert result["score"].tolist() == [105, 98]

        ar.unregister_step("boost")

        # Now fails via pandas accessor
        df = pd.read_csv(csv_file)
        with pytest.raises(ar.UnknownStepError):
            df.arnio.clean(steps=[("boost", {"points": 10})])

        ar.reset_steps()

    def test_unregister_step_disappears_from_list_steps(self):
        """Unregistered steps no longer appear in list_steps()."""
        def step(df):
            return df

        ar.register_step("temporary", step)
        assert "temporary" in ar.list_steps()

        ar.unregister_step("temporary")
        assert "temporary" not in ar.list_steps()

        ar.reset_steps()

    def test_unregister_namespace_qualified_step(self, tmp_path):
        """Unregister works with namespaced custom step names."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")

        def custom(df):
            df = df.copy()
            df["val"] = df["val"] + 1
            return df

        ar.register_step("team:process", custom)

        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("team:process",)])
        assert ar.to_pandas(result)["val"].tolist() == [2]

        ar.unregister_step("team:process")

        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(ar.read_csv(str(csv_file)), [("team:process",)])

        ar.reset_steps()

    def test_unregister_is_case_sensitive(self):
        """Step names are case-sensitive for unregister."""
        def step(df):
            return df

        ar.register_step("MyStep", step)

        # Lowercase doesn't match
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("mystep")

        # Uppercase doesn't match
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("MYSTEP")

        # Exact case works
        ar.unregister_step("MyStep")
        assert "MyStep" not in ar.list_steps()

        ar.reset_steps()

    def test_unregister_caught_in_pipeline_validation(self, tmp_path):
        """Pipeline validation catches unregistered steps before execution."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")

        def step(df):
            df = df.copy()
            df["val"] = df["val"] + 1
            return df

        ar.register_step("step1", step)
        ar.unregister_step("step1")

        frame = ar.read_csv(str(csv_file))

        # Validation fails even when other valid steps follow
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("step1",), ("drop_nulls",)])

        ar.reset_steps()

    def test_unregister_with_dry_run_validates(self, tmp_path):
        """Dry run mode still validates unregistered steps."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")

        def step(df):
            return df

        ar.register_step("custom", step)
        ar.unregister_step("custom")

        frame = ar.read_csv(str(csv_file))

        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("custom",)], dry_run=True)

        ar.reset_steps()

    def test_reset_steps_clears_unregistered_steps(self):
        """reset_steps() clears all custom steps, restoring only built-ins."""
        def step(df):
            return df

        ar.register_step("custom1", step)
        ar.register_step("custom2", step)

        ar.unregister_step("custom1")
        assert "custom1" not in ar.list_steps()
        assert "custom2" in ar.list_steps()

        ar.reset_steps()

        assert "custom1" not in ar.list_steps()
        assert "custom2" not in ar.list_steps()
        assert "drop_nulls" in ar.list_steps()

    def test_unregister_full_workflow_with_io(self, tmp_path):
        """Full workflow: register, process data, unregister, verify unavailable."""
        csv_file = tmp_path / "data.csv"
        output_file = tmp_path / "output.csv"

        csv_file.write_text("val\n1\n2\n")

        def process(df):
            df = df.copy()
            df["val"] = df["val"] * 10
            return df

        ar.register_step("process", process)

        # Use custom step in pipeline and write output
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [("process",)])
        ar.write_csv(result, str(output_file))

        df = pd.read_csv(output_file)
        assert df["val"].tolist() == [10, 20]

        # Unregister the step
        ar.unregister_step("process")

        # Verify unavailable for new pipeline
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("process",)])

        ar.reset_steps()
