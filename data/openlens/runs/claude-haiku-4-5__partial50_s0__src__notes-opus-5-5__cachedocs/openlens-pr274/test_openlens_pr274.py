import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.geometric import GeometricTraceAnalysis
from openlens.ray import Ray3D, RefractionResult
from openlens.tracer_3d import LensRayTracer3D
from openlens.vector3 import vec3
from openlens.constants import (
    WAVELENGTH_GREEN,
    SPHERICAL_ABERRATION_EXCELLENT,
)
import openlens.aberrations as ab


class TestAnalyzeLensQuality:
    """Test openlens.aberrations.analyze_lens_quality"""

    def test_basic_scoring(self, monkeypatch):
        """Test basic quality scoring with clean aberrations."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        
        fake_result = {
            "spherical": 0.0,
            "chromatic": 0.0,
            "distortion": 0.0,
            "astigmatism": 0.0,
        }
        
        monkeypatch.setattr(
            ab.AberrationsCalculator,
            "calculate_all_aberrations",
            lambda self, **kw: fake_result
        )
        
        result = ab.analyze_lens_quality(lens, field_angle=5.0)
        
        assert result["quality_score"] == 100
        assert result["rating"] == "Excellent"
        assert result["aberrations"] is fake_result
        assert "No significant aberrations detected" in result["issues"]

    def test_error_branch(self, monkeypatch):
        """Test error handling when focal length is unavailable."""
        lens = Lens()
        
        fake_result = {"error": "Cannot calculate focal length (zero optical power)"}
        
        monkeypatch.setattr(
            ab.AberrationsCalculator,
            "calculate_all_aberrations",
            lambda self, **kw: fake_result
        )
        
        result = ab.analyze_lens_quality(lens)
        
        assert result["quality_score"] == 0
        assert result["rating"] == "Error"
        assert "aberrations" not in result
        assert result["issues"] == ["Cannot calculate focal length (zero optical power)"]

    def test_scoring_penalties(self, monkeypatch):
        """Test scoring penalty application for high spherical aberration."""
        lens = Lens()
        
        fake_result = {
            "spherical": 2.0 * SPHERICAL_ABERRATION_EXCELLENT,
            "chromatic": 0.0,
            "distortion": 0.0,
            "astigmatism": 0.0,
        }
        
        monkeypatch.setattr(
            ab.AberrationsCalculator,
            "calculate_all_aberrations",
            lambda self, **kw: fake_result
        )
        
        result = ab.analyze_lens_quality(lens)
        # 100 - 40 = 60 for high spherical
        assert result["quality_score"] == 60
        assert result["rating"] == "Fair"


class TestGeometricTraceAnalysis:
    """Test openlens.analysis.geometric.GeometricTraceAnalysis"""

    def test_calculate_field_curvature_distortion_basic(self):
        """Test basic field curvature and distortion calculation."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        sys = OpticalSystem()
        sys.add_lens(lens)
        
        ga = GeometricTraceAnalysis(sys)
        result = ga.calculate_field_curvature_distortion(
            max_field_angle_deg=10.0,
            num_points=3
        )
        
        assert "field_angles_deg" in result
        assert "tan_focus_shift_mm" in result
        assert "sag_focus_shift_mm" in result
        assert "distortion_pct" in result
        assert "image_plane_x_mm" in result
        
        assert result["tan_focus_shift"] is result["tan_focus_shift_mm"]
        assert result["sag_focus_shift"] is result["sag_focus_shift_mm"]
        assert len(result["field_angles_deg"]) == 3
        assert result["distortion_pct"][0] == 0.0  # On-axis

    def test_calculate_field_curvature_distortion_invalid_kwargs(self):
        """Test that invalid kwargs raise TypeError."""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        sys = OpticalSystem()
        sys.add_lens(lens)
        ga = GeometricTraceAnalysis(sys)
        
        with pytest.raises(TypeError, match=r"Unexpected kwargs"):
            ga.calculate_field_curvature_distortion(invalid_param=1)

    def test_calculate_ray_fan_basic(self):
        """Test basic ray fan (transverse ray aberration) calculation."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        sys = OpticalSystem()
        sys.add_lens(lens)
        
        ga = GeometricTraceAnalysis(sys)
        result = ga.calculate_ray_fan(field_angle_deg=0.0, num_points=11)
        
        assert "pupil_coords" in result
        assert "ray_errors_mm" in result
        assert len(result["pupil_coords"]) <= 11
        assert len(result["pupil_coords"]) == len(result["ray_errors_mm"])
        
        assert result["ray_errors"] is result["ray_errors_mm"]
        assert result["transverse_aberration"] is result["ray_errors_mm"]
        assert result["axis"] == "y"
        assert result["field_angle_deg"] == 0.0
        assert result["wavelength_nm"] == WAVELENGTH_GREEN

    def test_calculate_ray_fan_center_error_zero(self):
        """Test that center pupil point has exactly zero error."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        sys = OpticalSystem()
        sys.add_lens(lens)
        ga = GeometricTraceAnalysis(sys)
        
        result = ga.calculate_ray_fan(num_points=21)
        
        if 0.0 in result["pupil_coords"]:
            idx = result["pupil_coords"].index(0.0)
            assert result["ray_errors_mm"][idx] == 0.0


class TestImageSimulator:
    """Test openlens.image_simulator.ImageSimulator"""

    def test_simulate_image_identity(self):
        """Test identity case: uniform image through simple system."""
        from openlens.image_simulator import ImageSimulator
        
        class FakeLens:
            refractive_index = 1.5
            diameter = 25.0
            def refractive_index_at(self, wl): return 1.5
        
        class FakeElement:
            def __init__(self):
                self.lens = FakeLens()
                self.position = 0.0
        
        class FakeSys:
            def __init__(self):
                self.elements = [FakeElement()]
            def get_system_focal_length(self): return 100.0
            def get_aberrations(self, wl): return {}
        
        sim = ImageSimulator(FakeSys())
        
        input_img = np.ones((64, 64))
        result = sim.simulate_image(
            input_img,
            object_distance=200.0,
            pixel_pitch_mm=0.0
        )
        
        assert "output_image" in result
        assert "magnification" in result
        assert "image_distance" in result
        assert "metrics" in result
        assert result["magnification"] == pytest.approx(-1.0)

    def test_simulate_image_zero_object_distance(self):
        """Test that zero object_distance raises ZeroDivisionError."""
        from openlens.image_simulator import ImageSimulator
        
        class FakeLens:
            refractive_index = 1.5
            diameter = 25.0
            def refractive_index_at(self, wl): return 1.5
        
        class FakeElement:
            lens = FakeLens()
            position = 0.0
        
        class FakeSys:
            elements = [FakeElement()]
            def get_system_focal_length(self): return 100.0
            def get_aberrations(self, wl): return {}
        
        sim = ImageSimulator(FakeSys())
        
        with pytest.raises(ZeroDivisionError):
            sim.simulate_image(np.ones((32, 32)), object_distance=0.0)


class TestOpticalSystemChromatic:
    """Test openlens.optical_system.OpticalSystem.calculate_chromatic_aberration"""

    def test_calculate_chromatic_aberration_empty(self):
        """Test that empty system returns zero longitudinal CA."""
        sys = OpticalSystem()
        result = sys.calculate_chromatic_aberration()
        
        assert result["longitudinal"] == 0.0
        assert result["corrected"] is False
        assert len(result) == 2

    def test_calculate_chromatic_aberration_singlet(self):
        """Test LCA calculation for a single BK7 lens."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        sys = OpticalSystem()
        sys.add_lens(lens)
        
        result = sys.calculate_chromatic_aberration()
        
        expected_keys = {
            "longitudinal", "bfl_F", "bfl_d", "bfl_C",
            "f_F", "f_d", "f_C", "corrected"
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["longitudinal"], float)
        assert isinstance(result["corrected"], bool)
        
        if result["bfl_F"] is not None and result["bfl_C"] is not None:
            assert result["longitudinal"] == pytest.approx(
                result["bfl_C"] - result["bfl_F"]
            )


class TestOpticalSystemLoad:
    """Test openlens.optical_system.OpticalSystem.load"""

    def test_load_roundtrip(self, tmp_path):
        """Test save/load roundtrip preserves system properties."""
        lens = Lens(
            name="TestLens",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        sys = OpticalSystem(name="TestSystem")
        sys.add_lens(lens)
        
        path = tmp_path / "system.json"
        assert sys.save(str(path))
        
        loaded = OpticalSystem.load(str(path))
        
        assert loaded is not None
        assert loaded.name == "TestSystem"
        assert len(loaded.elements) == 1
        assert loaded.elements[0].lens.name == "TestLens"

    def test_load_missing_file(self, tmp_path):
        """Test that loading missing file returns None."""
        path = tmp_path / "nonexistent.json"
        result = OpticalSystem.load(str(path))
        assert result is None


class TestLensRayTracer3D:
    """Test openlens.tracer_3d.LensRayTracer3D.trace_surface"""

    def test_trace_surface_on_axis_front_hit(self):
        """Test on-axis hit on front surface of lens."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0))
        initial_opl = ray.optical_path_length
        
        result = tracer.trace_surface(ray, "front", interaction="refract")
        
        assert result in (RefractionResult.REFRACTED, RefractionResult.REFLECTED)
        assert ray.origin.x == pytest.approx(0.0, abs=1e-6)
        assert ray.origin.y == pytest.approx(0.0, abs=1e-6)
        assert ray.optical_path_length > initial_opl

    def test_trace_surface_aperture_miss(self):
        """Test that ray outside aperture returns MISSED."""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5
        )
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10.0, 15.0, 0.0), vec3(1.0, 0.0, 0.0))
        initial_origin = vec3(ray.origin)
        initial_opl = ray.optical_path_length
        
        result = tracer.trace_surface(ray, "front")
        
        assert result == RefractionResult.MISSED
        assert ray.origin == initial_origin
        assert ray.optical_path_length == initial_opl
