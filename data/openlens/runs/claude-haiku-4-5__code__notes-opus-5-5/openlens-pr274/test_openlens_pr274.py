import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from openlens.aberrations import analyze_lens_quality, AberrationsCalculator
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.geometric import GeometricTraceAnalysis
from openlens.image_simulator import ImageSimulator
from openlens.tracer_3d import LensRayTracer3D
from openlens.ray import Ray3D, RefractionResult
from openlens.vector3 import vec3
from openlens.constants import (
    WAVELENGTH_GREEN, WAVELENGTH_D_LINE, WAVELENGTH_F_LINE, WAVELENGTH_C_LINE,
    SPHERICAL_ABERRATION_EXCELLENT, QUALITY_EXCELLENT_THRESHOLD
)


class TestAnalyzeLensQuality:
    """Test analyze_lens_quality function"""
    
    def test_perfect_lens_all_zeros(self, monkeypatch):
        """Test perfect lens with all aberrations at zero yields Excellent"""
        fake_results = {
            "spherical": 0.0,
            "chromatic": 0.0,
            "distortion": 0.0,
            "astigmatism": 0.0,
        }
        monkeypatch.setattr(
            AberrationsCalculator, "calculate_all_aberrations",
            lambda self, **kw: fake_results
        )
        
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        result = analyze_lens_quality(lens)
        
        assert result["quality_score"] == 100
        assert result["rating"] == "Excellent"
        assert result["issues"] == ["No significant aberrations detected"]
        assert result["aberrations"] is fake_results
    
    def test_error_path_zero_focal_length(self, monkeypatch):
        """Test error branch when focal length calculation fails"""
        monkeypatch.setattr(
            AberrationsCalculator, "calculate_all_aberrations",
            lambda self, **kw: {"error": "Cannot calculate focal length (zero optical power)"}
        )
        
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        result = analyze_lens_quality(lens)
        
        assert result["quality_score"] == 0
        assert result["rating"] == "Error"
        assert "aberrations" not in result
    
    def test_field_angle_kwarg_override(self, monkeypatch):
        """Test that field_angle_deg kwarg overrides field_angle positional"""
        seen = {}
        fake_results = {
            "spherical": 0.0, "chromatic": 0.0, 
            "distortion": 0.0, "astigmatism": 0.0
        }
        
        def capture_kwargs(self, **kw):
            seen.update(kw)
            return fake_results
        
        monkeypatch.setattr(AberrationsCalculator, "calculate_all_aberrations", capture_kwargs)
        
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        analyze_lens_quality(lens, field_angle=3.0, field_angle_deg=7.0)
        
        assert seen["field_angle"] == 7.0


class TestCalculateFieldCurvatureDistortion:
    """Test GeometricTraceAnalysis.calculate_field_curvature_distortion"""
    
    def test_return_structure_and_aliases(self):
        """Test return dict has correct keys and aliases"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = GeometricTraceAnalysis(system)
        
        result = analyzer.calculate_field_curvature_distortion(
            max_field_angle_deg=10.0, num_points=3
        )
        
        required_keys = {
            "field_angles_deg", "tan_focus_shift_mm", "sag_focus_shift_mm",
            "tan_focus_shift", "sag_focus_shift", "distortion_pct", "image_plane_x_mm"
        }
        assert set(result.keys()) == required_keys
        assert result["tan_focus_shift"] is result["tan_focus_shift_mm"]
        assert result["sag_focus_shift"] is result["sag_focus_shift_mm"]
        assert len(result["field_angles_deg"]) == 3
    
    def test_on_axis_zero_distortion(self):
        """Test on-axis (angle=0) has exactly zero distortion"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = GeometricTraceAnalysis(system)
        
        result = analyzer.calculate_field_curvature_distortion(
            max_field_angle_deg=0.0, num_points=1
        )
        
        assert result["distortion_pct"][0] == 0.0
    
    def test_unexpected_kwargs_raises_typeerror(self):
        """Test that unexpected kwargs raise TypeError with proper message"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0)
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = GeometricTraceAnalysis(system)
        
        with pytest.raises(TypeError, match=r"Unexpected kwargs"):
            analyzer.calculate_field_curvature_distortion(unknown_kwarg=42)


class TestCalculateRayFan:
    """Test GeometricTraceAnalysis.calculate_ray_fan"""
    
    def test_return_structure_and_aliases(self):
        """Test return dict structure with correct aliases"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = GeometricTraceAnalysis(system)
        
        result = analyzer.calculate_ray_fan(num_points=11)
        
        required_keys = {
            "pupil_coords", "ray_errors_mm", "ray_errors", "transverse_aberration",
            "axis", "field_angle_deg", "wavelength_nm"
        }
        assert set(result.keys()) == required_keys
        assert result["ray_errors"] is result["ray_errors_mm"]
        assert result["transverse_aberration"] is result["ray_errors_mm"]
        assert len(result["pupil_coords"]) <= 11
    
    def test_on_axis_center_zero_error(self):
        """Test that center pupil point (p=0) has exactly zero error on-axis"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_lens(lens)
        analyzer = GeometricTraceAnalysis(system)
        
        result = analyzer.calculate_ray_fan(field_angle_deg=0.0, num_points=11)
        
        assert 0.0 in result["pupil_coords"]
        idx = result["pupil_coords"].index(0.0)
        assert result["ray_errors_mm"][idx] == 0.0


class TestSimulateImage:
    """Test ImageSimulator.simulate_image"""
    
    def test_identity_monochrome_zero_pitch(self):
        """Test identity case: constant monochrome image with zero pitch"""
        fake_lens = MagicMock()
        fake_lens.refractive_index = 1.5
        fake_lens.diameter = 25.0
        fake_lens.refractive_index_at = MagicMock(return_value=1.5)
        
        fake_elem = MagicMock()
        fake_elem.lens = fake_lens
        
        fake_system = MagicMock()
        fake_system.elements = [fake_elem]
        fake_system.get_system_focal_length = MagicMock(return_value=100.0)
        fake_system.get_aberrations = MagicMock(return_value={})
        
        simulator = ImageSimulator(fake_system)
        input_img = np.ones((32, 32), dtype=np.float32)
        
        result = simulator.simulate_image(
            input_img, object_distance=200.0, pixel_pitch_mm=0.0
        )
        
        assert result["output_image"].shape == input_img.shape
        assert result["magnification"] == pytest.approx(-1.0)
        assert result["image_distance"] == pytest.approx(200.0)
        assert "metrics" in result
    
    def test_zero_object_distance_raises(self):
        """Test that zero object distance raises ZeroDivisionError"""
        fake_system = MagicMock()
        fake_system.elements = [MagicMock(lens=MagicMock(diameter=25.0))]
        
        simulator = ImageSimulator(fake_system)
        input_img = np.ones((16, 16))
        
        with pytest.raises(ZeroDivisionError):
            simulator.simulate_image(input_img, object_distance=0.0)


class TestCalculateChromaticAberration:
    """Test OpticalSystem.calculate_chromatic_aberration"""
    
    def test_empty_system_returns_fallback(self):
        """Test empty system returns fallback dict"""
        system = OpticalSystem()
        result = system.calculate_chromatic_aberration()
        
        assert result == {"longitudinal": 0.0, "corrected": False}
    
    def test_single_lens_returns_full_structure(self):
        """Test single lens returns full 8-key structure"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, material="BK7")
        system = OpticalSystem()
        system.add_lens(lens)
        
        result = system.calculate_chromatic_aberration()
        
        expected_keys = {"longitudinal", "bfl_F", "bfl_d", "bfl_C", 
                        "f_F", "f_d", "f_C", "corrected"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["corrected"], bool)
    
    def test_pinned_indices_nonzero_longitudinal(self):
        """Test with pinned indices shows wavelength dependence"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        
        indices = {
            WAVELENGTH_F_LINE: 1.522,
            WAVELENGTH_D_LINE: 1.5168,
            WAVELENGTH_C_LINE: 1.514,
        }
        lens.refractive_index_at = lambda wl: indices.get(wl, 1.5)
        
        system = OpticalSystem()
        system.add_lens(lens)
        
        result = system.calculate_chromatic_aberration()
        
        assert result["bfl_F"] is not None
        assert result["bfl_C"] is not None


class TestOpticalSystemLoad:
    """Test OpticalSystem.load"""
    
    def test_round_trip_preserves_system(self, tmp_path):
        """Test save and load round-trip preserves structure"""
        lens1 = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                    thickness=5.0, diameter=25.0, name="L1")
        lens2 = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                    thickness=5.0, diameter=25.0, name="L2")
        
        system = OpticalSystem(name="TestSys")
        system.add_lens(lens1)
        system.add_lens(lens2, air_gap_before=10.0)
        
        filepath = tmp_path / "test_sys.json"
        assert system.save(str(filepath))
        
        loaded = OpticalSystem.load(str(filepath))
        
        assert loaded is not None
        assert loaded.name == "TestSys"
        assert len(loaded.elements) == 2
        assert loaded.elements[0].lens.name == "L1"
        assert loaded.elements[1].lens.name == "L2"
    
    def test_missing_file_returns_none(self, tmp_path):
        """Test that missing file returns None, not raises"""
        result = OpticalSystem.load(str(tmp_path / "does_not_exist.json"))
        assert result is None
    
    def test_invalid_json_returns_none(self, tmp_path):
        """Test that invalid JSON returns None"""
        filepath = tmp_path / "invalid.json"
        filepath.write_text("{ not valid json")
        
        result = OpticalSystem.load(str(filepath))
        assert result is None


class TestTraceSurface:
    """Test LensRayTracer3D.trace_surface"""
    
    def test_on_axis_front_surface_hit(self):
        """Test on-axis ray hitting front surface and refracting"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, refractive_index=1.5)
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        initial_opl = ray.optical_path_length
        
        result = tracer.trace_surface(ray, "front")
        
        assert result == RefractionResult.REFRACTED
        assert ray.origin == pytest.approx(vec3(0, 0, 0), abs=1e-6)
        assert ray.optical_path_length > initial_opl
    
    def test_aperture_miss_state_unchanged(self):
        """Test that ray outside aperture returns MISSED with state unchanged"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, refractive_index=1.5)
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10, 13, 0), vec3(1, 0, 0))
        initial_origin = vec3(ray.origin.x, ray.origin.y, ray.origin.z)
        initial_opl = ray.optical_path_length
        
        result = tracer.trace_surface(ray, "front")
        
        assert result == RefractionResult.MISSED
        assert ray.origin == initial_origin
        assert ray.optical_path_length == initial_opl
    
    def test_invalid_surface_type_returns_missed(self):
        """Test that invalid surface_type returns MISSED"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0)
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        result = tracer.trace_surface(ray, "middle")
        
        assert result == RefractionResult.MISSED
    
    def test_reflect_interaction(self):
        """Test reflection interaction returns REFLECTED"""
        lens = Lens(radius_of_curvature_1=50.0, radius_of_curvature_2=-50.0,
                   thickness=5.0, diameter=25.0, refractive_index=1.5)
        tracer = LensRayTracer3D(lens)
        
        ray = Ray3D(vec3(-10, 0, 0), vec3(1, 0, 0))
        result = tracer.trace_surface(ray, "front", interaction="reflect")
        
        assert result == RefractionResult.REFLECTED
        assert ray.origin == pytest.approx(vec3(0, 0, 0), abs=1e-6)
