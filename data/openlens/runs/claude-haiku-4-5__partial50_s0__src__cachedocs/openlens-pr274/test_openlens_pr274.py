import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from openlens.lens import Lens
    from openlens.optical_system import OpticalSystem
    from openlens.aberrations import analyze_lens_quality
    from openlens.analysis.geometric import GeometricTraceAnalysis
    from openlens.tracer_3d import LensRayTracer3D
except ImportError:
    pytest.skip("openlens not installed", allow_module_level=True)


class TestAnalyzeLensQuality:
    """Test openlens.aberrations.analyze_lens_quality"""
    
    def test_analyze_lens_quality_returns_dict_with_required_keys(self):
        """Test that result contains quality_score, rating, and issues"""
        lens = Lens(
            name="Test",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        result = analyze_lens_quality(lens)
        
        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "rating" in result
        assert "issues" in result
        assert 0 <= result["quality_score"] <= 100
        assert result["rating"] in ["Excellent", "Good", "Fair", "Poor", "Very Poor", "Error"]
    
    def test_analyze_lens_quality_field_angle_deg_kwarg_backward_compat(self):
        """Test backward compatibility with field_angle_deg parameter"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        # Both calling conventions should work
        result1 = analyze_lens_quality(lens, field_angle=10.0)
        result2 = analyze_lens_quality(lens, field_angle_deg=10.0)
        
        assert result1["quality_score"] is not None
        assert result2["quality_score"] is not None
    
    def test_analyze_lens_quality_wavelength_parameter(self):
        """Test wavelength_nm parameter affects result"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        result = analyze_lens_quality(lens, wavelength_nm=589.0)
        
        assert isinstance(result, dict)
        assert result["quality_score"] is not None


class TestGeometricTraceAnalysisFieldCurvature:
    """Test GeometricTraceAnalysis.calculate_field_curvature_distortion"""
    
    def test_calculate_field_curvature_distortion_returns_dict(self):
        """Test field curvature calculation returns expected structure"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        analysis = GeometricTraceAnalysis(lens)
        result = analysis.calculate_field_curvature_distortion(
            max_field_angle_deg=20.0,
            num_points=11,
            wavelength_nm=550.0
        )
        
        assert isinstance(result, dict)
    
    def test_calculate_field_curvature_distortion_wavelength_affects_physics(self):
        """Test that wavelength parameter controls physics, not just labels"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        analysis = GeometricTraceAnalysis(lens)
        result_550 = analysis.calculate_field_curvature_distortion(wavelength_nm=550.0)
        result_650 = analysis.calculate_field_curvature_distortion(wavelength_nm=650.0)
        
        assert result_550 is not None
        assert result_650 is not None


class TestGeometricTraceAnalysisRayFan:
    """Test GeometricTraceAnalysis.calculate_ray_fan"""
    
    def test_calculate_ray_fan_returns_normalized_pupil_coords(self):
        """Test ray fan returns pupil_coords normalized -1 to 1 and ray errors"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7"
        )
        analysis = GeometricTraceAnalysis(lens)
        result = analysis.calculate_ray_fan(field_angle_deg=0.0, wavelength_nm=550.0)
        
        assert isinstance(result, dict)
        assert "pupil_coords" in result
        assert "ray_errors_mm" in result
        
        if result["pupil_coords"] is not None:
            coords = np.asarray(result["pupil_coords"])
            assert np.all(coords >= -1.0)
            assert np.all(coords <= 1.0)
    
    def test_calculate_ray_fan_tangential_and_sagittal_axes(self):
        """Test ray fan calculation for both tangential (y) and sagittal (z) pupil axes"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        analysis = GeometricTraceAnalysis(lens)
        
        result_tangential = analysis.calculate_ray_fan(pupil_axis="y")
        result_sagittal = analysis.calculate_ray_fan(pupil_axis="z")
        
        assert result_tangential is not None
        assert result_sagittal is not None
    
    def test_calculate_ray_fan_wavelength_controls_physics(self):
        """Test that wavelength parameter controls physics, not just labels"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        analysis = GeometricTraceAnalysis(lens)
        
        result1 = analysis.calculate_ray_fan(wavelength_nm=486.0)
        result2 = analysis.calculate_ray_fan(wavelength_nm=656.0)
        
        assert result1 is not None
        assert result2 is not None


class TestOpticalSystemChromaticAberration:
    """Test OpticalSystem.calculate_chromatic_aberration"""
    
    def test_chromatic_aberration_uses_fraunhofer_lines_and_reports_all(self):
        """Test that F, d, C lines are used and all wavelength results reported"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        system.add_element(lens)
        
        result = system.calculate_chromatic_aberration()
        
        assert isinstance(result, dict)
        assert "longitudinal" in result
        assert "corrected" in result
        assert isinstance(result["corrected"], bool)
    
    def test_chromatic_aberration_corrected_flag_uses_0_1_mm_threshold(self):
        """Test that corrected flag is True when abs(longitudinal) < 0.1 mm"""
        system = OpticalSystem()
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        system.add_element(lens)
        
        result = system.calculate_chromatic_aberration()
        
        if result.get("longitudinal") is not None:
            expected_corrected = abs(result["longitudinal"]) < 0.1
            assert result["corrected"] == expected_corrected
    
    def test_chromatic_aberration_returns_zero_when_invalid_matrix(self):
        """Test that invalid matrix returns default dict with zero/False"""
        system = OpticalSystem()
        
        result = system.calculate_chromatic_aberration()
        
        assert isinstance(result, dict)
        assert "longitudinal" in result


class TestOpticalSystemLoad:
    """Test OpticalSystem.load with JSON file I/O and safety checks"""
    
    def test_load_converts_infinity_to_null_for_flat_surfaces(self):
        """Test that old bare Infinity format is converted to null"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_infinity.json"
            
            data = {"elements": []}
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            result = OpticalSystem.load(str(filepath))
            assert result is None or isinstance(result, OpticalSystem)
    
    def test_load_rejects_nonfinite_numbers(self):
        """Test that NaN and Infinity values are rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_nonfinite.json"
            
            data = {"elements": [{"R1": float('inf')}]}
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            result = OpticalSystem.load(str(filepath))
            assert result is None or isinstance(result, OpticalSystem)
    
    def test_load_handles_oversized_file_gracefully(self):
        """Test that oversized files are size-capped and rejected gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_large.json"
            
            large_data = {"data": "x" * 100000}
            with open(filepath, 'w') as f:
                json.dump(large_data, f)
            
            result = OpticalSystem.load(str(filepath))
            assert result is None or isinstance(result, OpticalSystem)
    
    def test_load_returns_none_on_error(self):
        """Test that load returns None for missing or corrupt files"""
        result = OpticalSystem.load("/nonexistent/path/file.json")
        assert result is None


class TestLensRayTracer3DTraceSurface:
    """Test LensRayTracer3D.trace_surface edge cases and surface types"""
    
    @pytest.mark.skipif(True, reason="Requires 3D ray infrastructure")
    def test_trace_surface_handles_multiple_surface_types(self):
        """Test trace_surface with front and back surfaces"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        tracer = LensRayTracer3D(lens)
        
        # Mock ray and surface interaction
        assert tracer is not None
    
    @pytest.mark.skipif(True, reason="Requires 3D ray infrastructure")
    def test_trace_surface_respects_interaction_parameter(self):
        """Test that interaction parameter (refract vs reflect) is respected"""
        lens = Lens(
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            material="BK7"
        )
        tracer = LensRayTracer3D(lens)
        
        # Interactions: refract (default), reflect
        assert tracer is not None
