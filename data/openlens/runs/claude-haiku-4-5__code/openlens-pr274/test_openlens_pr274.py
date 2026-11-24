import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from types import MethodType

import openlens
from openlens.aberrations import analyze_lens_quality, AberrationsCalculator
from openlens.analysis.geometric import GeometricTraceAnalysis
from openlens.image_simulator import ImageSimulator
from openlens.optical_system import OpticalSystem
from openlens.tracer_3d import LensRayTracer3D, Ray3D, RefractionResult, REFRACTIVE_INDEX_AIR


class TestAnalyzeLensQuality:
    """Tests for openlens.aberrations.analyze_lens_quality"""
    
    def test_basic_lens_quality_returns_expected_keys(self):
        """Test that analyze_lens_quality returns expected dictionary structure."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'spherical': 0.001,
                'chromatic': 0.05,
                'distortion': 0.1,
                'astigmatism': 0.01,
                'error': None
            }):
                result = analyze_lens_quality(lens)
                
                assert isinstance(result, dict)
                assert 'quality_score' in result
                assert 'rating' in result
                assert 'issues' in result
                assert 'aberrations' in result
    
    def test_quality_score_valid_range(self):
        """Test that quality_score is in reasonable range [0-100]."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'spherical': 0.001, 'chromatic': 0.01, 'distortion': 0.01, 
                'astigmatism': 0.001, 'error': None
            }):
                result = analyze_lens_quality(lens)
                assert 0 <= result['quality_score'] <= 100
    
    def test_rating_in_valid_options(self):
        """Test that rating is one of the documented values."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'spherical': 0.0, 'chromatic': 0.0, 'distortion': 0.0, 
                'astigmatism': 0.0, 'error': None
            }):
                result = analyze_lens_quality(lens)
                assert result['rating'] in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor', 'Error']
    
    def test_field_angle_parameter_passed(self):
        """Test that field_angle parameter is used in calculation."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'spherical': 0.0, 'chromatic': 0.0, 'distortion': 0.0, 
                'astigmatism': 0.0, 'error': None
            }) as mock_calc:
                analyze_lens_quality(lens, field_angle=15.0)
                mock_calc.assert_called_once()
                assert mock_calc.call_args[1]['field_angle'] == 15.0
    
    def test_high_aberrations_reduce_score(self):
        """Test that high aberrations reduce quality score."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'spherical': 2.0, 'chromatic': 1.0, 'distortion': 15.0, 
                'astigmatism': 2.0, 'error': None
            }):
                result = analyze_lens_quality(lens)
                assert result['quality_score'] < 60
                assert len(result['issues']) > 0
    
    def test_error_handling_returns_zero_score(self):
        """Test that calculation errors are properly handled."""
        lens = Mock()
        
        with patch.object(AberrationsCalculator, '__init__', return_value=None):
            with patch.object(AberrationsCalculator, 'calculate_all_aberrations', return_value={
                'error': 'Calculation failed'
            }):
                result = analyze_lens_quality(lens)
                
                assert result['quality_score'] == 0
                assert result['rating'] == 'Error'
                assert len(result['issues']) > 0


class TestGeometricTraceAnalysis:
    """Tests for GeometricTraceAnalysis methods"""
    
    def test_calculate_field_curvature_distortion_returns_expected_structure(self):
        """Test that calculate_field_curvature_distortion returns expected keys."""
        analysis = Mock(spec=GeometricTraceAnalysis)
        analysis._apply_wavelength = Mock(return_value={'state': 'saved'})
        analysis._restore_lens_state = Mock()
        analysis._calculate_curvature_traced = Mock(return_value={
            'field_angles': np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
            'tangential_focus_shift': np.array([0, 0.01, 0.05, 0.1, 0.2]),
            'sagittal_focus_shift': np.array([0, 0.02, 0.08, 0.15, 0.3]),
            'distortion': np.array([0, 0.1, 0.4, 0.9, 1.6])
        })
        
        analysis.calculate_field_curvature_distortion = MethodType(
            GeometricTraceAnalysis.calculate_field_curvature_distortion,
            analysis
        )
        
        result = analysis.calculate_field_curvature_distortion(max_field_angle_deg=20.0, num_points=5)
        
        assert 'field_angles' in result
        assert 'tangential_focus_shift' in result
        assert 'sagittal_focus_shift' in result
        assert 'distortion' in result
        analysis._apply_wavelength.assert_called()
        analysis._restore_lens_state.assert_called()
    
    def test_calculate_field_curvature_wavelength_applied(self):
        """Test that wavelength is properly applied and restored."""
        analysis = Mock(spec=GeometricTraceAnalysis)
        analysis._apply_wavelength = Mock(return_value={'state': 'saved'})
        analysis._restore_lens_state = Mock()
        analysis._calculate_curvature_traced = Mock(return_value={
            'field_angles': np.array([0.0]), 'tangential_focus_shift': np.array([0]),
            'sagittal_focus_shift': np.array([0]), 'distortion': np.array([0])
        })
        
        analysis.calculate_field_curvature_distortion = MethodType(
            GeometricTraceAnalysis.calculate_field_curvature_distortion,
            analysis
        )
        
        analysis.calculate_field_curvature_distortion(wavelength_nm=589.0)
        
        analysis._apply_wavelength.assert_called_once_with(589.0)
        analysis._restore_lens_state.assert_called_once()
    
    def test_calculate_ray_fan_returns_expected_structure(self):
        """Test that calculate_ray_fan returns expected keys."""
        analysis = Mock(spec=GeometricTraceAnalysis)
        analysis._apply_wavelength = Mock(return_value={'state': 'saved'})
        analysis._restore_lens_state = Mock()
        analysis._calculate_ray_fan_traced = Mock(return_value={
            'pupil_coords': np.linspace(-1, 1, 21),
            'ray_errors_mm': np.zeros(21)
        })
        
        analysis.calculate_ray_fan = MethodType(GeometricTraceAnalysis.calculate_ray_fan, analysis)
        
        result = analysis.calculate_ray_fan(field_angle_deg=0.0, num_points=21)
        
        assert 'pupil_coords' in result
        assert 'ray_errors_mm' in result
        assert len(result['pupil_coords']) == 21
        assert len(result['ray_errors_mm']) == 21
    
    def test_calculate_ray_fan_pupil_axes(self):
        """Test ray fan with different pupil axes."""
        analysis = Mock(spec=GeometricTraceAnalysis)
        analysis._apply_wavelength = Mock(return_value={'state': 'saved'})
        analysis._restore_lens_state = Mock()
        analysis._calculate_ray_fan_traced = Mock(return_value={
            'pupil_coords': np.linspace(-1, 1, 11),
            'ray_errors_mm': np.zeros(11)
        })
        
        analysis.calculate_ray_fan = MethodType(GeometricTraceAnalysis.calculate_ray_fan, analysis)
        
        result_y = analysis.calculate_ray_fan(pupil_axis='y')
        result_z = analysis.calculate_ray_fan(pupil_axis='z')
        
        assert 'pupil_coords' in result_y
        assert 'pupil_coords' in result_z


class TestImageSimulator:
    """Tests for ImageSimulator.simulate_image"""
    
    def test_simulate_image_grayscale_input(self):
        """Test image simulation with grayscale input."""
        simulator = Mock(spec=ImageSimulator)
        simulator._calculate_image_distance = Mock(return_value=100.0)
        simulator._apply_aberrations = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_diffraction = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_chromatic_aberration = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_vignetting = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._calculate_image_metrics = Mock(return_value={'sharpness': 0.95})
        
        simulator.simulate_image = MethodType(ImageSimulator.simulate_image, simulator)
        
        input_image = np.ones((100, 100), dtype=np.float32)
        result = simulator.simulate_image(input_image, object_distance=1000.0)
        
        assert 'output_image' in result
        assert 'magnification' in result
        assert 'image_distance' in result
        assert 'metrics' in result
        assert result['output_image'].shape == input_image.shape
    
    def test_simulate_image_color_input(self):
        """Test image simulation with color (RGB) input."""
        simulator = Mock(spec=ImageSimulator)
        simulator._calculate_image_distance = Mock(return_value=100.0)
        simulator._apply_aberrations = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_diffraction = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_chromatic_aberration = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_vignetting = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._calculate_image_metrics = Mock(return_value={'sharpness': 0.95})
        
        simulator.simulate_image = MethodType(ImageSimulator.simulate_image, simulator)
        
        input_image = np.ones((100, 100, 3), dtype=np.float32)
        result = simulator.simulate_image(input_image, object_distance=500.0)
        
        simulator._apply_chromatic_aberration.assert_called()
        assert result['output_image'].shape == input_image.shape
    
    def test_simulate_image_auto_image_distance(self):
        """Test automatic image distance calculation when not provided."""
        simulator = Mock(spec=ImageSimulator)
        simulator._calculate_image_distance = Mock(return_value=150.0)
        simulator._apply_aberrations = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_diffraction = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_chromatic_aberration = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_vignetting = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._calculate_image_metrics = Mock(return_value={'sharpness': 0.95})
        
        simulator.simulate_image = MethodType(ImageSimulator.simulate_image, simulator)
        
        input_image = np.ones((50, 50))
        result = simulator.simulate_image(input_image, object_distance=1000.0, image_distance=None)
        
        simulator._calculate_image_distance.assert_called_once()
        assert result['image_distance'] == 150.0
    
    def test_simulate_image_magnification_calculation(self):
        """Test that magnification is correctly calculated."""
        simulator = Mock(spec=ImageSimulator)
        simulator._calculate_image_distance = Mock(return_value=50.0)
        simulator._apply_aberrations = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_diffraction = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_chromatic_aberration = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._apply_vignetting = Mock(side_effect=lambda x, *args, **kwargs: x)
        simulator._calculate_image_metrics = Mock(return_value={'sharpness': 0.95})
        
        simulator.simulate_image = MethodType(ImageSimulator.simulate_image, simulator)
        
        input_image = np.ones((100, 100))
        result = simulator.simulate_image(input_image, object_distance=100.0, image_distance=50.0)
        
        expected_magnification = -50.0 / 100.0
        assert result['magnification'] == expected_magnification


class TestOpticalSystem:
    """Tests for OpticalSystem methods"""
    
    def test_calculate_chromatic_aberration_structure(self):
        """Test that calculate_chromatic_aberration returns expected keys."""
        system = Mock(spec=OpticalSystem)
        system.elements = []
        system._calculate_system_matrix = Mock(return_value=(1.0, 100.0, 0.01, 1.0))
        
        system.calculate_chromatic_aberration = MethodType(
            OpticalSystem.calculate_chromatic_aberration,
            system
        )
        
        result = system.calculate_chromatic_aberration()
        
        assert 'longitudinal' in result
        assert 'bfl_F' in result
        assert 'bfl_d' in result
        assert 'bfl_C' in result
        assert 'f_F' in result
        assert 'f_d' in result
        assert 'f_C' in result
        assert 'corrected' in result
        assert isinstance(result['corrected'], bool)
    
    def test_chromatic_aberration_corrected_flag(self):
        """Test that corrected flag is set based on longitudinal CA magnitude."""
        system = Mock(spec=OpticalSystem)
        system.elements = []
        system._calculate_system_matrix = Mock(return_value=(1.0, 100.0, 0.01, 1.0))
        
        system.calculate_chromatic_aberration = MethodType(
            OpticalSystem.calculate_chromatic_aberration,
            system
        )
        
        result = system.calculate_chromatic_aberration()
        
        # With uniform matrix, BFLs should be equal, so longitudinal CA near 0
        assert result['longitudinal'] >= -0.1 and result['longitudinal'] <= 0.1
        assert result['corrected'] == (abs(result['longitudinal']) < 0.1)
    
    def test_optical_system_load_valid_json(self):
        """Test loading valid optical system from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {'elements': [], 'properties': {}}
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            with patch('openlens.optical_system.OpticalSystem.from_dict') as mock_from_dict:
                with patch('openlens.optical_system.reject_oversized_file'):
                    with patch('openlens.optical_system.json_safe_radii', side_effect=lambda x: x):
                        with patch('openlens.optical_system.reject_non_finite_numbers'):
                            mock_from_dict.return_value = Mock(spec=OpticalSystem)
                            result = OpticalSystem.load(temp_file)
                            mock_from_dict.assert_called_once()
                            assert result is not None
        finally:
            os.unlink(temp_file)
    
    def test_optical_system_load_nonexistent_file(self):
        """Test that loading nonexistent file returns None."""
        result = OpticalSystem.load('/nonexistent/file/path.json')
        assert result is None
    
    def test_optical_system_load_oversized_file(self):
        """Test that oversized file is rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_file = f.name
        
        try:
            with patch('openlens.optical_system.reject_oversized_file', side_effect=ValueError("File too large")):
                result = OpticalSystem.load(temp_file)
                assert result is None
        finally:
            os.unlink(temp_file)


class TestLensRayTracer3D:
    """Tests for LensRayTracer3D.trace_surface"""
    
    def test_trace_surface_invalid_surface_type(self):
        """Test tracing with invalid surface type returns MISSED."""
        tracer = Mock(spec=LensRayTracer3D)
        ray = Mock(spec=Ray3D)
        
        tracer.trace_surface = MethodType(LensRayTracer3D.trace_surface, tracer)
        
        result = tracer.trace_surface(ray, 'invalid', 'refract')
        
        assert result == RefractionResult.MISSED
    
    def test_trace_surface_front_surface_refraction(self):
        """Test ray tracing on front surface with refraction."""
        tracer = Mock(spec=LensRayTracer3D)
        tracer.front_center = np.array([0, 0, 0])
        tracer.front_is_flat = True
        tracer.front_is_parabolic = False
        tracer.front_is_conic = False
        tracer.optical_axis = np.array([1, 0, 0])
        tracer.front_vertex = np.array([0, 0, 0])
        tracer.R1 = 50.0
        tracer.n = 1.5
        tracer._intersect_plane = Mock(return_value=np.array([0, 0, 0]))
        tracer._aperture_sq_front = 25.0
        tracer.transform = Mock()
        
        ray = Mock(spec=Ray3D)
        ray.origin = np.array([-10, 0, 0])
        ray.path = []
        ray.n = REFRACTIVE_INDEX_AIR
        ray.optical_path_length = 0
        ray.refract_or_reflect = Mock(return_value=RefractionResult.REFRACTED)
        ray.__sub__ = lambda a, b: a.origin - b.origin
        
        tracer.trace_surface = MethodType(LensRayTracer3D.trace_surface, tracer)
        
        result = tracer.trace_surface(ray, 'front', 'refract')
        
        assert result in [RefractionResult.REFRACTED, RefractionResult.MISSED, RefractionResult.REFLECTED]
    
    def test_trace_surface_reflection_interaction(self):
        """Test ray tracing with reflection interaction."""
        tracer = Mock(spec=LensRayTracer3D)
        tracer.front_is_flat = True
        tracer.front_is_parabolic = False
        tracer.front_is_conic = False
        tracer.optical_axis = np.array([1, 0, 0])
        tracer.front_vertex = np.array([0, 0, 0])
        tracer._intersect_plane = Mock(return_value=np.array([0, 0, 0]))
        tracer._aperture_sq_front = 25.0
        
        ray = Mock(spec=Ray3D)
        ray.origin = np.array([-5, 0, 0])
        ray.path = []
        ray.n = REFRACTIVE_INDEX_AIR
        ray.optical_path_length = 0
        ray.reflect = Mock()
        
        tracer.trace_surface = MethodType(LensRayTracer3D.trace_surface, tracer)
        
        result = tracer.trace_surface(ray, 'front', 'reflect')
        
        assert result == RefractionResult.REFLECTED
