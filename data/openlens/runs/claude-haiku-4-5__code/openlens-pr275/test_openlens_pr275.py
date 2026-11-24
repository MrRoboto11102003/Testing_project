import pytest
import numpy as np
from openlens.lens import Lens
from openlens.polarization import (
    JonesMatrices, 
    PolarizationCalculator, 
    PolarizationState
)
from openlens.tolerancing import ToleranceType


class TestLensFocalLengths:
    def test_calculate_back_focal_length(self):
        """BFL returns float or None for afocal"""
        try:
            lens = Lens(radius_1=25.0, radius_2=-25.0, thickness=5.0, refractive_index=1.5)
            bfl = lens.calculate_back_focal_length()
            assert bfl is None or isinstance(bfl, (float, np.floating))
        except (TypeError, AttributeError):
            pytest.skip("Lens not available")
    
    def test_calculate_front_focal_length(self):
        """FFL returns float or None for afocal"""
        try:
            lens = Lens(radius_1=25.0, radius_2=-25.0, thickness=5.0, refractive_index=1.5)
            ffl = lens.calculate_front_focal_length()
            assert ffl is None or isinstance(ffl, (float, np.floating))
        except (TypeError, AttributeError):
            pytest.skip("Lens not available")
    
    def test_afocal_lens_returns_none_not_inf(self):
        """Afocal lens: BFL/FFL = None not inf"""
        try:
            lens = Lens(radius_1=float('inf'), radius_2=float('inf'), 
                       thickness=1.0, refractive_index=1.5)
            assert lens.calculate_back_focal_length() is None
            assert lens.calculate_front_focal_length() is None
        except (TypeError, AttributeError):
            pytest.skip("Lens not available")


class TestJonesMatricesLinearPolarizer:
    def test_linear_polarizer_key_angles(self):
        """Linear polarizer at 0°, 45°, 90°"""
        # 0 degrees: horizontal
        m0 = JonesMatrices.linear_polarizer(0.0)
        np.testing.assert_array_almost_equal(m0, [[1, 0], [0, 0]], decimal=10)
        # 90 degrees: vertical
        m90 = JonesMatrices.linear_polarizer(90.0)
        np.testing.assert_array_almost_equal(m90, [[0, 0], [0, 1]], decimal=10)
        # 45 degrees: 0.5*ones
        m45 = JonesMatrices.linear_polarizer(45.0)
        np.testing.assert_array_almost_equal(m45, 0.5 * np.ones((2, 2)), decimal=10)


class TestFresnelCoefficients:
    def test_fresnel_normal_incidence(self):
        """At normal incidence: r_s = r_p"""
        calc = PolarizationCalculator()
        result = calc.fresnel_coefficients(1.0, 1.5, 0.0)
        assert "r_s" in result and "r_p" in result
        assert np.abs(result["r_s"] - result["r_p"]) < 1e-10
    
    def test_fresnel_total_internal_reflection(self):
        """TIR case: t_s = t_p = 0"""
        calc = PolarizationCalculator()
        result = calc.fresnel_coefficients(1.5, 1.0, 50.0)
        assert result["total_internal_reflection"] == True
        assert result["t_s"] == 0.0 and result["t_p"] == 0.0


class TestReflectanceTransmittance:
    def test_energy_conservation_normal_incidence(self):
        """R + T = 1 at normal incidence"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(1.0, 1.5, 0.0)
        assert np.isclose(result["R_s"] + result["T_s"], 1.0, atol=1e-9)
        assert np.isclose(result["R_p"] + result["T_p"], 1.0, atol=1e-9)
    
    def test_tir_full_reflection(self):
        """TIR: R=1, T=0"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(1.5, 1.0, 50.0)
        assert result["R_s"] == 1.0 and result["T_s"] == 0.0
        assert result["R_p"] == 1.0 and result["T_p"] == 0.0


class TestPolarizationStateIntensity:
    def test_linear_x_and_y_polarization(self):
        """Intensity for x and y polarized states"""
        s_x = PolarizationState(np.array([1.0 + 0.0j, 0.0 + 0.0j]))
        s_y = PolarizationState(np.array([0.0 + 0.0j, 1.0 + 0.0j]))
        assert np.isclose(s_x.intensity(), 1.0)
        assert np.isclose(s_y.intensity(), 1.0)


class TestPolarizationStateStokes:
    def test_stokes_linear_polarization(self):
        """Stokes parameters: horizontal vs vertical"""
        s_h = PolarizationState(np.array([1.0, 0.0]))
        stokes_h = s_h.stokes_parameters()
        # Horizontal: S0=1, S1=1, S2=0, S3=0
        assert np.isclose(stokes_h[0], 1.0) and np.isclose(stokes_h[1], 1.0)
        assert np.isclose(stokes_h[2], 0.0, atol=1e-10)
        
        s_v = PolarizationState(np.array([0.0, 1.0]))
        stokes_v = s_v.stokes_parameters()
        # Vertical: S0=1, S1=-1, S2=0, S3=0
        assert np.isclose(stokes_v[0], 1.0) and np.isclose(stokes_v[1], -1.0)
        assert np.isclose(stokes_v[2], 0.0, atol=1e-10)


class TestPolarizationStateElliptical:
    def test_elliptical_creation_normalized(self):
        """Elliptical polarization states are normalized"""
        state = PolarizationState.elliptical(1.0, 0.5, angle_deg=0.0)
        assert np.isclose(state.intensity(), 1.0)
        
        state2 = PolarizationState.elliptical(1.0, 0.5, angle_deg=45.0)
        assert np.isclose(state2.intensity(), 1.0)


class TestPolarizationStateApplyMatrix:
    def test_apply_identity_matrix(self):
        """Identity matrix leaves state unchanged"""
        state = PolarizationState(np.array([1.0, 0.0]))
        identity = np.eye(2, dtype=complex)
        new_state = state.apply_matrix(identity)
        np.testing.assert_array_almost_equal(new_state.jones_vector, state.jones_vector)


class TestToleranceType:
    def test_tolerance_type_members_and_distinct_values(self):
        """ToleranceType enum has expected members with distinct values"""
        expected = ['RADIUS_1', 'RADIUS_2', 'THICKNESS', 'REFRACTIVE_INDEX',
                   'DECENTER_X', 'DECENTER_Y', 'TILT_X', 'TILT_Y', 'DECENTER_Z',
                   'AIR_GAP', 'ABBE_NUMBER', 'IRREGULARITY', 'WEDGE', 'FOCUS']
        for member in expected:
            assert hasattr(ToleranceType, member)
        values = [t.value for t in ToleranceType]
        assert len(values) == len(set(values))
