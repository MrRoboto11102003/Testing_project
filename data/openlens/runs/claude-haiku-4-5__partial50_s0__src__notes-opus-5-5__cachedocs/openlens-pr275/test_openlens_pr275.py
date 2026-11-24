import pytest
import numpy as np
from openlens.lens import Lens
from openlens.polarization import (
    JonesMatrices,
    PolarizationState,
    PolarizationCalculator,
)
from openlens.tolerancing import ToleranceType, COMPENSATOR_ONLY_TYPES
import enum


class TestLensFocalLengths:
    """Test Lens focal length calculations"""

    def test_bfl_basic_cases(self):
        """BFL for biconvex, afocal, flat surfaces, and negative lenses"""
        # Biconvex
        lens1 = Lens(radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0, 
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        assert lens1.calculate_back_focal_length() == pytest.approx(11800.0 / 119, rel=1e-9)
        
        # Afocal (both surfaces flat)
        lens2 = Lens(radius_of_curvature_1=float("inf"), radius_of_curvature_2=float("inf"),
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        assert lens2.calculate_back_focal_length() is None
        
        # Flat front: BFL = f
        lens3 = Lens(radius_of_curvature_1=float("inf"), radius_of_curvature_2=-100.0,
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        bfl3 = lens3.calculate_back_focal_length()
        f3 = lens3.calculate_focal_length()
        assert bfl3 == pytest.approx(f3, rel=1e-9)
        
        # Negative diverging lens
        lens4 = Lens(radius_of_curvature_1=-100.0, radius_of_curvature_2=100.0,
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        bfl4 = lens4.calculate_back_focal_length()
        assert bfl4 is not None and bfl4 < 0

    def test_ffl_symmetry_and_diverging(self):
        """FFL for symmetric lens (negative of BFL) and diverging lenses"""
        # Symmetric: FFL ≈ -BFL
        lens1 = Lens(radius_of_curvature_1=100.0, radius_of_curvature_2=-100.0,
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        ffl1 = lens1.calculate_front_focal_length()
        bfl1 = lens1.calculate_back_focal_length()
        assert ffl1 == pytest.approx(-bfl1, rel=1e-9)
        
        # Diverging: positive FFL
        lens2 = Lens(radius_of_curvature_1=-100.0, radius_of_curvature_2=100.0,
                     thickness=5.0, diameter=25.0, refractive_index=1.5)
        ffl2 = lens2.calculate_front_focal_length()
        assert ffl2 is not None and ffl2 > 0


class TestPolarizationMatrices:
    """Test Jones matrices and polarization operations"""

    def test_linear_polarizer_angles(self):
        """Linear polarizers at 0°, 45°, and 90°"""
        # 0°: transmits horizontal
        P0 = JonesMatrices.linear_polarizer(0.0)
        np.testing.assert_allclose(P0, [[1, 0], [0, 0]], atol=1e-14)
        
        # 45°
        P45 = JonesMatrices.linear_polarizer(45.0)
        np.testing.assert_allclose(P45, [[0.5, 0.5], [0.5, 0.5]], atol=1e-12)
        
        # 90°
        P90 = JonesMatrices.linear_polarizer(90.0)
        assert np.allclose(P90[1, 1], 1.0, atol=1e-14)
        assert np.allclose(P90[0, 0], 0.0, atol=1e-12)

    def test_polarizer_properties(self):
        """Polarizers are idempotent with trace = 1 and determinant = 0"""
        for angle in [0, 30, 45, 60, 90]:
            M = JonesMatrices.linear_polarizer(float(angle))
            # Idempotent: M @ M = M
            np.testing.assert_allclose(M @ M, M, atol=1e-12)
            # Trace = 1
            assert np.allclose(np.trace(M), 1.0, atol=1e-12)
            # Determinant = 0 (rank 1 projection)
            assert np.allclose(np.linalg.det(M), 0, atol=1e-12)


class TestFresnelOpticalProperties:
    """Test Fresnel equations and reflectance properties"""

    def test_fresnel_coefficients_normal_and_tir(self):
        """Fresnel at normal incidence and total internal reflection"""
        calc = PolarizationCalculator()
        
        # Normal incidence n1=1, n2=1.5
        result = calc.fresnel_coefficients(1.0, 1.5, 0.0)
        assert result["r_s"] == pytest.approx(-0.2, abs=1e-9)
        assert result["r_p"] == pytest.approx(0.2, abs=1e-9)
        assert result["total_internal_reflection"] is False
        assert "theta_transmitted_deg" in result
        
        # TIR at 60° (n1=1.5, n2=1)
        tir = calc.fresnel_coefficients(1.5, 1.0, 60.0)
        assert tir["total_internal_reflection"] is True
        assert abs(tir["r_s"]) == pytest.approx(1.0, abs=1e-12)
        assert tir["t_s"] == 0j
        assert "theta_transmitted_deg" not in tir

    def test_reflectance_energy_conservation_and_tir(self):
        """Energy conservation R + T = 1; TIR returns exact values"""
        calc = PolarizationCalculator()
        
        # Energy conservation at multiple angles
        for angle in [0, 20, 45, 70]:
            result = calc.reflectance_transmittance(1.0, 1.5, float(angle))
            assert result["R_s"] + result["T_s"] == pytest.approx(1.0, abs=1e-11)
            assert result["R_p"] + result["T_p"] == pytest.approx(1.0, abs=1e-11)
        
        # TIR exact dict
        tir = calc.reflectance_transmittance(1.5, 1.0, 60.0)
        assert tir == {"R_s": 1.0, "R_p": 1.0, "T_s": 0.0, "T_p": 0.0}
        
        # Grazing incidence (90°) has zero transmission
        grazing = calc.reflectance_transmittance(1.0, 1.5, 90.0)
        assert grazing["T_s"] == 0.0 and grazing["T_p"] == 0.0


class TestPolarizationStates:
    """Test polarization state generation and optical properties"""

    def test_malus_law(self):
        """Malus law verification: I = I₀ cos²(Δθ)"""
        s45 = PolarizationState.linear_45()
        for angle in [0, 30, 45, 60]:
            M = JonesMatrices.linear_polarizer(float(angle))
            result = s45.apply_matrix(M)
            expected = np.cos(np.radians(45 - angle)) ** 2
            assert result.intensity() == pytest.approx(expected, abs=1e-11)

    def test_elliptical_and_intensity(self):
        """Elliptical(1,1) equals circular_right; intensity consistency"""
        ell = PolarizationState.elliptical(1.0, 1.0)
        cr = PolarizationState.circular_right()
        np.testing.assert_allclose(ell.jones_vector, cr.jones_vector, atol=1e-12)
        
        # Factory states have intensity ≈ 1 and intensity == S0
        for state in [PolarizationState.linear_horizontal(), 
                      PolarizationState.linear_angle(30),
                      PolarizationState.elliptical(2, 1, 45)]:
            assert state.intensity() == pytest.approx(1.0, abs=1e-12)
            s = state.stokes_parameters()
            assert state.intensity() == pytest.approx(s[0], rel=1e-12)

    def test_stokes_parameters_key_states(self):
        """Key polarization states have known Stokes parameters"""
        cases = [
            (PolarizationState.linear_horizontal(), [1, 1, 0, 0]),
            (PolarizationState.linear_vertical(), [1, -1, 0, 0]),
            (PolarizationState.circular_right(), [1, 0, 0, -1]),
            (PolarizationState.circular_left(), [1, 0, 0, 1]),
        ]
        for state, expected_stokes in cases:
            np.testing.assert_allclose(state.stokes_parameters(), 
                                      expected_stokes, atol=1e-12)

    def test_fully_polarized_identity(self):
        """Fully polarized states satisfy S0² = S1² + S2² + S3²"""
        for angle in [15, 30, 60]:
            state = PolarizationState.linear_angle(float(angle))
            s = state.stokes_parameters()
            s0, s1, s2, s3 = s
            assert s0 ** 2 == pytest.approx(s1 ** 2 + s2 ** 2 + s3 ** 2, abs=1e-11)


class TestToleranceTypeEnum:
    """Test ToleranceType enum structure and functionality"""

    def test_enum_members_values_ordering(self):
        """ToleranceType has 14 unique members; DECENTER_Z after TILT_Y"""
        assert len(ToleranceType) == 14
        
        # All values unique
        values = [t.value for t in ToleranceType]
        assert len(values) == len(set(values))
        
        # Key members verified
        assert ToleranceType.RADIUS_1.value == "Radius 1"
        assert ToleranceType.THICKNESS.value == "Thickness"
        assert ToleranceType.FOCUS.value == "Focus"
        assert ToleranceType.IRREGULARITY.value == "Surface Irregularity"
        
        # Ordering: DECENTER_Z after TILT_Y
        members = list(ToleranceType)
        assert members.index(ToleranceType.DECENTER_Z) > members.index(ToleranceType.TILT_Y)

    def test_round_trip_serialization_compensator_only(self):
        """Round trip by value and name; FOCUS in COMPENSATOR_ONLY_TYPES"""
        for t in ToleranceType:
            assert ToleranceType(t.value) is t
            assert ToleranceType[t.name] is t
        
        # FOCUS is compensator-only
        assert ToleranceType.FOCUS in COMPENSATOR_ONLY_TYPES
        assert isinstance(ToleranceType.RADIUS_1, enum.Enum)
        assert not isinstance(ToleranceType.RADIUS_1, int)
