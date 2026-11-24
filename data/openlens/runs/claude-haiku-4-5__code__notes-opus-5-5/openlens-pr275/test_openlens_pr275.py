import pytest
import numpy as np
from openlens.lens import Lens
from openlens.polarization import (
    JonesMatrices,
    PolarizationCalculator,
    PolarizationState,
)
from openlens.tolerancing import ToleranceType, COMPENSATOR_ONLY_TYPES


class TestLensBackFocalLength:
    """Tests for Lens.calculate_back_focal_length()"""

    def test_bfl_thick_lens_worked_example(self):
        """BFL for thick biconvex lens: R1=100, R2=-100, t=5, n=1.5"""
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            refractive_index=1.5,
        )
        bfl = lens.calculate_back_focal_length()
        assert bfl == pytest.approx(11800 / 119, rel=1e-9)  # ≈ 99.159664

    def test_bfl_afocal_returns_none(self):
        """BFL returns None for afocal lens (both surfaces flat)"""
        lens = Lens(
            radius_of_curvature_1=float("inf"),
            radius_of_curvature_2=float("inf"),
            thickness=5,
            refractive_index=1.5,
        )
        assert lens.calculate_back_focal_length() is None

    def test_bfl_flat_first_surface_equals_focal_length(self):
        """When R1 is infinite, BFL equals focal length"""
        lens = Lens(
            radius_of_curvature_1=float("inf"),
            radius_of_curvature_2=-100,
            thickness=5,
            refractive_index=1.5,
        )
        f = lens.calculate_focal_length()
        bfl = lens.calculate_back_focal_length()
        assert bfl == pytest.approx(f, rel=1e-9)

    def test_bfl_zero_thickness(self):
        """BFL equals focal length when thickness is zero"""
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=0,
            refractive_index=1.5,
        )
        f = lens.calculate_focal_length()
        bfl = lens.calculate_back_focal_length()
        assert bfl == pytest.approx(f, rel=1e-9)


class TestLensFrontFocalLength:
    """Tests for Lens.calculate_front_focal_length()"""

    def test_ffl_converging_lens_negative(self):
        """FFL is negative for converging lens: R1=100, R2=-100, t=5, n=1.5"""
        lens = Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            refractive_index=1.5,
        )
        ffl = lens.calculate_front_focal_length()
        assert ffl == pytest.approx(-11800 / 119, rel=1e-9)  # ≈ -99.159664

    def test_ffl_afocal_returns_none(self):
        """FFL returns None for afocal lens"""
        lens = Lens(
            radius_of_curvature_1=float("inf"),
            radius_of_curvature_2=float("inf"),
            thickness=5,
            refractive_index=1.5,
        )
        assert lens.calculate_front_focal_length() is None

    def test_ffl_flat_back_surface_negative_focal_length(self):
        """FFL = -f when back surface is flat"""
        lens = Lens(
            radius_of_curvature_1=50,
            radius_of_curvature_2=float("inf"),
            thickness=5,
            refractive_index=1.5,
        )
        f = lens.calculate_focal_length()
        ffl = lens.calculate_front_focal_length()
        assert ffl == pytest.approx(-f, rel=1e-9)


class TestLinearPolarizer:
    """Tests for JonesMatrices.linear_polarizer()"""

    def test_polarizer_0_degrees_horizontal(self):
        """Linear polarizer at 0° is horizontal projection"""
        M = JonesMatrices.linear_polarizer(0)
        expected = np.array([[1, 0], [0, 0]])
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_polarizer_45_degrees(self):
        """Linear polarizer at 45°"""
        M = JonesMatrices.linear_polarizer(45)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_polarizer_90_degrees_vertical(self):
        """Linear polarizer at 90° is vertical projection"""
        M = JonesMatrices.linear_polarizer(90)
        expected = np.array([[0, 0], [0, 1]])
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_polarizer_is_idempotent(self):
        """M @ M ≈ M (idempotent property)"""
        M = JonesMatrices.linear_polarizer(30)
        np.testing.assert_allclose(M @ M, M, atol=1e-12)

    def test_polarizer_is_real_dtype(self):
        """Linear polarizer matrix is real (not complex)"""
        M = JonesMatrices.linear_polarizer(45)
        assert np.isrealobj(M)
        assert M.dtype == np.float64


class TestFrenelCoefficients:
    """Tests for PolarizationCalculator.fresnel_coefficients()"""

    def test_fresnel_normal_incidence(self):
        """Normal incidence: r_s=-0.2, r_p=0.2, t_s=t_p=0.8"""
        calc = PolarizationCalculator()
        coeffs = calc.fresnel_coefficients(1.0, 1.5, 0.0)
        assert coeffs["r_s"] == pytest.approx(-0.2, abs=1e-9)
        assert coeffs["r_p"] == pytest.approx(0.2, abs=1e-9)
        assert coeffs["t_s"] == pytest.approx(0.8, abs=1e-9)
        assert coeffs["total_internal_reflection"] is False
        assert "theta_transmitted_deg" in coeffs

    def test_fresnel_brewster_angle_rp_zero(self):
        """At Brewster angle, r_p ≈ 0"""
        calc = PolarizationCalculator()
        brewster = np.degrees(np.arctan(1.5))
        coeffs = calc.fresnel_coefficients(1.0, 1.5, brewster)
        assert abs(coeffs["r_p"]) < 1e-10
        assert coeffs["total_internal_reflection"] is False

    def test_fresnel_total_internal_reflection(self):
        """TIR case: (1.5→1.0, 60°) gives t=0, |r|=1"""
        calc = PolarizationCalculator()
        coeffs = calc.fresnel_coefficients(1.5, 1.0, 60.0)
        assert coeffs["total_internal_reflection"] is True
        assert coeffs["t_s"] == 0j
        assert coeffs["t_p"] == 0j
        assert abs(coeffs["r_s"]) == pytest.approx(1.0, abs=1e-12)
        assert "theta_transmitted_deg" not in coeffs


class TestReflectanceTransmittance:
    """Tests for PolarizationCalculator.reflectance_transmittance()"""

    def test_reflectance_transmittance_normal_incidence(self):
        """Normal incidence: R=0.04, T=0.96"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(1.0, 1.5, 0.0)
        assert result["R_s"] == pytest.approx(0.04, abs=1e-9)
        assert result["R_p"] == pytest.approx(0.04, abs=1e-9)
        assert result["T_s"] == pytest.approx(0.96, abs=1e-9)

    def test_reflectance_transmittance_energy_conservation(self):
        """Energy conservation: R_s + T_s ≈ 1 at multiple angles"""
        calc = PolarizationCalculator()
        for angle in [0, 30, 45, 60]:
            result = calc.reflectance_transmittance(1.0, 1.5, angle)
            assert (result["R_s"] + result["T_s"]) == pytest.approx(1.0, abs=1e-9)
            assert (result["R_p"] + result["T_p"]) == pytest.approx(1.0, abs=1e-9)

    def test_reflectance_transmittance_tir(self):
        """TIR case: (1.5→1.0, 60°) gives R=1, T=0"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(1.5, 1.0, 60.0)
        assert result["R_s"] == 1.0
        assert result["R_p"] == 1.0
        assert result["T_s"] == 0.0
        assert result["T_p"] == 0.0


class TestApplyMatrix:
    """Tests for PolarizationState.apply_matrix()"""

    def test_apply_malus_law(self):
        """Horizontal light through 60° polarizer: intensity = cos²(60°) = 0.25"""
        h = PolarizationState.linear_horizontal()
        polarizer = JonesMatrices.linear_polarizer(60)
        result = h.apply_matrix(polarizer)
        assert result.intensity() == pytest.approx(0.25, abs=1e-9)

    def test_apply_matrix_returns_new_object(self):
        """apply_matrix returns a new distinct PolarizationState"""
        s = PolarizationState.linear_horizontal()
        result = s.apply_matrix(np.eye(2))
        assert isinstance(result, PolarizationState)
        assert result is not s

    def test_apply_crossed_polarizers_zero_intensity(self):
        """Horizontal through 0° then 90° polarizer gives ~0 intensity"""
        h = PolarizationState.linear_horizontal()
        p0 = JonesMatrices.linear_polarizer(0)
        p90 = JonesMatrices.linear_polarizer(90)
        result = h.apply_matrix(p0).apply_matrix(p90)
        assert result.intensity() == pytest.approx(0, abs=1e-12)


class TestElliptical:
    """Tests for PolarizationState.elliptical()"""

    def test_elliptical_circular_right(self):
        """elliptical(1, 1, 0) is circular right"""
        state = PolarizationState.elliptical(1, 1, 0)
        cr = PolarizationState.circular_right()
        np.testing.assert_allclose(state.jones_vector, cr.jones_vector, atol=1e-12)

    def test_elliptical_circular_left(self):
        """elliptical(1, -1, 0) is circular left"""
        state = PolarizationState.elliptical(1, -1, 0)
        cl = PolarizationState.circular_left()
        np.testing.assert_allclose(state.jones_vector, cl.jones_vector, atol=1e-12)

    def test_elliptical_normalized_intensity(self):
        """Elliptical states have unit intensity"""
        state = PolarizationState.elliptical(3, 4, 30)
        assert state.intensity() == pytest.approx(1.0, rel=1e-9)

    def test_elliptical_s3_sign_convention(self):
        """elliptical(1, 1) has S3 = -1 (circular right convention)"""
        state = PolarizationState.elliptical(1, 1, 0)
        stokes = state.stokes_parameters()
        assert stokes[3] == pytest.approx(-1.0, abs=1e-9)


class TestIntensity:
    """Tests for PolarizationState.intensity()"""

    def test_intensity_basic_magnitude_squared(self):
        """Intensity of [3, 4] is 3²+4²=25"""
        state = PolarizationState(np.array([3.0, 4.0]))
        assert state.intensity() == pytest.approx(25.0, abs=1e-12)

    def test_intensity_complex_components(self):
        """Intensity with complex: [3, 4j] gives |3|²+|4j|²=25"""
        state = PolarizationState(np.array([3.0, 4.0j]))
        assert state.intensity() == pytest.approx(25.0, abs=1e-12)

    def test_intensity_quadratic_scaling(self):
        """Scaling vector by k scales intensity by k²"""
        s1 = PolarizationState(np.array([1.0, 0.0]))
        s2 = PolarizationState(np.array([3.0, 0.0]))
        assert s2.intensity() == pytest.approx(9 * s1.intensity(), abs=1e-12)


class TestStokesParameters:
    """Tests for PolarizationState.stokes_parameters()"""

    def test_stokes_linear_horizontal(self):
        """Horizontal: [S0, S1, S2, S3] = [1, 1, 0, 0]"""
        state = PolarizationState.linear_horizontal()
        stokes = state.stokes_parameters()
        np.testing.assert_allclose(stokes, [1, 1, 0, 0], atol=1e-12)

    def test_stokes_linear_vertical(self):
        """Vertical: [S0, S1, S2, S3] = [1, -1, 0, 0]"""
        state = PolarizationState.linear_vertical()
        stokes = state.stokes_parameters()
        np.testing.assert_allclose(stokes, [1, -1, 0, 0], atol=1e-12)

    def test_stokes_circular_right_s3_negative(self):
        """Circular right has S3 = -1"""
        state = PolarizationState.circular_right()
        stokes = state.stokes_parameters()
        assert stokes[0] == pytest.approx(1.0, abs=1e-9)
        assert stokes[3] == pytest.approx(-1.0, abs=1e-9)

    def test_stokes_circular_left_s3_positive(self):
        """Circular left has S3 = +1"""
        state = PolarizationState.circular_left()
        stokes = state.stokes_parameters()
        assert stokes[0] == pytest.approx(1.0, abs=1e-9)
        assert stokes[3] == pytest.approx(1.0, abs=1e-9)

    def test_stokes_s0_equals_intensity(self):
        """S0 component equals intensity()"""
        state = PolarizationState.linear_45()
        stokes = state.stokes_parameters()
        assert stokes[0] == pytest.approx(state.intensity(), rel=1e-9)


class TestToleranceType:
    """Tests for ToleranceType enum"""

    def test_tolerance_type_exact_members(self):
        """ToleranceType has exactly 14 members"""
        members = list(ToleranceType)
        assert len(members) == 14

    def test_tolerance_type_exact_values(self):
        """Key members have exact values"""
        assert ToleranceType.RADIUS_1.value == "Radius 1"
        assert ToleranceType.THICKNESS.value == "Thickness"
        assert ToleranceType.REFRACTIVE_INDEX.value == "Refractive Index"
        assert ToleranceType.IRREGULARITY.value == "Surface Irregularity"
        assert ToleranceType.FOCUS.value == "Focus"

    def test_tolerance_type_decenter_z_after_tilt_y(self):
        """DECENTER_Z appears after TILT_Y in enum order"""
        members = list(ToleranceType)
        tilt_y_idx = members.index(ToleranceType.TILT_Y)
        decenter_z_idx = members.index(ToleranceType.DECENTER_Z)
        assert decenter_z_idx > tilt_y_idx

    def test_tolerance_type_round_trip_value(self):
        """ToleranceType(member.value) returns member"""
        for member in ToleranceType:
            assert ToleranceType(member.value) is member

    def test_tolerance_type_compensator_only_set(self):
        """FOCUS is the only member in COMPENSATOR_ONLY_TYPES"""
        assert ToleranceType.FOCUS in COMPENSATOR_ONLY_TYPES
        assert len(COMPENSATOR_ONLY_TYPES) == 1
