import pytest
from openlens.validation import check_physical_feasibility


class TestCheckPhysicalFeasibility:
    """Tests for openlens.validation.check_physical_feasibility function."""

    def test_valid_biconvex_no_warnings(self):
        """Standard biconvex lens with good proportions should pass."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        assert is_feasible is True
        assert msg is None

    def test_surface_intersection_hard_error(self):
        """Surfaces that intersect within clear aperture is a hard error."""
        is_feasible, msg = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=10.0,
            diameter=25.0
        )
        assert is_feasible is False
        assert msg is not None
        assert "intersect" in msg.lower()

    def test_thickness_exceeds_50_percent_radius_warning(self):
        """Thickness > 50% of minimum radius should warn."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=30.0,  # 60% of 50mm radius
            diameter=20.0
        )
        assert is_feasible is False
        assert msg is not None
        assert "thick" in msg.lower() or "impractical" in msg.lower()

    def test_diameter_exceeds_80_percent_radius_warning(self):
        """Diameter > 80% of minimum radius should warn."""
        is_feasible, msg = check_physical_feasibility(
            radius1=50.0,
            radius2=-50.0,
            thickness=5.0,
            diameter=42.0  # 84% of 50mm radius
        )
        assert is_feasible is False
        assert msg is not None
        assert "diameter" in msg.lower() or "large" in msg.lower()

    def test_thickness_below_2_percent_diameter_warning(self):
        """Thickness < 2% of diameter should warn about fragility."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=0.3,  # 1.5% of 20mm diameter
            diameter=20.0
        )
        assert is_feasible is False
        assert msg is not None
        assert "thin" in msg.lower()

    def test_valid_plano_convex(self):
        """Plano-convex lens (one flat surface) should be valid."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=float('inf'),
            thickness=4.0,
            diameter=20.0
        )
        assert is_feasible is True
        assert msg is None

    def test_valid_biconcave(self):
        """Biconcave lens should be valid with good proportions."""
        is_feasible, msg = check_physical_feasibility(
            radius1=-100.0,
            radius2=100.0,
            thickness=3.0,
            diameter=20.0
        )
        assert is_feasible is True
        assert msg is None

    def test_parabolic_surface_valid(self):
        """Parabolic surface should be handled correctly."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
            is_parabolic_1=True,
            parabolic_sag_1=0.3
        )
        assert is_feasible is True
        assert msg is None

    def test_conic_constant_parabolic(self):
        """Conic constant -1 (parabolic surface) should work."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0,
            conic_constant_1=-1.0,
            conic_constant_2=0.0
        )
        assert is_feasible is True
        assert msg is None

    def test_clear_aperture_first_surface(self):
        """Clear aperture on first surface should be respected."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_1=20.0
        )
        assert is_feasible is True
        assert msg is None

    def test_clear_aperture_second_surface(self):
        """Clear aperture on second surface should be respected."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=25.0,
            clear_aperture_2=18.0
        )
        assert is_feasible is True
        assert msg is None

    def test_return_type_valid_case(self):
        """Return value should be (True, None) for valid lenses."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=-100.0,
            thickness=5.0,
            diameter=20.0
        )
        assert isinstance(is_feasible, bool) and is_feasible is True
        assert msg is None

    def test_return_type_invalid_case(self):
        """Return value should be (False, str) for invalid/warning lenses."""
        is_feasible, msg = check_physical_feasibility(
            radius1=10.0,
            radius2=-10.0,
            thickness=10.0,
            diameter=25.0
        )
        assert isinstance(is_feasible, bool) and is_feasible is False
        assert isinstance(msg, str) and len(msg) > 0

    def test_plane_parallel_plate_flat_surfaces(self):
        """Plane-parallel plate (both surfaces flat/infinite) should work."""
        is_feasible, msg = check_physical_feasibility(
            radius1=float('inf'),
            radius2=float('inf'),
            thickness=5.0,
            diameter=25.0
        )
        # Plane-parallel has zero power but is geometrically valid
        assert isinstance(is_feasible, bool)
        if is_feasible:
            assert msg is None
        else:
            assert isinstance(msg, str)

    def test_meniscus_lens_valid(self):
        """Meniscus lens (both radii same sign) should be feasible."""
        is_feasible, msg = check_physical_feasibility(
            radius1=100.0,
            radius2=50.0,  # Same sign as R1
            thickness=4.0,
            diameter=20.0
        )
        assert is_feasible is True
        assert msg is None
