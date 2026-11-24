import pytest
from openlens.material_database import MaterialDatabase, MaterialProperties


class TestMaterialProperties:
    """Test MaterialProperties dataclass."""

    def test_construction_and_defaults(self):
        """MaterialProperties with required fields has all correct defaults."""
        m = MaterialProperties("TestGlass", "Custom", 1.5, 60.0)
        
        # Check key fields
        assert m.name == "TestGlass"
        assert m.nd == 1.5
        assert m.vd == 60.0
        # Check defaults
        assert m.B1 == 0.0
        assert m.thermal_expansion == 7.1e-6
        assert m.transmission_data == []
        assert m.transmission_thickness == 10.0
        assert m.density == 2.51

    def test_to_dict_and_defensive_copy(self):
        """to_dict() returns all fields and does not share references."""
        m = MaterialProperties(
            "Glass1", "Schott", 1.52, 64.0,
            B1=1.0, transmission_data=[(500, 0.99)],
            transmission_thickness=5.0, density=2.6
        )
        
        d = m.to_dict()
        
        assert isinstance(d, dict)
        assert d["name"] == "Glass1"
        assert d["B1"] == 1.0
        assert d["transmission_data"] == [(500, 0.99)]
        
        # Verify no shared reference
        d["transmission_data"].append((600, 0.98))
        assert m.transmission_data == [(500, 0.99)]

    def test_from_dict_and_round_trip(self):
        """from_dict creates equal instance; round trip preserves data."""
        m1 = MaterialProperties(
            "Glass1", "Custom", 1.5, 60.0,
            B1=0.5, transmission_data=[(400, 0.9), (600, 0.95)],
            transmission_thickness=8.0
        )
        
        d = m1.to_dict()
        m2 = MaterialProperties.from_dict(d)
        
        assert m1 == m2
        assert m2.B1 == 0.5
        assert m2.transmission_data == [(400, 0.9), (600, 0.95)]

    def test_constructor_and_from_dict_type_errors(self):
        """Missing required field raises TypeError."""
        with pytest.raises(TypeError):
            MaterialProperties("GlassName", "Custom", 1.5)  # missing vd
        
        with pytest.raises(TypeError):
            MaterialProperties.from_dict({"name": "X", "catalog": "C", "nd": 1.5})

    def test_per_instance_default_lists_separate(self):
        """Each instance gets its own transmission_data list."""
        m1 = MaterialProperties("A", "Custom", 1.5, 60.0)
        m2 = MaterialProperties("B", "Custom", 1.5, 60.0)
        
        m1.transmission_data.append((500, 0.99))
        
        assert m1.transmission_data == [(500, 0.99)]
        assert m2.transmission_data == []


class TestMaterialDatabaseClearCache:
    """Test MaterialDatabase.clear_cache() method."""

    def test_clear_cache_resets_state(self, tmp_path):
        """clear_cache() resets _index_cache and _alias_keys."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        db.get_refractive_index("BK7", 587.6)
        assert len(db._index_cache) > 0
        
        db.clear_cache()
        
        assert db._index_cache == {}
        assert db._alias_keys is None

    def test_clear_cache_per_instance_isolation(self, tmp_path):
        """Clearing one instance's cache does not affect another."""
        db1 = MaterialDatabase(db_path=str(tmp_path / "test1.json"))
        db2 = MaterialDatabase(db_path=str(tmp_path / "test2.json"))
        
        db1.get_refractive_index("BK7", 587.6)
        db2.get_refractive_index("BK7", 587.6)
        
        db1.clear_cache()
        
        assert db1._index_cache == {}
        assert len(db2._index_cache) > 0

    def test_clear_cache_staleness_and_alias_reset(self, tmp_path):
        """Clear cache allows fresh recomputation and rebuilds alias map."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        # Cache initial value
        before = db.get_refractive_index("BK7", 587.6)
        
        # Replace material with zero coefficients (n ≈ 1.0)
        db.materials["BK7"] = MaterialProperties(
            name="BK7", catalog="X", nd=1.6, vd=40.0
        )
        
        # Stale cache
        stale = db.get_refractive_index("BK7", 587.6)
        assert stale == pytest.approx(before, rel=1e-3)
        
        # Clear cache; recompute and rebuild aliases
        db.clear_cache()
        fresh = db.get_refractive_index("BK7", 587.6)
        assert fresh == pytest.approx(1.0, abs=1e-3)
        
        # Alias also rebuilt
        m = MaterialProperties(name="FOO_BAR", catalog="X", nd=1.5, vd=50.0)
        db.materials["FOO_BAR"] = m
        db.clear_cache()
        assert db.get_material("foo bar") is m


class TestMaterialDatabaseGetTransmission:
    """Test MaterialDatabase.get_transmission() method."""

    def test_bk7_exact_value_and_thickness_scaling(self, tmp_path):
        """BK7 at 500nm exact; scales by thickness via Beer-Lambert."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        # Exact value at default thickness
        assert db.get_transmission("BK7", 500.0) == 0.995
        
        # Scaled at 20mm: 0.995^(20/10)
        result_thick = db.get_transmission("BK7", 500.0, thickness_mm=20.0)
        expected = 0.995 ** 2
        assert result_thick == pytest.approx(expected, rel=1e-6)

    def test_bk7_interpolation_and_clamping(self, tmp_path):
        """BK7 interpolates between points; clamps at boundaries."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        # Interpolate between (400, 0.97) and (500, 0.995) at 450nm
        result = db.get_transmission("BK7", 450.0)
        expected = 0.97 + 0.5 * 0.025
        assert result == pytest.approx(expected, rel=1e-6)
        
        # Clamp below range (first point at 380nm, T=0.95)
        assert db.get_transmission("BK7", 300.0) == 0.95

    def test_fallback_for_material_without_data(self, tmp_path):
        """Material without transmission data uses fallback 0.99 per 10mm."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        # N-BK7 has no data
        assert db.get_transmission("N-BK7", 500.0) == pytest.approx(0.99)
        
        # Fallback scales with thickness
        result_thick = db.get_transmission("N-BK7", 500.0, thickness_mm=20.0)
        assert result_thick == pytest.approx(0.99 ** 2, rel=1e-6)

    def test_custom_material_with_transmission_data(self, tmp_path):
        """Custom material with specific transmission and thickness."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        custom = MaterialProperties(
            name="CUSTOM", catalog="Custom", nd=1.5, vd=60.0,
            transmission_data=[(500, 0.81)], transmission_thickness=5.0
        )
        db.add_material(custom, save=False)
        
        # At 10mm: 0.81^(10/5)
        result = db.get_transmission("CUSTOM", 500.0, thickness_mm=10.0)
        assert result == pytest.approx(0.81 ** 2, abs=1e-6)

    def test_zero_thickness_and_exact_endpoints(self, tmp_path):
        """Zero thickness returns 1.0; exact endpoints work."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        # Zero thickness: T^(0/10) = 1.0
        assert db.get_transmission("BK7", 500.0, thickness_mm=0.0) == 1.0
        
        # FUSEDSILICA at exact point (1000, 0.998)
        assert db.get_transmission("FUSEDSILICA", 1000.0) == 0.998

    def test_case_insensitive_names(self, tmp_path):
        """Material names are case-insensitive via alias."""
        db = MaterialDatabase(db_path=str(tmp_path / "test.json"))
        
        result_upper = db.get_transmission("BK7", 500.0)
        result_lower = db.get_transmission("bk7", 500.0)
        
        assert result_upper == result_lower == 0.995
