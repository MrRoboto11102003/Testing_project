import pytest
import os
from openlens.material_database import MaterialDatabase


class TestImportAGFCatalog:
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary MaterialDatabase for testing."""
        db_path = str(tmp_path / "mats.json")
        return MaterialDatabase(db_path=db_path)
    
    def test_nonexistent_file(self, temp_db):
        """Test that importing a nonexistent file returns 0."""
        result = temp_db.import_agf_catalog("/nonexistent/path/catalog.agf")
        assert result == 0
    
    def test_empty_file(self, temp_db, tmp_path):
        """Test that importing an empty file returns 0 and doesn't save."""
        agf_file = tmp_path / "empty.agf"
        agf_file.write_text("", encoding="utf-8")
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert not os.path.exists(temp_db.db_path)
    
    def test_single_valid_glass(self, temp_db, tmp_path):
        """Test importing a single valid glass with correct properties."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM MYGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        assert "MYGLASS" in temp_db.materials
        mat = temp_db.get_material("MYGLASS")
        assert mat.nd == pytest.approx(1.5168, abs=1e-4)
        assert mat.vd == pytest.approx(64.17, abs=1e-2)
        assert mat.catalog == "AGF_Import"
        # Check de-interleaved coefficients: B1=c[0], C1=c[1], B2=c[2], C2=c[3], B3=c[4], C3=c[5]
        assert mat.B2 == pytest.approx(0.231792344)
        assert mat.C1 == pytest.approx(6.00069867e-3)
    
    def test_formula_validation(self, temp_db, tmp_path):
        """Test that invalid formula (not 1 or 2) is skipped."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM GLASS1 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "NM GLASS2 3 517642 1.6 50 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "NM GLASS3 1 517642 1.52 55 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 2
        assert "GLASS1" in temp_db.materials
        assert "GLASS2" not in temp_db.materials
        assert "GLASS3" in temp_db.materials
    
    def test_insufficient_dispersion_coefficients(self, temp_db, tmp_path):
        """Test that glasses with fewer than 6 CD coefficients are skipped."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM GLASS1 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert "GLASS1" not in temp_db.materials
    
    def test_lowercase_keys(self, temp_db, tmp_path):
        """Test that lowercase keys (nm, cd, etc.) are accepted."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "nm LOWGLASS 2 517642 1.51 65 7.0 2.5\n"
            "cd 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        assert "LOWGLASS" in temp_db.materials
    
    def test_quoted_names(self, temp_db, tmp_path):
        """Test that quoted names are handled correctly."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            'NM "QUOTED GLASS" 2 517642 1.52 60 7.1 2.5\n'
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        mat = temp_db.get_material("QUOTED GLASS")
        assert mat.name == "QUOTED GLASS"
    
    def test_tce_conversion(self, temp_db, tmp_path):
        """Test that TCE is converted from 1e-6/K to K^-1."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM TCETEST 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        mat = temp_db.get_material("TCETEST")
        assert mat.thermal_expansion == pytest.approx(7.1e-6)
    
    def test_ed_line_overrides_tce(self, temp_db, tmp_path):
        """Test that ED line can override TCE from NM line."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM EDTEST 2 517642 1.5 60 5.0 2.5\n"
            "ED 8.0 0 0 2.6\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        mat = temp_db.get_material("EDTEST")
        assert mat.thermal_expansion == pytest.approx(8.0e-6)
    
    def test_it_lines_transmission_data(self, temp_db, tmp_path):
        """Test that IT lines for transmission are accumulated correctly."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM ITGLASS 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "IT 400 0.9 10\n"
            "IT 500 0.95 25\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        mat = temp_db.get_material("ITGLASS")
        assert len(mat.transmission_data) == 2
        assert [float(x) for x in mat.transmission_data[0]] == [400.0, 0.9]
        assert [float(x) for x in mat.transmission_data[1]] == [500.0, 0.95]
        assert mat.transmission_thickness == pytest.approx(25.0)
    
    def test_it_line_invalid_transmission(self, temp_db, tmp_path):
        """Test that IT lines with transmission out of [0, 1] are ignored."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM INVALIDIT 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "IT 400 1.5 10\n"
            "IT 500 0.95 25\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 1
        mat = temp_db.get_material("INVALIDIT")
        assert len(mat.transmission_data) == 1
        assert [float(x) for x in mat.transmission_data[0]] == [500.0, 0.95]
    
    def test_multiple_glasses(self, temp_db, tmp_path):
        """Test importing multiple glasses correctly."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM GLASS1 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "NM GLASS2 2 517642 1.6 50 6.5 2.3\n"
            "CD 1.1 0.6 0.4 0.25 0.15 0.08\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 2
        assert "GLASS1" in temp_db.materials
        assert "GLASS2" in temp_db.materials
        assert temp_db.get_material("GLASS1").nd == pytest.approx(1.5)
        assert temp_db.get_material("GLASS2").nd == pytest.approx(1.6)
    
    def test_glass_with_no_cd_line(self, temp_db, tmp_path):
        """Test that a glass without CD line (0 coefficients) is skipped."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM NOCD 2 517642 1.5 60 7.1 2.5\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert "NOCD" not in temp_db.materials
    
    def test_malformed_nm_line_error_recovery(self, temp_db, tmp_path):
        """Test that malformed NM lines are skipped but parsing continues."""
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM GLASS1 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n"
            "NM BADGLASS 2\n"
            "NM GLASS2 2 517642 1.52 65 7.0 2.4\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = temp_db.import_agf_catalog(str(agf_file))
        assert result == 2
        assert "GLASS1" in temp_db.materials
        assert "BADGLASS" not in temp_db.materials
        assert "GLASS2" in temp_db.materials
    
    def test_database_saved_on_success(self, temp_db, tmp_path):
        """Test that the database file is saved when count > 0."""
        db_path = str(tmp_path / "new_db.json")
        db = MaterialDatabase(db_path=db_path)
        agf_file = tmp_path / "test.agf"
        agf_file.write_text(
            "NM TESTGLASS 2 517642 1.5 60 7.1 2.5\n"
            "CD 1.0 0.5 0.3 0.2 0.1 0.05\n",
            encoding="utf-8"
        )
        result = db.import_agf_catalog(str(agf_file))
        assert result == 1
        assert os.path.exists(db_path)
        db2 = MaterialDatabase(db_path=db_path)
        assert "TESTGLASS" in db2.materials
        assert db2.get_material("TESTGLASS").nd == pytest.approx(1.5)
