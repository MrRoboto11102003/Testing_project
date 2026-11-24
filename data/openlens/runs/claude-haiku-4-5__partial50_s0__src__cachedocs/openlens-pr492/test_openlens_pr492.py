import pytest
import tempfile
import os
from unittest.mock import patch
from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseImportAGFCatalog:
    """Test suite for MaterialDatabase.import_agf_catalog() method."""

    def test_import_agf_nonexistent_file_returns_zero(self):
        """File not found should return 0."""
        db = MaterialDatabase()
        result = db.import_agf_catalog("/nonexistent/path/to/file.agf")
        assert result == 0
        assert isinstance(result, int)

    def test_import_agf_empty_file_returns_zero(self):
        """Empty file should return 0 without errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, 'save_database'):
                result = db.import_agf_catalog(temp_path)
            assert result == 0
        finally:
            os.unlink(temp_path)

    def test_import_agf_single_valid_material(self):
        """Valid NM line should import one material."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True):
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)
            assert result == 1
        finally:
            os.unlink(temp_path)

    def test_import_agf_multiple_materials(self):
        """Multiple NM lines should import multiple materials."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\nNM SF2 2 601613 1.6204 36.37 7.2 2.75\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)
            assert result == 2
            assert mock_process.call_count == 2
        finally:
            os.unlink(temp_path)

    def test_import_agf_with_dispersion_coefficients(self):
        """CD line should populate dispersion field with six coefficients."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\nCD 1.03961 0.0060 0.2318 0.02002 1.0105 103.56\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            assert result == 1
            glass_dict = mock_process.call_args[0][0]
            assert len(glass_dict['dispersion']) == 6
        finally:
            os.unlink(temp_path)

    def test_import_agf_with_thermal_data_ed_line(self):
        """ED line should override NM TCE and set density."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\nED 7.3 1.0 2.4 2.75\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            glass_dict = mock_process.call_args[0][0]
            assert glass_dict['tce'] == 7.3 * 1e-6
            assert glass_dict['density'] == 2.75
        finally:
            os.unlink(temp_path)

    def test_import_agf_with_transmission_data(self):
        """IT lines should populate transmission list with wavelength/transmission tuples."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\nIT 587.6 0.95 10.0\nIT 656.3 0.96 10.0\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            glass_dict = mock_process.call_args[0][0]
            assert len(glass_dict['transmission']) == 2
            assert glass_dict['transmission'][0] == (587.6, 0.95)
            assert glass_dict['transmission'][1] == (656.3, 0.96)
        finally:
            os.unlink(temp_path)

    def test_import_agf_invalid_transmission_outside_range(self):
        """Transmission values outside [0,1] should be ignored with warning."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\nIT 587.6 1.5 10.0\nIT 656.3 0.96 10.0\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            glass_dict = mock_process.call_args[0][0]
            assert len(glass_dict['transmission']) == 1
            assert glass_dict['transmission'][0][1] == 0.96
        finally:
            os.unlink(temp_path)

    def test_import_agf_malformed_nm_line_skipped(self):
        """Malformed NM line should be skipped; subsequent valid lines imported."""
        content = "NM N-BK7 1 517642\nNM SF2 2 601613 1.6204 36.37 7.2 2.75\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True):
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)
            assert result == 1
        finally:
            os.unlink(temp_path)

    def test_import_agf_blank_lines_and_whitespace_handled(self):
        """Blank lines and extra whitespace should be properly ignored."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\n\n  \nCD 1.03961 0.0060\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)
            assert result == 1
            glass_dict = mock_process.call_args[0][0]
            assert 'dispersion' in glass_dict
        finally:
            os.unlink(temp_path)

    def test_import_agf_save_not_called_if_zero_materials(self):
        """save_database should not be called if no materials are successfully imported."""
        content = "NM N-BK7 1 517642\n"  # Malformed line

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=False):
                with patch.object(db, 'save_database') as mock_save:
                    result = db.import_agf_catalog(temp_path)
            assert result == 0
            mock_save.assert_not_called()
        finally:
            os.unlink(temp_path)

    def test_import_agf_save_called_when_materials_imported(self):
        """save_database should be called when materials are successfully imported."""
        content = "NM N-BK7 1 517642 1.5168 64.17 2.3 2.51\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True):
                with patch.object(db, 'save_database') as mock_save:
                    result = db.import_agf_catalog(temp_path)
            assert result == 1
            mock_save.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_import_agf_quoted_name_handling(self):
        """Quoted material names should have quotes stripped."""
        content = 'NM "N-BK7" 1 517642 1.5168 64.17 2.3 2.51\n'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            glass_dict = mock_process.call_args[0][0]
            assert glass_dict['name'] == 'N-BK7'
        finally:
            os.unlink(temp_path)

    def test_import_agf_tce_si_units_conversion(self):
        """TCE from NM line should be converted from 1e-6/K to K^-1."""
        content = "NM N-BK7 1 517642 1.5168 64.17 7.3 2.51\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database'):
                    result = db.import_agf_catalog(temp_path)

            glass_dict = mock_process.call_args[0][0]
            assert glass_dict['tce'] == 7.3 * 1e-6
        finally:
            os.unlink(temp_path)

    def test_import_agf_complete_realistic_catalog(self):
        """Complete realistic AGF catalog with multiple materials and all data types."""
        content = """NM N-BK7 1 517642 1.5168 64.17 2.3 2.51
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
ED 7.3 1.0 2.4 2.51
IT 587.6 0.95 10.0
IT 656.3 0.96 10.0
NM SF2 2 601613 1.6204 36.37 7.2 2.75
CD 1.47343127 0.00926681282 0.216901417 0.0424658612 0.904803580 105.613573
IT 656.3 0.96 10.0
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            db = MaterialDatabase()
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                with patch.object(db, 'save_database') as mock_save:
                    result = db.import_agf_catalog(temp_path)

            assert result == 2
            assert mock_process.call_count == 2
            mock_save.assert_called_once()
        finally:
            os.unlink(temp_path)
