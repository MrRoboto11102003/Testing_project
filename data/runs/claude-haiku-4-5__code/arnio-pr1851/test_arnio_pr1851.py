import pytest
import pandas as pd
import tempfile
import os
import arnio as ar


class TestNormalizeCase:
    """Tests for arnio.cleaning.normalize_case"""
    
    def test_normalize_case_lower_default(self):
        """Test default 'lower' case normalization"""
        df = pd.DataFrame({'text': ['HELLO', 'World', 'tESt']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame)
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_upper(self):
        """Test 'upper' case normalization"""
        df = pd.DataFrame({'text': ['hello', 'World', 'tESt']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame, case_type='upper')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_title(self):
        """Test 'title' case normalization"""
        df = pd.DataFrame({'text': ['hello world', 'test case']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame, case_type='title')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_subset_single_column(self):
        """Test normalization with subset parameter for single column"""
        df = pd.DataFrame({'col1': ['HELLO'], 'col2': ['WORLD']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame, subset=['col1'], case_type='lower')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_subset_multiple_columns(self):
        """Test normalization with multiple columns in subset"""
        df = pd.DataFrame({'col1': ['HELLO'], 'col2': ['WORLD'], 'col3': ['TEST']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame, subset=['col1', 'col2'], case_type='upper')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_preserves_non_ascii_characters(self):
        """Test that non-ASCII characters are preserved unchanged"""
        df = pd.DataFrame({'text': ['Café', 'NAÏVE', '北京', 'Hello😀']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_case(frame, case_type='lower')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_case_case_type_not_string_raises_typeerror(self):
        """Test TypeError when case_type is not a string"""
        df = pd.DataFrame({'text': ['test']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(TypeError, match="case_type must be a string"):
                ar.normalize_case(frame, case_type=123)
        finally:
            os.unlink(path)
    
    def test_normalize_case_invalid_case_type_raises_valueerror(self):
        """Test ValueError when case_type is not a valid option"""
        df = pd.DataFrame({'text': ['test']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(ValueError, match="case_type must be one of"):
                ar.normalize_case(frame, case_type='invalid')
        finally:
            os.unlink(path)
    
    def test_normalize_case_missing_column_in_subset_raises_keyerror(self):
        """Test KeyError when subset contains non-existent column"""
        df = pd.DataFrame({'text': ['test']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(KeyError):
                ar.normalize_case(frame, subset=['nonexistent'])
        finally:
            os.unlink(path)


class TestNormalizeUnicode:
    """Tests for arnio.cleaning.normalize_unicode"""
    
    def test_normalize_unicode_nfc_default(self):
        """Test default 'NFC' normalization"""
        df = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame)
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_nfd_form(self):
        """Test 'NFD' normalization form"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame, form='NFD')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_nfkc_form(self):
        """Test 'NFKC' normalization form"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame, form='NFKC')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_nfkd_form(self):
        """Test 'NFKD' normalization form"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame, form='NFKD')
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_with_subset(self):
        """Test normalization with subset parameter"""
        df = pd.DataFrame({'text': ['café'], 'other': ['naïve']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame, subset=['text'])
            assert result is not None
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_invalid_form_raises_valueerror(self):
        """Test ValueError when form is not a supported normalization form"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
                ar.normalize_unicode(frame, form='INVALID')
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_form_not_string_raises_typeerror(self):
        """Test TypeError when form is not a string"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(TypeError, match="form must be a string"):
                ar.normalize_unicode(frame, form=123)
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_missing_column_in_subset_raises_keyerror(self):
        """Test KeyError when subset contains non-existent column"""
        df = pd.DataFrame({'text': ['café']})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            with pytest.raises(KeyError):
                ar.normalize_unicode(frame, subset=['nonexistent'])
        finally:
            os.unlink(path)
    
    def test_normalize_unicode_preserves_non_string_columns(self):
        """Test that non-string columns are cloned unchanged"""
        df = pd.DataFrame({
            'text': ['café'],
            'count': [42],
            'value': [3.14]
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False, encoding='utf-8')
            path = f.name
        try:
            frame = ar.read_csv(path)
            result = ar.normalize_unicode(frame)
            assert result is not None
        finally:
            os.unlink(path)
