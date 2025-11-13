# -*- coding: utf-8 -*-
"""
Hash Utils 테스트
유틸리티 hash_utils 모듈 단위 테스트
"""

import pytest
import hashlib
import json

from lawfirm_langgraph.core.utils.hash_utils import (
    canonicalize_record,
    compute_hash_from_text,
    compute_hash_from_record
)


class TestHashUtils:
    """Hash Utils 테스트"""
    
    def test_canonicalize_record_basic(self):
        """기본 레코드 정규화 테스트"""
        record = {"b": 2, "a": 1, "c": 3}
        result = canonicalize_record(record)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2, "c": 3}
    
    def test_canonicalize_record_unicode(self):
        """유니코드 문자 포함 레코드 정규화 테스트"""
        record = {"한글": "테스트", "english": "test"}
        result = canonicalize_record(record)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"english": "test", "한글": "테스트"}
    
    def test_canonicalize_record_nested(self):
        """중첩된 레코드 정규화 테스트"""
        record = {
            "outer": {
                "inner": "value"
            },
            "list": [1, 2, 3]
        }
        result = canonicalize_record(record)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == record
    
    def test_canonicalize_record_deterministic(self):
        """정규화 결과의 일관성 테스트"""
        record1 = {"b": 2, "a": 1}
        record2 = {"a": 1, "b": 2}
        
        result1 = canonicalize_record(record1)
        result2 = canonicalize_record(record2)
        
        assert result1 == result2
    
    def test_compute_hash_from_text_basic(self):
        """기본 텍스트 해시 계산 테스트"""
        text = "test string"
        result = compute_hash_from_text(text)
        
        assert isinstance(result, str)
        assert len(result) == 64
        
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected
    
    def test_compute_hash_from_text_unicode(self):
        """유니코드 텍스트 해시 계산 테스트"""
        text = "한글 테스트"
        result = compute_hash_from_text(text)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_compute_hash_from_text_empty(self):
        """빈 문자열 해시 계산 테스트"""
        text = ""
        result = compute_hash_from_text(text)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_compute_hash_from_text_consistency(self):
        """동일 텍스트의 해시 일관성 테스트"""
        text = "test"
        result1 = compute_hash_from_text(text)
        result2 = compute_hash_from_text(text)
        
        assert result1 == result2
    
    def test_compute_hash_from_record_basic(self):
        """기본 레코드 해시 계산 테스트"""
        record = {"a": 1, "b": 2}
        result = compute_hash_from_record(record)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_compute_hash_from_record_consistency(self):
        """동일 레코드의 해시 일관성 테스트"""
        record1 = {"b": 2, "a": 1}
        record2 = {"a": 1, "b": 2}
        
        result1 = compute_hash_from_record(record1)
        result2 = compute_hash_from_record(record2)
        
        assert result1 == result2
    
    def test_compute_hash_from_record_different(self):
        """다른 레코드의 해시 차이 테스트"""
        record1 = {"a": 1, "b": 2}
        record2 = {"a": 1, "b": 3}
        
        result1 = compute_hash_from_record(record1)
        result2 = compute_hash_from_record(record2)
        
        assert result1 != result2
    
    def test_compute_hash_from_record_nested(self):
        """중첩된 레코드 해시 계산 테스트"""
        record = {
            "outer": {
                "inner": "value"
            },
            "list": [1, 2, 3]
        }
        result = compute_hash_from_record(record)
        
        assert isinstance(result, str)
        assert len(result) == 64

