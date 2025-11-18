#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_utils 모듈 단위 테스트
"""

import pytest

from scripts.utils.text_utils import (
    extract_keywords,
    normalize_text,
    remove_special_chars,
    extract_legal_terms,
    calculate_text_similarity
)


class TestTextUtils:
    """text_utils 모듈 테스트"""
    
    def test_extract_keywords(self):
        """키워드 추출 테스트"""
        query = "계약 해지 사유에 대해 알려주세요"
        keywords = extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "계약" in keywords
        assert "해지" in keywords
        assert "사유" in keywords
        assert "에" not in keywords  # 조사 제거 확인
        assert "알려주세요" not in keywords  # 불필요한 단어 제거 확인
    
    def test_extract_keywords_empty(self):
        """빈 쿼리 키워드 추출 테스트"""
        keywords = extract_keywords("")
        assert keywords == []
    
    def test_normalize_text(self):
        """텍스트 정규화 테스트"""
        text = "  여러   공백이    있는   텍스트  "
        normalized = normalize_text(text)
        
        assert normalized == "여러 공백이 있는 텍스트"
        assert "  " not in normalized
    
    def test_remove_special_chars(self):
        """특수문자 제거 테스트"""
        text = "테스트!@#$%^&*()텍스트"
        cleaned = remove_special_chars(text, keep_spaces=True)
        
        assert "!" not in cleaned
        assert "@" not in cleaned
        assert "테스트" in cleaned
        assert "텍스트" in cleaned
    
    def test_extract_legal_terms(self):
        """법률 용어 추출 테스트"""
        text = "민법 제123조에 따라 계약법 규칙을 적용합니다"
        terms = extract_legal_terms(text)
        
        assert isinstance(terms, set)
        assert len(terms) > 0
    
    def test_calculate_text_similarity(self):
        """텍스트 유사도 계산 테스트"""
        text1 = "계약 해지 사유"
        text2 = "계약 해지 사유"
        similarity = calculate_text_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity == 1.0  # 동일한 텍스트는 유사도 1.0
    
    def test_calculate_text_similarity_different(self):
        """서로 다른 텍스트 유사도 테스트"""
        text1 = "계약 해지"
        text2 = "완전히 다른 내용"
        similarity = calculate_text_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0

