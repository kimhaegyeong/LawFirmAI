# -*- coding: utf-8 -*-
"""
Legal Term Normalizer 테스트
데이터 legal_term_normalizer 모듈 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.data.legal_term_normalizer import (
    LegalTermNormalizer,
    LegalTerm
)


class TestLegalTermNormalizer:
    """Legal Term Normalizer 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer is not None
        assert hasattr(normalizer, 'term_mappings')
        assert hasattr(normalizer, 'patterns')
        assert isinstance(normalizer.term_mappings, dict)
        assert isinstance(normalizer.patterns, dict)
    
    def test_normalize_term_basic_terms(self):
        """기본 용어 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer.normalize_term("민법") == "민법"
        assert normalizer.normalize_term("형법") == "형법"
        assert normalizer.normalize_term("상법") == "상법"
    
    def test_normalize_term_article_patterns(self):
        """조문 패턴 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        result = normalizer.normalize_term("제750조")
        assert "제" in result
        assert "조" in result
    
    def test_normalize_term_court_patterns(self):
        """법원 패턴 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer.normalize_term("대법원") == "대법원"
        assert normalizer.normalize_term("고등법원") == "고등법원"
        assert normalizer.normalize_term("지방법원") == "지방법원"
    
    def test_normalize_term_contract_terms(self):
        """계약 관련 용어 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer.normalize_term("계약") == "계약"
        assert normalizer.normalize_term("매매") == "매매"
        assert normalizer.normalize_term("임대차") == "임대차"
    
    def test_normalize_term_damage_terms(self):
        """손해 관련 용어 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer.normalize_term("손해배상") == "손해배상"
        assert normalizer.normalize_term("불법행위") == "불법행위"
    
    def test_normalize_term_empty_string(self):
        """빈 문자열 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        result = normalizer.normalize_term("")
        assert result == ""
    
    def test_normalize_term_unknown_term(self):
        """알 수 없는 용어 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        result = normalizer.normalize_term("알수없는용어")
        assert isinstance(result, str)
    
    def test_normalize_text(self):
        """텍스트 정규화 테스트"""
        normalizer = LegalTermNormalizer()
        
        text = "민법 제750조에 따라 손해배상을 청구할 수 있습니다."
        result = normalizer.normalize_text(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_extract_legal_terms(self):
        """법률 용어 추출 테스트"""
        normalizer = LegalTermNormalizer()
        
        text = "민법 제750조에 따라 손해배상을 청구할 수 있습니다."
        terms = normalizer.extract_legal_terms(text)
        
        assert isinstance(terms, list)
        assert len(terms) > 0
    
    def test_is_legal_term(self):
        """법률 용어 여부 확인 테스트"""
        normalizer = LegalTermNormalizer()
        
        assert normalizer.is_legal_term("민법") is True
        assert normalizer.is_legal_term("형법") is True
        assert normalizer.is_legal_term("일반용어") is False
    
    def test_get_similar_terms(self):
        """유사 용어 검색 테스트"""
        normalizer = LegalTermNormalizer()
        
        similar = normalizer.get_similar_terms("민법")
        assert isinstance(similar, list)
    
    def test_get_term_frequency(self):
        """용어 빈도 계산 테스트"""
        normalizer = LegalTermNormalizer()
        
        text = "민법 제750조에 따라 손해배상을 청구할 수 있습니다. 민법은 중요한 법률입니다."
        frequency = normalizer.get_term_frequency(text)
        
        assert isinstance(frequency, dict)
        assert "민법" in frequency
    
    def test_validate_legal_document(self):
        """법률 문서 유효성 검사 테스트"""
        normalizer = LegalTermNormalizer()
        
        text = "민법 제750조에 따라 손해배상을 청구할 수 있습니다."
        result = normalizer.validate_legal_document(text)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "issues" in result
        assert "suggestions" in result
        assert "term_count" in result
        assert "normalized_text" in result

