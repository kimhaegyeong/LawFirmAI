# -*- coding: utf-8 -*-
"""
Processing Quality Validators 테스트
langgraph_core/processing/quality_validators.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from lawfirm_langgraph.langgraph_core.processing.quality_validators import ContextValidator


class TestContextValidator:
    """ContextValidator 테스트"""
    
    def test_calculate_relevance_basic(self):
        """기본 관련성 계산 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        query = "계약 해지 방법"
        
        relevance = ContextValidator.calculate_relevance(context_text, query)
        
        assert isinstance(relevance, float)
        assert 0.0 <= relevance <= 1.0
    
    def test_calculate_relevance_empty_context(self):
        """빈 컨텍스트 관련성 계산 테스트"""
        context_text = ""
        query = "계약 해지 방법"
        
        relevance = ContextValidator.calculate_relevance(context_text, query)
        
        assert relevance == 0.0
    
    def test_calculate_relevance_with_semantic_calculator(self):
        """의미적 계산기 포함 관련성 계산 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        query = "계약 해지 방법"
        
        def semantic_calc(q: str, c: str) -> float:
            return 0.85
        
        relevance = ContextValidator.calculate_relevance(
            context_text, query, semantic_calculator=semantic_calc
        )
        
        assert relevance == 0.85
    
    def test_calculate_relevance_semantic_calculator_exception(self):
        """의미적 계산기 예외 처리 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        query = "계약 해지 방법"
        
        def semantic_calc(q: str, c: str) -> float:
            raise Exception("Test error")
        
        relevance = ContextValidator.calculate_relevance(
            context_text, query, semantic_calculator=semantic_calc
        )
        
        assert isinstance(relevance, float)
        assert 0.0 <= relevance <= 1.0
    
    def test_calculate_coverage_basic(self):
        """기본 커버리지 계산 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        extracted_keywords = ["계약", "해지"]
        legal_references = ["민법 제543조"]
        citations = []
        
        coverage = ContextValidator.calculate_coverage(
            context_text, extracted_keywords, legal_references, citations
        )
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
    
    def test_calculate_coverage_empty(self):
        """빈 입력 커버리지 계산 테스트"""
        context_text = ""
        extracted_keywords = []
        legal_references = []
        citations = []
        
        coverage = ContextValidator.calculate_coverage(
            context_text, extracted_keywords, legal_references, citations
        )
        
        assert coverage == 0.0
    
    def test_calculate_coverage_with_keywords(self):
        """키워드 포함 커버리지 계산 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        extracted_keywords = ["계약", "해지", "조건"]
        legal_references = []
        citations = []
        
        coverage = ContextValidator.calculate_coverage(
            context_text, extracted_keywords, legal_references, citations
        )
        
        assert isinstance(coverage, float)
        assert coverage > 0.0
    
    def test_calculate_coverage_with_references(self):
        """법률 참조 포함 커버리지 계산 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        extracted_keywords = []
        legal_references = ["민법 제543조", "민법 제544조"]
        citations = []
        
        coverage = ContextValidator.calculate_coverage(
            context_text, extracted_keywords, legal_references, citations
        )
        
        assert isinstance(coverage, float)
        assert coverage > 0.0

