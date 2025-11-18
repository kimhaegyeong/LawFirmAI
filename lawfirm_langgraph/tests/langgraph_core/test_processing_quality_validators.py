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
    
    def test_calculate_relevance_edge_cases(self):
        """관련성 계산 엣지 케이스 테스트"""
        empty_context = ""
        query = "계약 해지 방법"
        relevance_empty = ContextValidator.calculate_relevance(empty_context, query)
        assert relevance_empty == 0.0
        
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        def semantic_calc(q: str, c: str) -> float:
            return 0.85
        relevance_semantic = ContextValidator.calculate_relevance(
            context_text, query, semantic_calculator=semantic_calc
        )
        assert relevance_semantic == 0.85
        
        def semantic_calc_error(q: str, c: str) -> float:
            raise Exception("Test error")
        relevance_error = ContextValidator.calculate_relevance(
            context_text, query, semantic_calculator=semantic_calc_error
        )
        assert isinstance(relevance_error, float)
        assert 0.0 <= relevance_error <= 1.0
    
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
    
    def test_calculate_coverage_variations(self):
        """커버리지 계산 다양한 케이스 테스트"""
        context_text = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다."
        
        coverage_empty = ContextValidator.calculate_coverage("", [], [], [])
        assert coverage_empty == 0.0
        
        coverage_keywords = ContextValidator.calculate_coverage(
            context_text, ["계약", "해지", "조건"], [], []
        )
        assert isinstance(coverage_keywords, float)
        assert coverage_keywords > 0.0
        
        coverage_references = ContextValidator.calculate_coverage(
            context_text, [], ["민법 제543조", "민법 제544조"], []
        )
        assert isinstance(coverage_references, float)
        assert coverage_references > 0.0

