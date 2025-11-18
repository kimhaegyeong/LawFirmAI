# -*- coding: utf-8 -*-
"""
Reasoning Extractor 테스트
langgraph_core/processing/reasoning_extractor.py 및 data/reasoning_extractor.py 단위 테스트
"""

import pytest
from unittest.mock import Mock, patch

from lawfirm_langgraph.langgraph_core.processing.reasoning_extractor import ReasoningExtractor


class TestReasoningExtractor:
    """ReasoningExtractor 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        extractor = ReasoningExtractor()
        
        assert extractor.logger is not None
        assert hasattr(extractor, '_compiled_reasoning_patterns')
        assert hasattr(extractor, '_compiled_output_patterns')
        assert hasattr(extractor, '_compiled_answer_patterns')
    
    def test_init_with_logger(self):
        """로거 포함 초기화 테스트"""
        mock_logger = Mock()
        extractor = ReasoningExtractor(logger=mock_logger)
        
        assert extractor.logger == mock_logger
    
    def test_extract_reasoning_with_reasoning_section(self):
        """추론 과정 섹션이 있는 응답 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = """## 🧠 추론 과정
### Step 1: 문제 분석
계약 해지에 대한 질문입니다.

## 📤 출력
계약 해지는 다음과 같이 가능합니다."""
        
        result = extractor.extract_reasoning(response)
        
        assert isinstance(result, dict)
        assert "reasoning" in result or "answer" in result
    
    def test_extract_reasoning_without_reasoning_section(self):
        """추론 과정 섹션이 없는 응답 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = "계약 해지는 다음과 같이 가능합니다."
        
        result = extractor.extract_reasoning(response)
        
        assert isinstance(result, dict)
    
    def test_extract_answer_with_output_section(self):
        """출력 섹션이 있는 답변 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = """## 📤 출력
계약 해지는 다음과 같이 가능합니다."""
        
        result = extractor.extract_actual_answer(response)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_extract_answer_without_section(self):
        """섹션이 없는 답변 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = "계약 해지는 다음과 같이 가능합니다."
        
        result = extractor.extract_actual_answer(response)
        
        assert isinstance(result, str)
    
    def test_validate_answer_quality(self):
        """답변 품질 검증 테스트"""
        extractor = ReasoningExtractor()
        
        original_answer = "## 🧠 추론 과정\n### Step 1: 분석\n계약 해지는 계약서에 명시된 조건에 따라 가능합니다. 민법 제543조에 따르면 계약 해제가 가능합니다."
        actual_answer = "계약 해지는 계약서에 명시된 조건에 따라 가능합니다. 민법 제543조에 따르면 계약 해제가 가능합니다."
        reasoning_info = extractor.extract_reasoning(original_answer)
        
        result = extractor.verify_extraction_quality(original_answer, actual_answer, reasoning_info)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "score" in result
    
    def test_clean_reasoning_markers(self):
        """추론 마커 정리 테스트"""
        extractor = ReasoningExtractor()
        
        text = "## 🧠 추론 과정\n### Step 1: 분석\n내용"
        
        result = extractor.clean_reasoning_keywords(text)
        
        assert isinstance(result, str)

