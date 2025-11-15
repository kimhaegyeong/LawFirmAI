# -*- coding: utf-8 -*-
"""
Data Reasoning Extractor 테스트
langgraph_core/data/reasoning_extractor.py 단위 테스트
"""

import pytest
from unittest.mock import Mock

from lawfirm_langgraph.langgraph_core.data.reasoning_extractor import ReasoningExtractor


class TestDataReasoningExtractor:
    """Data ReasoningExtractor 테스트"""
    
    def test_init(self):
        """초기화 테스트"""
        extractor = ReasoningExtractor()
        
        assert extractor.logger is not None
        assert hasattr(extractor, '_compiled_reasoning_patterns')
    
    def test_extract_reasoning_basic(self):
        """기본 추론 과정 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = """## 🧠 추론 과정
### Step 1: 분석
계약 해지에 대한 질문입니다.

## 📤 출력
계약 해지는 다음과 같이 가능합니다."""
        
        result = extractor.extract_reasoning(response)
        
        assert isinstance(result, dict)
    
    def test_extract_answer_basic(self):
        """기본 답변 추출 테스트"""
        extractor = ReasoningExtractor()
        
        response = "계약 해지는 다음과 같이 가능합니다."
        
        result = extractor.extract_answer(response)
        
        assert isinstance(result, str)
        assert len(result) > 0

