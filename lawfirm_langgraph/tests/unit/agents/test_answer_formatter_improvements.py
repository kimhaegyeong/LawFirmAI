# -*- coding: utf-8 -*-
"""
Answer Formatter 개선 사항 테스트
sources 변환률, legal_references 생성, sources_detail 생성 등 개선 사항 검증
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from lawfirm_langgraph.core.agents.handlers.answer_formatter import AnswerFormatterHandler
from lawfirm_langgraph.core.agents.state_definitions import LegalWorkflowState


class TestAnswerFormatterImprovements:
    """Answer Formatter 개선 사항 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.formatter = AnswerFormatterHandler(logger=Mock())
    
    def test_restore_retrieved_docs_enhanced(self):
        """retrieved_docs 복구 로직 테스트"""
        state = {
            "retrieved_docs": [],
            "search": {
                "retrieved_docs": [
                    {"id": 1, "content": "test1", "type": "statute_article"},
                    {"id": 2, "content": "test2", "type": "case_paragraph"}
            ]
            }
        }
        
        result = self.formatter._restore_retrieved_docs_enhanced(state)
        
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert state["retrieved_docs"] == result  # 표준화 확인
    
    def test_restore_query_type_enhanced(self):
        """query_type 복구 로직 테스트"""
        state = {
            "query_type": "",
            "classification": {
                "query_type": "legal_advice"
            }
        }
        
        result = self.formatter._restore_query_type_enhanced(state)
        
        assert result == "legal_advice"
        assert state["query_type"] == "legal_advice"
    
    def test_restore_query_type_default(self):
        """query_type 기본값 테스트"""
        state = {
            "query_type": ""
        }
        
        result = self.formatter._restore_query_type_enhanced(state)
        
        assert result == "general_question"
        assert state["query_type"] == "general_question"
    
    def test_source_type_inference_from_content(self):
        """content 기반 source_type 추론 테스트"""
        state = {
            "retrieved_docs": [
                {
                    "content": "민법 제750조에 따르면...",
                    "type": None
                },
                {
                    "content": "대법원 2020다12345 판결에서...",
                    "type": None
                }
            ]
        }
        
        # prepare_final_response_part 호출 시 source_type 추론 확인
        # 실제 테스트는 통합 테스트에서 수행
    
    def test_sources_detail_creation_guarantee(self):
        """sources_detail 생성 보장 테스트"""
        state = {
            "retrieved_docs": [
                {
                    "id": 1,
                    "content": "test content",
                    "type": "statute_article",
                    "statute_name": "민법",
                    "article_no": "750"
                }
            ],
            "sources": [],
            "sources_detail": []
        }
        
        # prepare_final_response_part 호출
        self.formatter.prepare_final_response_part(state, "moderate", True)
        
        # sources_detail이 생성되었는지 확인
        assert len(state.get("sources_detail", [])) > 0
    
    def test_legal_references_extraction_from_content(self):
        """content에서 legal_references 추출 테스트"""
        state = {
            "retrieved_docs": [
                {
                    "content": "민법 제750조에 따르면 손해배상책임이 발생합니다.",
                    "type": "statute_article"
                }
            ],
            "legal_references": []
        }
        
        # prepare_final_response_part 호출
        self.formatter.prepare_final_response_part(state, "moderate", True)
        
        # legal_references가 생성되었는지 확인
        legal_refs = state.get("legal_references", [])
        assert len(legal_refs) > 0 or any("민법" in str(ref) for ref in legal_refs)
    
    def test_sources_normalization(self):
        """sources 정규화 테스트"""
        state = {
            "sources": [
                {"source": "민법 제750조"},
                "대법원 2020다12345",
                {"name": "판례"}
            ],
            "retrieved_docs": []
        }
        
        # prepare_final_response_part 호출
        self.formatter.prepare_final_response_part(state, "moderate", True)
        
        # sources가 문자열 리스트로 정규화되었는지 확인
        sources = state.get("sources", [])
        assert all(isinstance(s, str) for s in sources)
    
    def test_answer_length_minimum_warning(self):
        """답변 최소 길이 경고 테스트"""
        answer = "짧은 답변"
        query_type = "legal_analysis"
        query_complexity = "moderate"
        
        result = self.formatter._adjust_answer_length(
            answer, query_type, query_complexity
        )
        
        # 최소 길이 미만인 경우 경고 로깅 확인
        assert result == answer  # 너무 짧은 경우 그대로 반환
    
    def test_coverage_calculation_improvement(self):
        """coverage 계산 개선 테스트"""
        from lawfirm_langgraph.core.generation.validators.quality_validators import ContextValidator
        
        context_text = "민법 제750조에 따르면 손해배상책임이 발생합니다. 대법원 판례도 참고할 수 있습니다."
        extracted_keywords = ["손해배상", "민법", "책임"]
        legal_references = ["민법 제750조"]
        citations = []
        
        coverage = ContextValidator.calculate_coverage(
            context_text, extracted_keywords, legal_references, citations
        )
        
        # 개선된 coverage 계산 (부분 일치, 컨텍스트 길이 등 고려)
        assert 0.0 <= coverage <= 1.0
        assert coverage > 0.5  # 키워드와 법률 참조가 포함되어 있으므로 높은 점수 예상

