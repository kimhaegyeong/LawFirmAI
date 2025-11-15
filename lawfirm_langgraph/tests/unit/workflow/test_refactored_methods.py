# -*- coding: utf-8 -*-
"""
리팩토링된 메서드들에 대한 단위 테스트
generate_answer_enhanced, process_search_results_combined, generate_answer_final의 헬퍼 메서드들 테스트
"""

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig


class TestRefactoredHelperMethods:
    """리팩토링된 헬퍼 메서드들 테스트"""
    
    @pytest.fixture
    def workflow(self):
        """워크플로우 인스턴스 생성"""
        config = LangGraphConfig()
        return EnhancedLegalQuestionWorkflow(config)
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """테스트용 State 생성"""
        return {
            "query": "전세금 반환 보증에 대해 설명해주세요",
            "query_type": "legal_advice",
            "retrieved_docs": [
                {
                    "content": "전세금 반환 보증 관련 내용",
                    "type": "statute_article",
                    "source": "주택임대차보호법",
                    "relevance_score": 0.9
                }
            ],
            "semantic_results": [],
            "keyword_results": [],
            "semantic_count": 0,
            "keyword_count": 0,
            "search_params": {},
            "extracted_keywords": ["전세금", "반환", "보증"],
            "legal_field": "부동산법",
            "metadata": {},
            "common": {},
            "search": {}
        }
    
    def test_prepare_search_inputs(self, workflow, sample_state):
        """_prepare_search_inputs 메서드 테스트"""
        result = workflow._prepare_search_inputs(sample_state)
        
        assert isinstance(result, dict)
        assert "semantic_results" in result
        assert "keyword_results" in result
        assert "query" in result
        assert "query_type_str" in result
        assert result["query"] == "전세금 반환 보증에 대해 설명해주세요"
    
    def test_restore_state_data_for_final(self, workflow, sample_state):
        """_restore_state_data_for_final 메서드 테스트"""
        sample_state["retrieved_docs"] = []
        sample_state["query_type"] = None
        
        workflow._restore_state_data_for_final(sample_state)
        
        retrieved_docs = workflow._get_state_value(sample_state, "retrieved_docs", [])
        query_type = workflow._get_state_value(sample_state, "query_type")
        
        assert isinstance(retrieved_docs, list)
        assert query_type is None or isinstance(query_type, str)
    
    def test_perform_conditional_retry_search_no_retry(self, workflow, sample_state):
        """_perform_conditional_retry_search - 재검색 불필요한 경우"""
        quality_evaluation = {
            "semantic_quality": {"needs_retry": False},
            "keyword_quality": {"needs_retry": False},
            "overall_quality": 0.8,
            "needs_retry": False
        }
        
        semantic_results, keyword_results, semantic_count, keyword_count = workflow._perform_conditional_retry_search(
            sample_state, [], [], 0, 0, quality_evaluation, "query", "legal_advice", {}, []
        )
        
        assert isinstance(semantic_results, list)
        assert isinstance(keyword_results, list)
        assert semantic_count == 0
        assert keyword_count == 0
    
    def test_merge_and_rerank_results(self, workflow, sample_state):
        """_merge_and_rerank_results 메서드 테스트"""
        semantic_results = [
            {"content": "Test 1", "type": "statute_article", "relevance_score": 0.9}
        ]
        keyword_results = [
            {"content": "Test 2", "type": "case_paragraph", "relevance_score": 0.8}
        ]
        
        result = workflow._merge_and_rerank_results(
            sample_state, semantic_results, keyword_results, "query"
        )
        
        assert isinstance(result, list)
        assert len(result) >= 0
    
    def test_apply_keyword_weights_and_rerank(self, workflow, sample_state):
        """_apply_keyword_weights_and_rerank 메서드 테스트"""
        merged_docs = [
            {"content": "Test content", "type": "statute_article", "relevance_score": 0.9}
        ]
        
        result = workflow._apply_keyword_weights_and_rerank(
            sample_state, merged_docs, "query", "legal_advice", ["키워드"], {}, 0.7
        )
        
        assert isinstance(result, list)
    
    def test_filter_and_validate_documents(self, workflow, sample_state):
        """_filter_and_validate_documents 메서드 테스트"""
        weighted_docs = [
            {
                "content": "Valid content with sufficient length",
                "type": "statute_article",
                "relevance_score": 0.9,
                "final_weighted_score": 0.9
            },
            {
                "content": "x",  # 너무 짧은 내용
                "type": "statute_article",
                "relevance_score": 0.1
            }
        ]
        
        result = workflow._filter_and_validate_documents(
            sample_state, weighted_docs, "query", ["키워드"], []
        )
        
        assert isinstance(result, list)
        assert len(result) <= len(weighted_docs)
    
    def test_ensure_diversity_and_limit(self, workflow, sample_state):
        """_ensure_diversity_and_limit 메서드 테스트"""
        filtered_docs = [
            {"content": "Test", "type": "statute_article", "relevance_score": 0.9}
        ]
        weighted_docs = filtered_docs.copy()
        merged_docs = filtered_docs.copy()
        
        result = workflow._ensure_diversity_and_limit(
            sample_state, filtered_docs, weighted_docs, merged_docs, "query", "legal_advice", []
        )
        
        assert isinstance(result, list)
    
    def test_validate_and_handle_regeneration(self, workflow, sample_state):
        """_validate_and_handle_regeneration 메서드 테스트"""
        sample_state["answer"] = "Test answer"
        sample_state["needs_regeneration"] = False
        
        result = workflow._validate_and_handle_regeneration(sample_state)
        
        assert isinstance(result, bool)
    
    def test_handle_format_errors(self, workflow, sample_state):
        """_handle_format_errors 메서드 테스트"""
        sample_state["answer"] = "Valid answer without format errors"
        
        result = workflow._handle_format_errors(sample_state, True)
        
        assert isinstance(result, bool)
    
    def test_format_and_finalize(self, workflow, sample_state):
        """_format_and_finalize 메서드 테스트"""
        sample_state["answer"] = "Test answer"
        sample_state["confidence"] = 0.9
        
        workflow._format_and_finalize(sample_state, 0.0)
        
        assert "answer" in sample_state
    
    def test_handle_final_node_error(self, workflow, sample_state):
        """_handle_final_node_error 메서드 테스트"""
        sample_state["answer"] = "Existing answer"
        
        workflow._handle_final_node_error(sample_state, Exception("Test error"))
        
        assert "answer" in sample_state
        assert sample_state.get("legal_validity_check") is True


class TestRefactoredMainMethods:
    """리팩토링된 메인 메서드들 통합 테스트"""
    
    @pytest.fixture
    def workflow(self):
        """워크플로우 인스턴스 생성"""
        config = LangGraphConfig()
        return EnhancedLegalQuestionWorkflow(config)
    
    @pytest.fixture
    def sample_state(self) -> LegalWorkflowState:
        """테스트용 State 생성"""
        return {
            "query": "전세금 반환 보증에 대해 설명해주세요",
            "query_type": "legal_advice",
            "retrieved_docs": [
                {
                    "content": "전세금 반환 보증 관련 내용",
                    "type": "statute_article",
                    "source": "주택임대차보호법",
                    "relevance_score": 0.9
                }
            ],
            "semantic_results": [],
            "keyword_results": [],
            "semantic_count": 0,
            "keyword_count": 0,
            "search_params": {},
            "extracted_keywords": ["전세금", "반환", "보증"],
            "legal_field": "부동산법",
            "metadata": {},
            "common": {},
            "search": {}
        }
    
    def test_generate_answer_final_structure(self, workflow, sample_state):
        """generate_answer_final 메서드 구조 테스트"""
        sample_state["answer"] = "Test answer"
        
        result = workflow.generate_answer_final(sample_state)
        
        assert isinstance(result, dict)
        assert "answer" in result or "query" in result
    
    def test_process_search_results_combined_structure(self, workflow, sample_state):
        """process_search_results_combined 메서드 구조 테스트"""
        sample_state["semantic_results"] = [
            {"content": "Test", "type": "statute_article", "relevance_score": 0.9}
        ]
        sample_state["keyword_results"] = []
        sample_state["semantic_count"] = 1
        sample_state["keyword_count"] = 0
        
        result = workflow.process_search_results_combined(sample_state)
        
        assert isinstance(result, dict)
        assert "retrieved_docs" in result or "query" in result

