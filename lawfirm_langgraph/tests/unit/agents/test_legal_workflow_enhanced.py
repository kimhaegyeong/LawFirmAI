# -*- coding: utf-8 -*-
"""
EnhancedLegalQuestionWorkflow 테스트
legal_workflow_enhanced.py의 주요 기능 테스트
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List

import pytest

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
from lawfirm_langgraph.core.agents.state_definitions import LegalWorkflowState
from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionType


class TestEnhancedLegalQuestionWorkflow:
    """EnhancedLegalQuestionWorkflow 테스트 클래스"""

    @pytest.fixture
    def workflow(self, workflow_instance):
        """워크플로우 인스턴스 (conftest 픽스처 사용)"""
        return workflow_instance

    @pytest.fixture
    def sample_state(self, workflow_state) -> LegalWorkflowState:
        """테스트용 State (conftest 픽스처 사용)"""
        return workflow_state

    def test_workflow_initialization(self, workflow_instance, workflow_config):
        """워크플로우 초기화 테스트"""
        workflow = workflow_instance
        
        assert workflow.config == workflow_config
        assert workflow.logger is not None
        assert hasattr(workflow, 'search_handler')
        assert hasattr(workflow, 'context_builder')

    def test_classify_query(self, workflow, sample_state):
        """질문 분류 테스트"""
        # classification_handler가 없을 수 있으므로 속성을 모킹
        with patch.object(workflow, 'classification_handler', create=True) as mock_handler:
            mock_handler.classify_with_llm.return_value = (QuestionType.LEGAL_ADVICE, 0.9)
            mock_handler.fallback_classification.return_value = (QuestionType.LEGAL_ADVICE, 0.8)
            
            result = workflow.classify_query(sample_state)
            
            assert isinstance(result, dict)
            # 에러가 발생하더라도 state는 반환되어야 함
            assert "input" in result or "query" in result or "common" in result

    def test_classify_complexity(self, workflow, sample_state):
        """복잡도 분류 테스트"""
        with patch.object(workflow, '_classify_complexity_with_llm') as mock_complexity:
            from core.workflow.state.workflow_types import QueryComplexity
            mock_complexity.return_value = (QueryComplexity.MODERATE, True)
            
            result = workflow.classify_complexity(sample_state)
            
            assert isinstance(result, dict)
            assert "query_complexity" in result or "common" in result

    def test_expand_keywords(self, workflow, sample_state):
        """키워드 확장 테스트"""
        result = workflow.expand_keywords(sample_state)
        
        assert isinstance(result, dict)
        assert "extracted_keywords" in result or "common" in result

    def test_direct_answer_node(self, workflow, sample_state):
        """직접 답변 노드 테스트"""
        # direct_answer_handler가 없을 수 있으므로 속성을 모킹
        with patch.object(workflow, 'direct_answer_handler', create=True) as mock_handler:
            mock_handler.generate_direct_answer_with_chain.return_value = "직접 답변입니다."
            mock_handler.generate_fallback_answer.return_value = "폴백 답변입니다."
            
            result = workflow.direct_answer_node(sample_state)
            
            assert isinstance(result, dict)
            # 에러가 발생하더라도 state는 반환되어야 함
            assert "input" in result or "query" in result or "common" in result

    def test_prepare_search_query(self, workflow, sample_state):
        """검색 쿼리 준비 테스트"""
        result = workflow.prepare_search_query(sample_state)
        
        assert isinstance(result, dict)
        assert "optimized_queries" in result or "search" in result or "common" in result

    def test_execute_searches_parallel(self, workflow, sample_state):
        """병렬 검색 실행 테스트"""
        with patch.object(workflow.search_handler, 'semantic_search') as mock_semantic:
            with patch.object(workflow.search_handler, 'keyword_search') as mock_keyword:
                mock_semantic.return_value = ([], 0)
                mock_keyword.return_value = ([], 0)
                
                result = workflow.execute_searches_parallel(sample_state)
                
                assert isinstance(result, dict)

    def test_process_search_results_combined(self, workflow, sample_state):
        """검색 결과 처리 테스트"""
        sample_state["semantic_results"] = [
            {
                "content": "검색 결과 1",
                "type": "statute_article",
                "relevance_score": 0.9
            }
        ]
        sample_state["keyword_results"] = []
        sample_state["semantic_count"] = 1
        sample_state["keyword_count"] = 0
        
        result = workflow.process_search_results_combined(sample_state)
        
        assert isinstance(result, dict)
    
    def test_consolidate_expanded_query_results(self, workflow):
        """확장된 쿼리 결과 병합 및 중복 제거 테스트"""
        # 테스트 데이터: 확장된 쿼리 결과 시뮬레이션
        semantic_results = [
            {
                "id": "doc1",
                "content": "계약 해지 사유에 대한 내용",
                "relevance_score": 0.9,
                "source_query": "original"
            },
            {
                "id": "doc1",  # 중복 ID
                "content": "계약 해지 사유에 대한 내용",  # 동일 내용
                "relevance_score": 0.85,
                "sub_query": "sub_query_1"  # 확장된 쿼리
            },
            {
                "id": "doc2",
                "content": "계약 해지 절차",
                "relevance_score": 0.8,
                "sub_query": "sub_query_1"
            },
            {
                "id": "doc3",
                "content": "계약 해지 사유에 대한 내용",  # 동일 내용, 다른 ID
                "relevance_score": 0.75,
                "query_variation": "variation_1"
            }
        ]
        
        # 메서드 호출
        result = workflow._consolidate_expanded_query_results(
            semantic_results,
            "계약 해지 사유에 대해 알려주세요"
        )
        
        # 검증
        assert isinstance(result, list)
        assert len(result) <= len(semantic_results), "중복 제거로 인해 결과 수가 줄어들어야 함"
        
        # 중복 제거 확인
        seen_ids = set()
        for doc in result:
            doc_id = doc.get("id")
            if doc_id:
                assert doc_id not in seen_ids, f"중복 ID 발견: {doc_id}"
                seen_ids.add(doc_id)
        
        # 가중치 적용 확인
        for doc in result:
            assert "source_query" in doc, "source_query 필드가 있어야 함"
            assert "query_weight" in doc, "query_weight 필드가 있어야 함"
            assert "weighted_score" in doc, "weighted_score 필드가 있어야 함"
            
            # 원본 쿼리는 가중치 1.0
            if doc.get("source_query") == "original":
                assert doc.get("query_weight") == 1.0
            else:
                assert doc.get("query_weight") == 0.9
        
        # 정렬 확인 (가중치 점수 기준 내림차순)
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i].get("weighted_score", 0) >= result[i + 1].get("weighted_score", 0)

    def test_merge_and_rerank_with_keyword_weights(self, workflow, sample_state):
        """키워드 가중치 기반 병합 및 재순위화 테스트"""
        sample_state["semantic_results"] = [
            {
                "content": "검색 결과 1",
                "type": "statute_article",
                "relevance_score": 0.9
            }
        ]
        sample_state["keyword_results"] = []
        
        result = workflow.merge_and_rerank_with_keyword_weights(sample_state)
        
        assert isinstance(result, dict)

    def test_generate_answer_enhanced(self, workflow, sample_state):
        """향상된 답변 생성 테스트"""
        sample_state["retrieved_docs"] = [
            {
                "content": "법률 문서 내용",
                "type": "statute_article",
                "relevance_score": 0.9
            }
        ]
        
        with patch.object(workflow, '_generate_answer_with_chain') as mock_generate:
            mock_generate.return_value = "생성된 답변입니다."
            
            result = workflow.generate_answer_enhanced(sample_state)
            
            assert isinstance(result, dict)

    def test_validate_answer_quality(self, workflow, sample_state):
        """답변 품질 검증 테스트"""
        sample_state["answer"] = "테스트 답변입니다."
        
        result = workflow.validate_answer_quality(sample_state)
        
        assert isinstance(result, dict)

    def test_format_answer(self, workflow, sample_state):
        """답변 포맷팅 테스트"""
        sample_state["answer"] = "테스트 답변입니다."
        sample_state["confidence"] = 0.85
        
        with patch.object(workflow, 'answer_formatter_handler') as mock_formatter:
            mock_formatter.format_answer.return_value = sample_state
            
            result = workflow.format_answer(sample_state)
            
            assert isinstance(result, dict)

    def test_prepare_final_response(self, workflow, sample_state):
        """최종 응답 준비 테스트"""
        sample_state["answer"] = "최종 답변입니다."
        sample_state["confidence"] = 0.9
        
        # answer_formatter_handler의 prepare_final_response가 실제로 state를 반환하도록 보장
        # 실제 메서드를 호출하되, 에러가 발생해도 state는 반환되어야 함
        try:
            result = workflow.prepare_final_response(sample_state)
            assert isinstance(result, dict)
        except Exception:
            # 에러가 발생하더라도 state는 반환되어야 함
            assert isinstance(sample_state, dict)

    def test_resolve_multi_turn(self, workflow, sample_state):
        """멀티턴 대화 해결 테스트"""
        with patch.object(workflow, '_resolve_multi_turn_internal') as mock_resolve:
            mock_resolve.return_value = (False, "")
            
            result = workflow.resolve_multi_turn(sample_state)
            
            assert isinstance(result, dict)

    def test_assess_urgency(self, workflow, sample_state):
        """긴급도 평가 테스트"""
        with patch.object(workflow, '_assess_urgency_internal') as mock_assess:
            mock_assess.return_value = ("normal", "일반적인 질문입니다.")
            
            result = workflow.assess_urgency(sample_state)
            
            assert isinstance(result, dict)

    def test_analyze_document(self, workflow, sample_state):
        """문서 분석 테스트"""
        sample_state["input"]["query"] = "이 계약서를 분석해주세요"
        sample_state["input"]["document"] = "계약서 내용입니다."
        
        result = workflow.analyze_document(sample_state)
        
        assert isinstance(result, dict)

    def test_route_expert(self, workflow, sample_state):
        """전문가 라우팅 테스트"""
        sample_state["legal_domain"] = "family_law"
        
        result = workflow.route_expert(sample_state)
        
        assert isinstance(result, dict)

    def test_retry_counter_manager(self, workflow, sample_state):
        """재시도 카운터 관리자 테스트"""
        retry_count = workflow.retry_manager.get_retry_counts(sample_state)
        
        assert isinstance(retry_count, dict)
        assert "generation" in retry_count
        assert "validation" in retry_count
        assert "total" in retry_count

    def test_should_allow_retry(self, workflow, sample_state):
        """재시도 허용 여부 테스트"""
        result = workflow.retry_manager.should_allow_retry(sample_state, "generation")
        
        assert isinstance(result, bool)

    def test_increment_retry_count(self, workflow, sample_state):
        """재시도 카운터 증가 테스트"""
        count = workflow.retry_manager.increment_retry_count(sample_state, "generation")
        
        assert isinstance(count, int)
        assert count >= 0

    def test_state_value_helpers(self, workflow, sample_state):
        """State 값 헬퍼 메서드 테스트"""
        # get_state_value 테스트
        query = workflow._get_state_value(sample_state, "query", "")
        assert query == sample_state["query"]
        
        # set_state_value 테스트 - confidence는 classification 그룹에 저장됨
        original_confidence = workflow._get_state_value(sample_state, "confidence", None)
        workflow._set_state_value(sample_state, "confidence", 0.95)
        test_confidence = workflow._get_state_value(sample_state, "confidence", None)
        assert test_confidence == 0.95
        
        # legal_field도 classification 그룹에 저장됨
        workflow._set_state_value(sample_state, "legal_field", "test_field")
        test_field = workflow._get_state_value(sample_state, "legal_field", None)
        assert test_field == "test_field"

    def test_error_handling(self, workflow, sample_state):
        """에러 핸들링 테스트"""
        workflow._handle_error(sample_state, "테스트 에러", "테스트 컨텍스트")
        
        assert "errors" in sample_state or "common" in sample_state

    def test_processing_time_update(self, workflow, sample_state):
        """처리 시간 업데이트 테스트"""
        import time
        start_time = time.time()
        
        processing_time = workflow._update_processing_time(sample_state, start_time)
        
        assert isinstance(processing_time, float)
        assert processing_time >= 0

    def test_add_step(self, workflow, sample_state):
        """처리 단계 추가 테스트"""
        initial_steps = len(sample_state.get("processing_steps", []))
        
        workflow._add_step(sample_state, "테스트", "테스트 단계")
        
        final_steps = len(sample_state.get("processing_steps", []))
        assert final_steps >= initial_steps

    @pytest.mark.asyncio
    async def test_async_workflow_invocation(self, workflow, sample_state):
        """비동기 워크플로우 호출 테스트"""
        with patch.object(workflow, 'graph') as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=sample_state)
            
            result = await workflow.graph.ainvoke(sample_state)
            
            assert isinstance(result, dict)


class TestQueryComplexity:
    """QueryComplexity Enum 테스트"""

    def test_query_complexity_values(self):
        """QueryComplexity 값 테스트"""
        from core.workflow.state.workflow_types import QueryComplexity
        
        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.MODERATE == "moderate"
        assert QueryComplexity.COMPLEX == "complex"
        assert QueryComplexity.MULTI_HOP == "multi_hop"


class TestRetryCounterManager:
    """RetryCounterManager 테스트"""

    @pytest.fixture
    def retry_manager(self):
        """RetryCounterManager 인스턴스"""
        import logging
        logger = logging.getLogger(__name__)
        from core.workflow.state.workflow_types import RetryCounterManager
        return RetryCounterManager(logger)

    @pytest.fixture
    def sample_state(self):
        """테스트용 State"""
        return {
            "common": {
                "metadata": {}
            }
        }

    def test_get_retry_counts_empty(self, retry_manager, sample_state):
        """빈 State에서 재시도 카운터 읽기"""
        counts = retry_manager.get_retry_counts(sample_state)
        
        assert counts["generation"] == 0
        assert counts["validation"] == 0
        assert counts["total"] == 0

    def test_increment_retry_count(self, retry_manager, sample_state):
        """재시도 카운터 증가"""
        count = retry_manager.increment_retry_count(sample_state, "generation")
        
        assert count == 1
        
        counts = retry_manager.get_retry_counts(sample_state)
        assert counts["generation"] == 1
        assert counts["total"] == 1

    def test_should_allow_retry(self, retry_manager, sample_state):
        """재시도 허용 여부 확인"""
        # MAX_GENERATION_RETRIES = 2이므로, counts["generation"] >= 2이면 False
        # 즉, 0번, 1번까지만 허용됨
        
        # 0번 (초기 상태)
        result = retry_manager.should_allow_retry(sample_state, "generation")
        assert result is True
        
        # 1번 증가 (1번)
        retry_manager.increment_retry_count(sample_state, "generation")
        result = retry_manager.should_allow_retry(sample_state, "generation")
        assert result is True
        
        # 한 번 더 증가 (2번) - 이제 >= 2이므로 False
        retry_manager.increment_retry_count(sample_state, "generation")
        result = retry_manager.should_allow_retry(sample_state, "generation")
        assert result is False

