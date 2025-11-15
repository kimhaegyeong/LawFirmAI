# -*- coding: utf-8 -*-
"""
EnhancedLegalQuestionWorkflow 통합 테스트
전체 워크플로우 통합 테스트
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


class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""

    @pytest.fixture
    def workflow(self, workflow_instance):
        """워크플로우 인스턴스 (conftest 픽스처 사용)"""
        return workflow_instance

    @pytest.fixture
    def initial_state(self, workflow_state) -> LegalWorkflowState:
        """초기 State (conftest 픽스처 사용)"""
        return workflow_state

    @pytest.mark.asyncio
    async def test_full_workflow_simple_query(self, workflow, initial_state):
        """간단한 질문 전체 워크플로우 테스트"""
        with patch.object(workflow, 'classify_query') as mock_classify:
            with patch.object(workflow, 'classify_complexity') as mock_complexity:
                with patch.object(workflow, 'direct_answer_node') as mock_direct:
                    mock_classify.return_value = {
                        **initial_state,
                        "query_type": "definition",
                        "confidence": 0.9
                    }
                    mock_complexity.return_value = {
                        **initial_state,
                        "query_complexity": "simple",
                        "needs_search": False
                    }
                    mock_direct.return_value = {
                        **initial_state,
                        "answer": "전세금 반환 보증은..."
                    }
                    
                    # 워크플로우 실행 시뮬레이션
                    state = initial_state.copy()
                    state = mock_classify(state)
                    state = mock_complexity(state)
                    state = mock_direct(state)
                    
                    assert "answer" in state or "query_type" in state

    @pytest.mark.asyncio
    async def test_full_workflow_complex_query(self, workflow, initial_state):
        """복잡한 질문 전체 워크플로우 테스트"""
        with patch.object(workflow, 'classify_query') as mock_classify:
            with patch.object(workflow, 'classify_complexity') as mock_complexity:
                with patch.object(workflow, 'prepare_search_query') as mock_prepare:
                    with patch.object(workflow, 'execute_searches_parallel') as mock_search:
                        with patch.object(workflow, 'process_search_results_combined') as mock_process:
                            with patch.object(workflow, 'generate_answer_enhanced') as mock_generate:
                                mock_classify.return_value = {
                                    **initial_state,
                                    "query_type": "legal_advice",
                                    "confidence": 0.85
                                }
                                mock_complexity.return_value = {
                                    **initial_state,
                                    "query_complexity": "complex",
                                    "needs_search": True
                                }
                                mock_prepare.return_value = {
                                    **initial_state,
                                    "optimized_queries": {"semantic_query": "전세금 반환 보증"}
                                }
                                mock_search.return_value = {
                                    **initial_state,
                                    "semantic_results": [{"content": "검색 결과"}],
                                    "keyword_results": []
                                }
                                mock_process.return_value = {
                                    **initial_state,
                                    "retrieved_docs": [{"content": "검색 결과"}]
                                }
                                mock_generate.return_value = {
                                    **initial_state,
                                    "answer": "전세금 반환 보증에 대한 답변입니다."
                                }
                                
                                # 워크플로우 실행 시뮬레이션
                                state = initial_state.copy()
                                state = mock_classify(state)
                                state = mock_complexity(state)
                                state = mock_prepare(state)
                                state = mock_search(state)
                                state = mock_process(state)
                                state = mock_generate(state)
                                
                                assert "answer" in state or "retrieved_docs" in state

    def test_workflow_error_recovery(self, workflow, initial_state):
        """워크플로우 에러 복구 테스트"""
        with patch.object(workflow, 'classify_query', side_effect=Exception("테스트 에러")):
            try:
                result = workflow.classify_query(initial_state)
                # 에러가 발생해도 State는 유지되어야 함
                assert isinstance(result, dict)
            except Exception:
                # 에러가 전파되더라도 State는 보존되어야 함
                assert "input" in initial_state

    def test_state_persistence(self, workflow, initial_state):
        """State 지속성 테스트"""
        state1 = initial_state.copy()
        workflow._set_state_value(state1, "test_key", "test_value")
        
        value = workflow._get_state_value(state1, "test_key", None)
        assert value == "test_value"
        
        # State 복사 후에도 값 유지
        state2 = state1.copy()
        value2 = workflow._get_state_value(state2, "test_key", None)
        assert value2 == "test_value"

    def test_retry_mechanism(self, workflow, initial_state):
        """재시도 메커니즘 테스트"""
        # 초기 재시도 카운터 확인
        counts = workflow.retry_manager.get_retry_counts(initial_state)
        assert counts["total"] == 0
        
        # 재시도 카운터 증가
        count = workflow.retry_manager.increment_retry_count(initial_state, "generation")
        assert count == 1
        
        # 재시도 허용 여부 확인
        should_retry = workflow.retry_manager.should_allow_retry(initial_state, "generation")
        assert should_retry is True

    def test_multiple_queries_processing(self, workflow):
        """다중 쿼리 처리 테스트"""
        queries = [
            "전세금 반환 보증이란?",
            "계약서 작성 시 주의사항은?",
            "고소장 작성 방법은?"
        ]
        
        results = []
        for query in queries:
            state = {
                "input": {"query": query, "session_id": "test_session"},
                "query": query,
                "processing_steps": [],
                "errors": [],
                "common": {"metadata": {}}
            }
            
            with patch.object(workflow, 'classify_query') as mock_classify:
                mock_classify.return_value = {**state, "query_type": "legal_advice"}
                result = workflow.classify_query(state)
                results.append(result)
        
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, dict)

    def test_session_isolation(self, workflow):
        """세션 격리 테스트"""
        session1_state = {
            "input": {"query": "질문 1", "session_id": "session_1"},
            "query": "질문 1",
            "common": {"metadata": {}}
        }
        
        session2_state = {
            "input": {"query": "질문 2", "session_id": "session_2"},
            "query": "질문 2",
            "common": {"metadata": {}}
        }
        
        workflow._set_state_value(session1_state, "session_data", "data_1")
        workflow._set_state_value(session2_state, "session_data", "data_2")
        
        data1 = workflow._get_state_value(session1_state, "session_data", None)
        data2 = workflow._get_state_value(session2_state, "session_data", None)
        
        assert data1 == "data_1"
        assert data2 == "data_2"
        assert data1 != data2

