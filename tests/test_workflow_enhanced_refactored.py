# -*- coding: utf-8 -*-
"""
Refactored Legal Workflow Tests
리팩토링된 법률 워크플로우 테스트
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
    WorkflowConstants,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.utils.langgraph_config import LangGraphConfig


class TestWorkflowConstants:
    """워크플로우 상수 테스트"""

    def test_constants_defined(self):
        """상수들이 정의되어 있는지 확인"""
        assert hasattr(WorkflowConstants, 'MAX_OUTPUT_TOKENS')
        assert hasattr(WorkflowConstants, 'TEMPERATURE')
        assert hasattr(WorkflowConstants, 'TIMEOUT')
        assert hasattr(WorkflowConstants, 'SEMANTIC_SEARCH_K')
        assert hasattr(WorkflowConstants, 'MAX_DOCUMENTS')
        assert hasattr(WorkflowConstants, 'MAX_RETRIES')
        assert hasattr(WorkflowConstants, 'LLM_CLASSIFICATION_CONFIDENCE')

    def test_constants_values(self):
        """상수값들이 유효한지 확인"""
        assert WorkflowConstants.MAX_OUTPUT_TOKENS > 0
        assert 0 <= WorkflowConstants.TEMPERATURE <= 1
        assert WorkflowConstants.TIMEOUT > 0
        assert WorkflowConstants.SEMANTIC_SEARCH_K > 0
        assert WorkflowConstants.MAX_DOCUMENTS > 0
        assert WorkflowConstants.MAX_RETRIES > 0
        assert 0 <= WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE <= 1


class TestWorkflowInitialization:
    """워크플로우 초기화 테스트"""

    @pytest.fixture
    def mock_config(self):
        """모의 설정 생성"""
        config = LangGraphConfig()
        config.llm_provider = "google"
        config.google_model = "gemini-2.5-flash-lite"
        config.google_api_key = "test-key"
        return config

    def test_workflow_initialization(self, mock_config):
        """워크플로우 초기화 테스트"""
        with patch('source.services.langgraph.legal_workflow_enhanced.SemanticSearchEngine'):
            with patch('source.services.langgraph.legal_workflow_enhanced.LegalKeywordMapper'):
                with patch('source.services.langgraph.legal_workflow_enhanced.LegalDataConnector'):
                    with patch('source.services.langgraph.legal_workflow_enhanced.PerformanceOptimizer'):
                        with patch('source.services.langgraph.legal_workflow_enhanced.TermIntegrator'):
                            with patch('source.services.langgraph.legal_workflow_enhanced.UnifiedPromptManager'):
                                with patch.object(EnhancedLegalQuestionWorkflow, '_initialize_llm'):
                                    workflow = EnhancedLegalQuestionWorkflow(mock_config)
                                    assert workflow is not None
                                    assert workflow.config == mock_config
                                    assert workflow.unified_prompt_manager is not None


class TestHelperMethods:
    """헬퍼 메서드 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 인스턴스 생성"""
        config = Mock(spec=LangGraphConfig)
        config.llm_provider = "google"
        config.google_model = "gemini-2.5-flash-lite"
        config.google_api_key = "test-key"

        with patch('source.services.langgraph.legal_workflow_enhanced.SemanticSearchEngine'):
            with patch('source.services.langgraph.legal_workflow_enhanced.LegalKeywordMapper'):
                with patch('source.services.langgraph.legal_workflow_enhanced.LegalDataConnector'):
                    with patch('source.services.langgraph.legal_workflow_enhanced.PerformanceOptimizer'):
                        with patch('source.services.langgraph.legal_workflow_enhanced.TermIntegrator'):
                            with patch('source.services.langgraph.legal_workflow_enhanced.UnifiedPromptManager'):
                                with patch.object(EnhancedLegalQuestionWorkflow, '_initialize_llm'):
                                    return EnhancedLegalQuestionWorkflow(config)

    def test_get_query_type_str(self, workflow):
        """QueryType 문자열 변환 테스트"""
        from source.services.question_classifier import QuestionType

        query_type = QuestionType.LEGAL_ADVICE
        result = workflow._get_query_type_str(query_type)
        assert result == "legal_advice"

    def test_get_category_mapping(self, workflow):
        """카테고리 매핑 테스트"""
        mapping = workflow._get_category_mapping()
        assert isinstance(mapping, dict)
        assert "precedent_search" in mapping
        assert "law_inquiry" in mapping
        assert "legal_advice" in mapping
        assert len(mapping["precedent_search"]) > 0

    def test_update_processing_time(self, workflow):
        """처리 시간 업데이트 테스트"""
        state = create_initial_legal_state("테스트 질문", "test-session")
        start_time = 100.0

        import time
        with patch('time.time', return_value=105.0):
            processing_time = workflow._update_processing_time(state, start_time)
            assert processing_time == 5.0
            assert state["processing_time"] == 5.0

    def test_add_step(self, workflow):
        """처리 단계 추가 테스트"""
        state = create_initial_legal_state("테스트 질문", "test-session")

        workflow._add_step(state, "테스트", "테스트 메시지")
        assert "테스트 메시지" in state["processing_steps"]

        # 중복 추가 방지 확인
        workflow._add_step(state, "테스트", "테스트 메시지2")
        assert "테스트 메시지2" not in state["processing_steps"]

    def test_handle_error(self, workflow):
        """에러 처리 테스트"""
        state = create_initial_legal_state("테스트 질문", "test-session")

        workflow._handle_error(state, "테스트 에러", "컨텍스트")
        assert "컨텍스트: 테스트 에러" in state["errors"]
        assert "컨텍스트: 테스트 에러" in state["processing_steps"]


class TestClassification:
    """질문 분류 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 인스턴스 생성"""
        config = Mock(spec=LangGraphConfig)
        config.llm_provider = "google"
        config.google_model = "gemini-2.5-flash-lite"
        config.google_api_key = "test-key"

        with patch('source.services.langgraph.legal_workflow_enhanced.SemanticSearchEngine'):
            with patch('source.services.langgraph.legal_workflow_enhanced.LegalKeywordMapper'):
                with patch('source.services.langgraph.legal_workflow_enhanced.LegalDataConnector'):
                    with patch('source.services.langgraph.legal_workflow_enhanced.PerformanceOptimizer'):
                        with patch('source.services.langgraph.legal_workflow_enhanced.TermIntegrator'):
                            with patch('source.services.langgraph.legal_workflow_enhanced.UnifiedPromptManager'):
                                with patch.object(EnhancedLegalQuestionWorkflow, '_initialize_llm'):
                                    return EnhancedLegalQuestionWorkflow(config)

    def test_fallback_classification(self, workflow):
        """폴백 분류 테스트"""
        from source.services.question_classifier import QuestionType

        # 판례 관련 질문
        result = workflow._fallback_classification("판례를 찾고 싶어요")
        assert result[0] == QuestionType.PRECEDENT_SEARCH
        assert result[1] == WorkflowConstants.FALLBACK_CONFIDENCE

        # 법률 관련 질문
        result = workflow._fallback_classification("법률 조문은 무엇인가요")
        assert result[0] == QuestionType.LAW_INQUIRY
        assert result[1] == WorkflowConstants.FALLBACK_CONFIDENCE

        # 절차 관련 질문
        result = workflow._fallback_classification("소송 절차는 어떻게 되나요")
        assert result[0] == QuestionType.PROCEDURE_GUIDE
        assert result[1] == WorkflowConstants.FALLBACK_CONFIDENCE

        # 일반 질문
        result = workflow._fallback_classification("일반적인 질문입니다")
        assert result[0] == QuestionType.GENERAL_QUESTION
        assert result[1] == WorkflowConstants.DEFAULT_CONFIDENCE


class TestSearchMethods:
    """검색 메서드 테스트"""

    @pytest.fixture
    def workflow(self):
        """워크플로우 인스턴스 생성"""
        config = Mock(spec=LangGraphConfig)
        config.llm_provider = "google"
        config.google_model = "gemini-2.5-flash-lite"
        config.google_api_key = "test-key"

        workflow = Mock(spec=EnhancedLegalQuestionWorkflow)
        workflow.logger = Mock()
        workflow.data_connector = Mock()
        workflow.performance_optimizer = Mock()
        workflow.semantic_search = None

        # 실제 인스턴스 메서드들 추가
        workflow._get_category_mapping = EnhancedLegalQuestionWorkflow._get_category_mapping
        workflow._keyword_search = EnhancedLegalQuestionWorkflow._keyword_search.__get__(workflow, EnhancedLegalQuestionWorkflow)
        workflow._merge_search_results = EnhancedLegalQuestionWorkflow._merge_search_results
        workflow._extract_terms_from_documents = EnhancedLegalQuestionWorkflow._extract_terms_from_documents

        return workflow

    def test_merge_search_results(self, workflow):
        """검색 결과 통합 테스트"""
        semantic_results = [
            {'id': 'sem_1', 'content': '내용1', 'relevance_score': 0.9},
            {'id': 'sem_2', 'content': '내용2', 'relevance_score': 0.8}
        ]
        keyword_results = [
            {'id': 'kw_1', 'content': '내용3', 'relevance_score': 0.7}
        ]

        merged = workflow._merge_search_results(semantic_results, keyword_results)
        assert len(merged) == 3
        assert merged[0]['id'] == 'sem_1'  # 높은 점수 순서


class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""

    def test_workflow_complete_graph(self):
        """전체 워크플로우 그래프 구성 테스트"""
        config = LangGraphConfig()
        config.llm_provider = "google"
        config.google_model = "gemini-2.5-flash-lite"
        config.google_api_key = "test-key"

        with patch('source.services.langgraph.legal_workflow_enhanced.SemanticSearchEngine'):
            with patch('source.services.langgraph.legal_workflow_enhanced.LegalKeywordMapper'):
                with patch('source.services.langgraph.legal_workflow_enhanced.LegalDataConnector'):
                    with patch('source.services.langgraph.legal_workflow_enhanced.PerformanceOptimizer'):
                        with patch('source.services.langgraph.legal_workflow_enhanced.TermIntegrator'):
                            with patch('source.services.langgraph.legal_workflow_enhanced.UnifiedPromptManager'):
                                with patch.object(EnhancedLegalQuestionWorkflow, '_initialize_llm'):
                                    workflow = EnhancedLegalQuestionWorkflow(config)

                                    # 그래프 노드 확인
                                    assert workflow.graph is not None
                                    assert hasattr(workflow.graph, 'edges')
                                    assert hasattr(workflow.graph, 'nodes')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
